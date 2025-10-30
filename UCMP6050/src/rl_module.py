"""
Reinforcement Learning Module for CTIS
Integrates Songxin's RL approach for spatio-temporal optimization
Dynamic route optimization and resource allocation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class DeliveryEnvironment:
    """
    Reinforcement Learning Environment for Last-mile Delivery
    State: Current locations, demand, traffic conditions
    Action: Route selection, resource allocation
    Reward: Delivery efficiency, time savings, customer satisfaction
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.n_nodes = config['n_nodes']
        self.n_vehicles = config.get('n_vehicles', 10)
        self.max_capacity = config.get('max_capacity', 20)
        
        # State space
        self.state_dim = self.n_nodes * 3  # position, demand, traffic
        # Action space
        self.action_dim = self.n_nodes  # next node to visit
        
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_positions = np.random.randint(0, self.n_nodes, self.n_vehicles)
        self.remaining_demand = np.random.rand(self.n_nodes) * 10
        self.traffic_conditions = np.random.rand(self.n_nodes)
        self.time_step = 0
        self.total_delivered = 0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        # One-hot encode positions
        position_state = np.zeros(self.n_nodes)
        for pos in self.current_positions:
            position_state[pos] += 1
        
        # Combine all state components
        state = np.concatenate([
            position_state,
            self.remaining_demand,
            self.traffic_conditions
        ])
        
        return state
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute actions in environment
        Args:
            actions: (n_vehicles,) - Next node for each vehicle
        Returns:
            next_state, reward, done, info
        """
        reward = 0.0
        
        # Update vehicle positions and deliver
        for vehicle_idx, action in enumerate(actions):
            current_pos = self.current_positions[vehicle_idx]
            next_pos = int(action)
            
            # Calculate travel cost (distance + traffic)
            travel_cost = self._calculate_travel_cost(current_pos, next_pos)
            
            # Deliver at next position
            delivered = min(self.remaining_demand[next_pos], self.max_capacity)
            self.remaining_demand[next_pos] -= delivered
            self.total_delivered += delivered
            
            # Update position
            self.current_positions[vehicle_idx] = next_pos
            
            # Calculate reward
            reward += delivered * 10  # Delivery reward
            reward -= travel_cost * 2  # Travel cost penalty
        
        # Time penalty
        self.time_step += 1
        reward -= self.time_step * 0.1
        
        # Check if done
        done = (self.remaining_demand.sum() < 0.1) or (self.time_step >= 100)
        
        next_state = self._get_state()
        
        info = {
            'total_delivered': self.total_delivered,
            'remaining_demand': self.remaining_demand.sum(),
            'time_step': self.time_step
        }
        
        return next_state, reward, done, info
    
    def _calculate_travel_cost(self, from_node: int, to_node: int) -> float:
        """Calculate travel cost considering distance and traffic"""
        # Simplified distance (in practice, use actual distance matrix)
        distance = abs(from_node - to_node)
        traffic_factor = (self.traffic_conditions[from_node] + self.traffic_conditions[to_node]) / 2
        
        return distance * (1 + traffic_factor)


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for learning optimal delivery policies
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim)
        Returns:
            q_values: (B, action_dim)
        """
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class RLOptimizer:
    """
    RL-based optimizer for delivery route optimization
    Uses DQN with experience replay
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize environment
        self.env = DeliveryEnvironment(config)
        
        # Initialize networks
        state_dim = self.env.state_dim
        action_dim = self.env.action_dim
        hidden_dim = config.get('rl_hidden_dim', 256)
        
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=config.get('rl_lr', 1e-4)
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.get('buffer_size', 100000))
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('rl_batch_size', 64)
        self.target_update_freq = config.get('target_update_freq', 10)
        
        self.episode = 0
        
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Select action using epsilon-greedy policy
        """
        if not eval_mode and random.random() < self.epsilon:
            # Random action
            return random.randint(0, self.env.action_dim - 1)
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(dim=1).item()
            return action
    
    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train_episode(self) -> Dict:
        """Train for one episode"""
        state = self.env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        
        while not done:
            # Select and perform action
            action = self.select_action(state)
            actions = np.array([action] * self.env.n_vehicles)  # All vehicles use same policy
            
            next_state, reward, done, info = self.env.step(actions)
            
            # Store transition
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train
            loss = self.train_step()
            if loss is not None:
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
        
        # Update target network
        if self.episode % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.episode += 1
        
        return {
            'episode': self.episode,
            'reward': episode_reward,
            'loss': np.mean(episode_loss) if episode_loss else 0,
            'epsilon': self.epsilon,
            'delivered': info['total_delivered']
        }
    
    def optimize_routes(self, 
                       current_state: Dict,
                       prediction: np.ndarray) -> List[List[int]]:
        """
        Optimize delivery routes based on current state and demand prediction
        Args:
            current_state: Current system state
            prediction: (N, T) - Predicted demand for next T timesteps
        Returns:
            routes: List of optimized routes for each vehicle
        """
        # Use trained policy to generate routes
        state = self._state_from_prediction(current_state, prediction)
        
        routes = []
        for _ in range(self.env.n_vehicles):
            route = []
            current_pos = 0
            visited = set([current_pos])
            
            for _ in range(10):  # Max route length
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_tensor)[0]
                    
                    # Mask visited nodes
                    for v in visited:
                        q_values[v] = float('-inf')
                    
                    next_node = q_values.argmax().item()
                    route.append(next_node)
                    visited.add(next_node)
                    current_pos = next_node
                    
                    if len(visited) >= self.env.n_nodes:
                        break
            
            routes.append(route)
        
        return routes
    
    def _state_from_prediction(self, current_state: Dict, prediction: np.ndarray) -> np.ndarray:
        """Convert prediction to state representation"""
        # Simplified state construction
        demand = prediction[:, 0]  # Use first timestep prediction
        traffic = current_state.get('traffic', np.random.rand(self.env.n_nodes))
        position = np.zeros(self.env.n_nodes)
        position[0] = 1  # Start position
        
        state = np.concatenate([position, demand, traffic])
        return state

