"""
Training script for CTIS model
Integrates all components: forecasting, RL, RAG
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import json
import os
from tqdm import tqdm
import wandb

from models import create_model
from data_loader import get_dataloader
from rag_module import CTISRAGSystem
from rl_module import RLOptimizer


class CTISTrainer:
    """
    Main trainer for CTIS system
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = create_model(config).to(self.device)
        
        # Initialize RAG system
        self.rag_system = CTISRAGSystem(config)
        
        # Initialize RL optimizer
        self.rl_optimizer = RLOptimizer(config)
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('lr', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Metrics
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            # Move data to device
            x = batch['x'].to(self.device)  # (B, N, T_in, F)
            y = batch['y'].to(self.device)  # (B, N, T_out, F)
            adj = batch['adj'].to(self.device)
            node_features = batch['node_features'].to(self.device)
            
            # Forward pass
            pred = self.model(x, adj, node_features)
            
            # Compute loss
            loss = self.criterion(pred, y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> dict:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_rmse = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)
                adj = batch['adj'].to(self.device)
                node_features = batch['node_features'].to(self.device)
                
                # Forward pass
                pred = self.model(x, adj, node_features)
                
                # Compute metrics
                loss = self.criterion(pred, y)
                mae = torch.abs(pred - y).mean()
                rmse = torch.sqrt(torch.pow(pred - y, 2).mean())
                
                total_loss += loss.item()
                total_mae += mae.item()
                total_rmse += rmse.item()
                num_batches += 1
        
        metrics = {
            'val_loss': total_loss / num_batches,
            'val_mae': total_mae / num_batches,
            'val_rmse': total_rmse / num_batches
        }
        
        return metrics
    
    def train_rl_component(self, num_episodes: int = 100):
        """Train RL component for route optimization"""
        print("\n=== Training RL Component ===")
        
        for episode in range(num_episodes):
            result = self.rl_optimizer.train_episode()
            
            if episode % 10 == 0:
                print(f"Episode {episode}: Reward={result['reward']:.2f}, "
                      f"Loss={result['loss']:.4f}, Epsilon={result['epsilon']:.3f}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        """Main training loop"""
        
        # Train forecasting model
        print("=== Training Forecasting Model ===")
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics['val_loss']
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Logging
            print(f"\nEpoch {epoch}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, MAE={val_metrics['val_mae']:.4f}, "
                  f"RMSE={val_metrics['val_rmse']:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, 'best_model.pth')
                print(f"Saved best model with val_loss={val_loss:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pth')
        
        # Train RL component
        if self.config.get('train_rl', True):
            self.train_rl_component(num_episodes=self.config.get('rl_episodes', 100))
        
        # Build RAG retrieval index
        if self.config.get('use_rag', True):
            print("\n=== Building RAG Retrieval Index ===")
            self.build_rag_index(train_loader)
    
    def build_rag_index(self, train_loader: DataLoader):
        """Build RAG retrieval index from training data"""
        self.model.eval()
        
        all_data = []
        with torch.no_grad():
            for batch in tqdm(train_loader, desc='Building RAG index'):
                x = batch['x'].to(self.device)
                all_data.append(x.cpu().numpy())
        
        historical_data = np.concatenate(all_data, axis=0)
        self.rag_system.build_retrieval_index(historical_data)
        print(f"Built RAG index with {len(historical_data)} patterns")
    
    def save_checkpoint(self, epoch: int, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        save_dir = self.config.get('save_dir', 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(checkpoint, os.path.join(save_dir, filename))
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train CTIS model')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default config
        config = {
            'n_nodes': 100,
            'input_dim': 10,
            'hidden_dim': 64,
            'output_dim': 10,
            'num_layers': 3,
            'window_size': 12,
            'horizon': 12,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'data_path': 'data/lade',
            'save_dir': args.save_dir,
            'n_features': 10,
            'node_feature_dim': 64,
            'num_workers': 4,
            'distance_threshold': 0.3,
            'use_rag': True,
            'train_rl': True,
            'rl_episodes': 100,
            'n_vehicles': 10,
            'max_capacity': 20,
            'rl_hidden_dim': 256,
            'rl_lr': 1e-4,
            'buffer_size': 100000,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'rl_batch_size': 64,
            'target_update_freq': 10,
            'embedding_dim': 128,
            'k_neighbors': 5,
            'llm_model': 'gpt2'
        }
    
    # Update with command line args
    config['batch_size'] = args.batch_size
    config['lr'] = args.lr
    config['save_dir'] = args.save_dir
    
    # Create data loaders
    train_loader = get_dataloader(config, mode='train')
    val_loader = get_dataloader(config, mode='val')
    
    # Create trainer
    trainer = CTISTrainer(config)
    
    # Train model
    trainer.train(train_loader, val_loader, num_epochs=args.epochs)
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()

