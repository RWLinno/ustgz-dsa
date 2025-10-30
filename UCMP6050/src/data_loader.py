"""
Data Loader for LaDe Dataset (Last-mile Delivery)
Based on: https://arxiv.org/pdf/2306.10675
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import json
from typing import Dict, List, Tuple, Optional
import pickle
import os


class LaDe Dataset:
    """
    LaDe (Last-mile Delivery) Dataset Loader
    
    The dataset contains:
    - Pickup and Delivery locations (spatio-temporal graph structure)
    - Temporal features (demand patterns)
    - Spatial features (location embeddings, distance matrix)
    - Environmental features (weather, traffic conditions)
    """
    
    def __init__(self, data_path: str, config: Dict):
        self.data_path = data_path
        self.config = config
        self.n_nodes = config.get('n_nodes', 100)  # Number of delivery points
        self.n_features = config.get('n_features', 10)
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load LaDe dataset
        Returns:
            - temporal_data: (N, T, F) - N nodes, T timesteps, F features
            - adj_matrix: (N, N) - Adjacency matrix (pickup-delivery relationships)
            - node_features: (N, D) - Static node features (location embeddings)
        """
        # Simulated data structure (replace with actual LaDe data loading)
        temporal_data = np.random.randn(self.n_nodes, 168, self.n_features)  # 1 week hourly
        
        # Build adjacency matrix based on pickup-delivery relationships
        adj_matrix = self._build_adjacency_matrix()
        
        # Node features: location embeddings + environmental context
        node_features = self._extract_node_features()
        
        return temporal_data, adj_matrix, node_features
    
    def _build_adjacency_matrix(self) -> np.ndarray:
        """
        Build graph structure from pickup-delivery relationships
        """
        # Distance-based + pickup-delivery relationship-based adjacency
        adj = np.zeros((self.n_nodes, self.n_nodes))
        
        # Add edges based on delivery routes and distance threshold
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                # Simulate distance-based connectivity
                distance = np.random.rand()
                if distance < self.config.get('distance_threshold', 0.3):
                    adj[i, j] = adj[j, i] = 1.0
        
        return adj
    
    def _extract_node_features(self) -> np.ndarray:
        """
        Extract static features for each delivery point:
        - Location coordinates
        - POI (Point of Interest) features
        - Historical demand statistics
        """
        node_features = np.random.randn(self.n_nodes, 64)
        return node_features


class CTISDataset(Dataset):
    """
    PyTorch Dataset wrapper for CTIS (Connected Transportation Information System)
    """
    
    def __init__(self, 
                 temporal_data: np.ndarray,
                 adj_matrix: np.ndarray,
                 node_features: np.ndarray,
                 window_size: int = 12,
                 horizon: int = 12,
                 mode: str = 'train'):
        
        self.temporal_data = torch.FloatTensor(temporal_data)
        self.adj_matrix = torch.FloatTensor(adj_matrix)
        self.node_features = torch.FloatTensor(node_features)
        self.window_size = window_size
        self.horizon = horizon
        self.mode = mode
        
        # Split data
        self.n_samples = temporal_data.shape[1] - window_size - horizon + 1
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Historical window
        x = self.temporal_data[:, idx:idx+self.window_size, :]  # (N, T_in, F)
        # Future horizon
        y = self.temporal_data[:, idx+self.window_size:idx+self.window_size+self.horizon, :]  # (N, T_out, F)
        
        return {
            'x': x,
            'y': y,
            'adj': self.adj_matrix,
            'node_features': self.node_features
        }


def get_dataloader(config: Dict, mode: str = 'train') -> DataLoader:
    """
    Create DataLoader for CTIS
    """
    # Load LaDe dataset
    lade = LaDeDataset(config['data_path'], config)
    temporal_data, adj_matrix, node_features = lade.load_data()
    
    # Create dataset
    dataset = CTISDataset(
        temporal_data=temporal_data,
        adj_matrix=adj_matrix,
        node_features=node_features,
        window_size=config['window_size'],
        horizon=config['horizon'],
        mode=mode
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=(mode == 'train'),
        num_workers=config.get('num_workers', 4)
    )
    
    return dataloader

