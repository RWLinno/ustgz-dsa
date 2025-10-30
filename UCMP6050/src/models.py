"""
CTIS Core Models
Integrating:
1. Weilin's RAST (Retrieval-Augmented Spatio-Temporal Forecasting)
2. Ziyu's WaveTS (Deep Time Series Forecasting)
3. Songxin's RL components
4. Yiming's LLM security mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class GraphConvolution(nn.Module):
    """
    Adaptive Graph Convolution for spatial relationships
    Inspired by RAST framework
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, F_in)
            adj: (N, N)
        Returns:
            out: (B, N, F_out)
        """
        # Normalize adjacency matrix
        d = adj.sum(dim=1, keepdim=True)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        adj_norm = d_inv_sqrt * adj * d_inv_sqrt.T
        
        # Graph convolution
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj_norm, support) + self.bias
        
        return output


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network inspired by WaveTS
    Captures multi-scale temporal patterns
    """
    def __init__(self, num_inputs: int, num_channels: List[int], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size,
                         padding=(kernel_size-1) * dilation_size,
                         dilation=dilation_size)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, T, F) -> reshape to (B*N, F, T)
        Returns:
            out: (B, N, T, F_out)
        """
        B, N, T, F = x.shape
        x = x.reshape(B*N, F, T)
        out = self.network(x)  # (B*N, F_out, T)
        _, F_out, T_out = out.shape
        out = out.reshape(B, N, T_out, F_out)
        return out


class RetrievalAugmentedModule(nn.Module):
    """
    Retrieval-Augmented Module (RAG) for CTIS
    Based on Weilin's RAST framework
    Retrieves relevant historical patterns to enhance prediction
    """
    def __init__(self, hidden_dim: int, memory_size: int = 1000, top_k: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.top_k = top_k
        
        # Memory bank for storing historical patterns
        self.register_buffer('memory_keys', torch.randn(memory_size, hidden_dim))
        self.register_buffer('memory_values', torch.randn(memory_size, hidden_dim))
        
        # Query projection
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, N, F)
        Returns:
            augmented: (B, N, F)
        """
        B, N, F = query.shape
        
        # Project query
        q = self.query_proj(query)  # (B, N, F)
        
        # Compute similarity with memory keys
        q_flat = q.reshape(-1, F)  # (B*N, F)
        similarity = torch.matmul(q_flat, self.memory_keys.T)  # (B*N, M)
        
        # Retrieve top-k
        topk_sim, topk_idx = torch.topk(similarity, self.top_k, dim=-1)  # (B*N, K)
        topk_values = self.memory_values[topk_idx]  # (B*N, K, F)
        
        # Weighted aggregation
        weights = F.softmax(topk_sim, dim=-1).unsqueeze(-1)  # (B*N, K, 1)
        retrieved = (topk_values * weights).sum(dim=1)  # (B*N, F)
        retrieved = retrieved.reshape(B, N, F)
        
        # Fusion
        augmented = self.fusion(torch.cat([query, retrieved], dim=-1))
        
        return augmented


class CTISModel(nn.Module):
    """
    Main CTIS Model integrating all components:
    - Spatio-temporal forecasting (RAST + WaveTS)
    - Retrieval-augmented learning
    - Multi-task learning for transportation metrics
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Input dimensions with safe defaults
        self.n_nodes = config.get('n_nodes', 100)
        self.input_dim = config.get('input_dim', 10)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.output_dim = config.get('output_dim', 10)
        self.num_layers = config.get('num_layers', 3)
        
        # Input embedding
        self.input_embed = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Node feature embedding
        self.node_embed = nn.Linear(config.get('node_feature_dim', 64), self.hidden_dim)
        
        # Spatio-temporal layers
        self.st_layers = nn.ModuleList([
            SpatioTemporalLayer(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_layers)
        ])
        
        # Retrieval-augmented module
        self.rag = RetrievalAugmentedModule(
            self.hidden_dim,
            memory_size=config.get('memory_size', 1000),
            top_k=config.get('top_k', 5)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor, node_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, T, F_in) - Input temporal features
            adj: (N, N) - Adjacency matrix
            node_features: (N, D) - Static node features
        Returns:
            out: (B, N, T_out, F_out) - Predictions
        """
        B, N, T, F_in = x.shape
        
        # Input embedding
        x = self.input_embed(x)  # (B, N, T, H)
        
        # Node embedding (broadcasted)
        node_emb = self.node_embed(node_features).unsqueeze(0).unsqueeze(2)  # (1, N, 1, H)
        x = x + node_emb  # (B, N, T, H)
        
        # Spatio-temporal processing
        for layer in self.st_layers:
            x = layer(x, adj)  # (B, N, T, H)
        
        # Aggregate temporal dimension for RAG
        x_agg = x.mean(dim=2)  # (B, N, H)
        
        # Retrieval-augmented enhancement
        x_aug = self.rag(x_agg)  # (B, N, H)
        
        # Add back to temporal features
        x = x + x_aug.unsqueeze(2)  # (B, N, T, H)
        
        # Output projection
        out = self.output_proj(x)  # (B, N, T, F_out)
        
        return out


class SpatioTemporalLayer(nn.Module):
    """
    Combined Spatio-Temporal Layer
    """
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        
        # Spatial convolution
        self.spatial_conv = GraphConvolution(hidden_dim, hidden_dim)
        
        # Temporal convolution
        self.temporal_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Residual connection
        self.residual = nn.Linear(hidden_dim, out_dim) if hidden_dim != out_dim else nn.Identity()
        
        # Layer norm
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, T, F)
            adj: (N, N)
        Returns:
            out: (B, N, T, F)
        """
        residual = x
        
        # Temporal convolution
        B, N, T, F = x.shape
        x_t = x.permute(0, 1, 3, 2).reshape(B*N, F, T)  # (B*N, F, T)
        x_t = self.temporal_conv(x_t)  # (B*N, F, T)
        x_t = x_t.reshape(B, N, F, T).permute(0, 1, 3, 2)  # (B, N, T, F)
        x_t = F.relu(x_t)
        
        # Spatial convolution (applied per timestep)
        x_s = []
        for t in range(T):
            x_s_t = self.spatial_conv(x_t[:, :, t, :], adj)  # (B, N, F)
            x_s.append(x_s_t)
        x_s = torch.stack(x_s, dim=2)  # (B, N, T, F)
        x_s = F.relu(x_s)
        
        # Residual connection
        out = self.norm(x_s + self.residual(residual))
        
        return out


def create_model(config: Dict) -> CTISModel:
    """
    Factory function to create CTIS model
    """
    model = CTISModel(config)
    return model

