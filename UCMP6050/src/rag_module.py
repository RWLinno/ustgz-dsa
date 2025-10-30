"""
RAG (Retrieval-Augmented Generation) Module for CTIS
Integrates Weilin's RAST framework for spatio-temporal forecasting
Based on: https://arxiv.org/pdf/2508.16623
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import faiss
from transformers import AutoModel, AutoTokenizer


class SpatioTemporalRetriever:
    """
    Retrieval module for finding similar spatio-temporal patterns
    Based on RAST (Retrieval-Augmented Spatio-Temporal forecasting)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.embedding_dim = config.get('embedding_dim', 128)
        self.k_neighbors = config.get('k_neighbors', 5)
        
        # Initialize FAISS index for efficient similarity search
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.historical_embeddings = []
        self.historical_data = []
        
    def build_index(self, embeddings: np.ndarray, data: np.ndarray):
        """
        Build FAISS index from historical embeddings
        Args:
            embeddings: (N, D) - Historical pattern embeddings
            data: (N, T, F) - Corresponding historical data
        """
        self.historical_embeddings = embeddings
        self.historical_data = data
        self.index.add(embeddings.astype('float32'))
        
    def retrieve(self, query_embedding: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve k most similar historical patterns
        Args:
            query_embedding: (D,) - Query embedding
            k: Number of neighbors to retrieve
        Returns:
            distances: (k,) - Distances to neighbors
            indices: (k,) - Indices of retrieved neighbors
        """
        k = k or self.k_neighbors
        query = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query, k)
        return distances[0], indices[0]
    
    def get_retrieved_data(self, indices: np.ndarray) -> np.ndarray:
        """
        Get historical data for retrieved indices
        """
        return self.historical_data[indices]


class RAGEncoder(nn.Module):
    """
    Encoder for generating embeddings for retrieval
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, T, F) - Batch of spatio-temporal data
        Returns: (B, D) - Embeddings
        """
        # Flatten spatio-temporal dimensions
        B = x.size(0)
        x_flat = x.reshape(B, -1)
        embeddings = self.encoder(x_flat)
        return embeddings


class RAGFusionModule(nn.Module):
    """
    Fusion module to combine retrieved patterns with current prediction
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            batch_first=True
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
    def forward(self, 
                current: torch.Tensor, 
                retrieved: torch.Tensor) -> torch.Tensor:
        """
        Args:
            current: (B, N, T, F) - Current prediction
            retrieved: (B, K, N, T, F) - Retrieved patterns
        Returns:
            fused: (B, N, T, F) - Fused representation
        """
        B, N, T, F = current.shape
        K = retrieved.size(1)
        
        # Reshape for attention
        current_flat = current.reshape(B, -1, F)  # (B, N*T, F)
        retrieved_flat = retrieved.reshape(B, K, -1, F).mean(dim=1)  # (B, N*T, F)
        
        # Apply attention
        attended, _ = self.attention(current_flat, retrieved_flat, retrieved_flat)
        
        # Fusion
        concatenated = torch.cat([current_flat, attended], dim=-1)  # (B, N*T, 2F)
        fused_flat = self.fusion_layer(concatenated)  # (B, N*T, F)
        
        # Reshape back
        fused = fused_flat.reshape(B, N, T, F)
        return fused


class LLMAgentModule:
    """
    LLM-based agent for interactive Q&A about transportation system
    Supports multiple user roles with different prompts
    """
    
    def __init__(self, config: Dict):
        self.config = config
        model_name = config.get('llm_model', 'gpt2')  # Use local model
        
        # Load model and tokenizer using transformers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Define role-based prompts
        self.role_prompts = {
            'driver': "You are a delivery driver. Answer questions about optimal routes and delivery schedules.",
            'dispatcher': "You are a dispatcher. Answer questions about overall fleet management and demand prediction.",
            'customer': "You are a customer service agent. Answer questions about delivery status and estimated arrival times.",
            'analyst': "You are a data analyst. Answer questions about traffic patterns and system performance."
        }
        
    def get_system_prompt(self, role: str) -> str:
        """Get system prompt for specific role"""
        return self.role_prompts.get(role, self.role_prompts['analyst'])
    
    def answer_query(self, 
                     query: str, 
                     context: Dict, 
                     role: str = 'analyst') -> str:
        """
        Answer user query with context from prediction system
        Args:
            query: User question
            context: Dict containing prediction results, node info, etc.
            role: User role for customized response
        Returns:
            answer: Generated response
        """
        # Build prompt with context
        system_prompt = self.get_system_prompt(role)
        
        # Extract relevant context
        prediction_summary = self._summarize_predictions(context)
        
        full_prompt = f"{system_prompt}\n\nContext: {prediction_summary}\n\nUser: {query}\n\nAssistant:"
        
        # Generate response (simplified - in practice use proper generation)
        inputs = self.tokenizer(full_prompt, return_tensors='pt', max_length=512, truncation=True)
        
        # For demo purposes, return context-aware response
        # In production, use proper LLM generation
        answer = self._generate_contextual_answer(query, context, role)
        
        return answer
    
    def _summarize_predictions(self, context: Dict) -> str:
        """Summarize prediction results for LLM context"""
        summary = "Current system status:\n"
        
        if 'predictions' in context:
            pred = context['predictions']
            summary += f"- Average demand: {pred.mean():.2f}\n"
            summary += f"- Peak demand: {pred.max():.2f}\n"
            summary += f"- Number of active locations: {len(pred)}\n"
        
        if 'timestamp' in context:
            summary += f"- Time: {context['timestamp']}\n"
            
        return summary
    
    def _generate_contextual_answer(self, query: str, context: Dict, role: str) -> str:
        """
        Generate context-aware answer
        (Simplified version - replace with actual LLM generation in production)
        """
        query_lower = query.lower()
        
        # Route-related queries
        if 'route' in query_lower or 'path' in query_lower:
            return f"Based on current traffic predictions, the optimal route is calculated using our spatio-temporal model. Current average travel time is {np.random.randint(15, 45)} minutes."
        
        # Demand-related queries
        elif 'demand' in query_lower or 'busy' in query_lower:
            if 'predictions' in context:
                peak = context['predictions'].max()
                return f"Peak demand is predicted to be {peak:.2f} deliveries. High-demand areas have been identified for resource allocation."
            return "Demand patterns are being analyzed from historical data."
        
        # Time-related queries
        elif 'when' in query_lower or 'time' in query_lower:
            return "Based on our multi-horizon forecasting model, we predict delivery completion within the next 2-3 hours for most orders."
        
        # General query
        else:
            return f"As a {role}, I can help you with information about the transportation system. Our models integrate RAST for pattern retrieval and WaveTS for accurate time series forecasting."


class CTISRAGSystem:
    """
    Complete RAG system for CTIS integrating retrieval and LLM
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize components
        self.retriever = SpatioTemporalRetriever(config)
        self.llm_agent = LLMAgentModule(config)
        
        # RAG encoder and fusion
        self.rag_encoder = RAGEncoder(
            input_dim=config['n_nodes'] * config['window_size'] * config['n_features'],
            hidden_dim=config.get('rag_hidden_dim', 256),
            embedding_dim=config.get('embedding_dim', 128)
        )
        
        self.fusion_module = RAGFusionModule(
            feature_dim=config['n_features'],
            hidden_dim=config.get('fusion_hidden_dim', 128)
        )
        
    def build_retrieval_index(self, historical_data: np.ndarray):
        """
        Build retrieval index from historical data
        Args:
            historical_data: (N_hist, N, T, F) - Historical patterns
        """
        # Generate embeddings for all historical patterns
        with torch.no_grad():
            data_tensor = torch.FloatTensor(historical_data)
            embeddings = self.rag_encoder(data_tensor).numpy()
        
        # Build FAISS index
        self.retriever.build_index(embeddings, historical_data)
        
    def retrieve_and_fuse(self, 
                         current_data: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Retrieve similar patterns and fuse with current data
        Args:
            current_data: (B, N, T, F) - Current input
        Returns:
            fused_data: (B, N, T, F) - Fused representation
            retrieved_indices: (B, K) - Indices of retrieved patterns
        """
        B = current_data.size(0)
        
        # Generate embeddings for current data
        with torch.no_grad():
            current_embeddings = self.rag_encoder(current_data).numpy()
        
        # Retrieve for each sample in batch
        all_retrieved = []
        all_indices = []
        
        for i in range(B):
            distances, indices = self.retriever.retrieve(current_embeddings[i])
            retrieved_data = self.retriever.get_retrieved_data(indices)
            all_retrieved.append(retrieved_data)
            all_indices.append(indices)
        
        # Stack retrieved data
        retrieved_tensor = torch.FloatTensor(np.stack(all_retrieved))  # (B, K, N, T, F)
        
        # Fuse with current data
        fused_data = self.fusion_module(current_data, retrieved_tensor)
        
        return fused_data, np.array(all_indices)
    
    def interactive_query(self, 
                         query: str,
                         prediction_results: Dict,
                         role: str = 'analyst') -> str:
        """
        Handle interactive user query
        Args:
            query: User question
            prediction_results: Current prediction context
            role: User role
        Returns:
            answer: Generated response
        """
        return self.llm_agent.answer_query(query, prediction_results, role)

