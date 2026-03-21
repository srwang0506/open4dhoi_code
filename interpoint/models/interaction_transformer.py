"""
Shared Interaction Transformer for HOI contact reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for query-to-memory attention."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, D) query tokens
            memory: (B, N_m, D) memory (visual features)
            query_pos: (B, N_q, D) query positional embedding
            memory_pos: (B, N_m, D) memory positional embedding
            
        Returns:
            Updated query: (B, N_q, D)
        """
        # Add positional embeddings
        q = query + query_pos if query_pos is not None else query
        k = memory + memory_pos if memory_pos is not None else memory
        v = memory
        
        # Cross-attention
        attn_out, _ = self.cross_attn(q, k, v)
        query = query + self.dropout1(attn_out)
        query = self.norm1(query)
        
        # Self-attention among queries (with query positional embedding)
        q_self = query + query_pos if query_pos is not None else query
        attn_out, _ = self.self_attn(q_self, q_self, query)
        query = query + self.dropout2(attn_out)
        query = self.norm2(query)
        
        # Feed-forward
        query = query + self.ffn(query)
        query = self.norm3(query)
        
        return query


class InteractionTransformer(nn.Module):
    """
    Shared Interaction Transformer for HOI reasoning.
    
    Takes enhanced visual features from both human and object branches,
    and uses learnable queries to reason about contact interactions.
    
    Architecture:
    - Memory: Concatenated features [F_human, F_object] from all views
    - Queries: 87 human contact queries + K object contact queries
    - Process: Cross-attention to memory + Self-attention for interaction
    - Output: Updated queries for prediction heads
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_body_points: int = 87,
        num_object_queries: int = 4,
        num_views: int = 4
    ):
        """
        Initialize Interaction Transformer.
        
        Args:
            d_model: Model dimension (d_tr)
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
            num_body_points: Number of body contact points (87)
            num_object_queries: Number of object contact queries (K)
            num_views: Number of rendered views (J)
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_body_points = num_body_points
        self.num_object_queries = num_object_queries
        self.num_views = num_views
        
        # Learnable queries for body points
        self.body_queries = nn.Parameter(
            torch.randn(1, num_body_points, d_model) * 0.02
        )
        
        # Learnable queries for object contacts
        self.object_queries = nn.Parameter(
            torch.randn(1, num_object_queries, d_model) * 0.02
        )
        
        # Positional encoding for queries
        self.body_pos_embed = nn.Parameter(
            torch.randn(1, num_body_points, d_model) * 0.02
        )
        self.object_pos_embed = nn.Parameter(
            torch.randn(1, num_object_queries, d_model) * 0.02
        )
        
        # View embedding for multi-view features
        self.view_embed = nn.Parameter(
            torch.randn(1, num_views, 1, d_model) * 0.02
        )
        
        # Type embedding (human vs object features)
        self.type_embed = nn.Parameter(
            torch.randn(1, 2, 1, d_model) * 0.02  # 0: human, 1: object
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projections
        self.human_out_proj = nn.Linear(d_model, d_model)
        self.object_out_proj = nn.Linear(d_model, d_model)
    
    def _prepare_memory(
        self,
        human_features: torch.Tensor,
        object_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare memory from human and object features.
        
        Args:
            human_features: (B, J, d_model, h, w) human features per view
            object_features: (B, J, d_model, h, w) object features per view
            
        Returns:
            memory: (B, 2*J*h*w, d_model) concatenated memory
            memory_pos: (B, 2*J*h*w, d_model) positional embeddings
        """
        B, J, D, h, w = human_features.shape
        
        # Flatten spatial dimensions
        human_flat = human_features.flatten(3)  # (B, J, D, h*w)
        object_flat = object_features.flatten(3)  # (B, J, D, h*w)
        
        # Permute to (B, J, h*w, D)
        human_flat = human_flat.permute(0, 1, 3, 2)
        object_flat = object_flat.permute(0, 1, 3, 2)
        
        # Add view embeddings
        view_emb = self.view_embed.expand(B, -1, h*w, -1)  # (B, J, h*w, D)
        human_flat = human_flat + view_emb
        object_flat = object_flat + view_emb
        
        # Add type embeddings
        human_flat = human_flat + self.type_embed[:, 0:1, :, :]
        object_flat = object_flat + self.type_embed[:, 1:2, :, :]
        
        # Flatten to sequence
        human_seq = human_flat.flatten(1, 2)  # (B, J*h*w, D)
        object_seq = object_flat.flatten(1, 2)  # (B, J*h*w, D)
        
        # Concatenate
        memory = torch.cat([human_seq, object_seq], dim=1)  # (B, 2*J*h*w, D)
        
        # Generate positional embeddings
        memory_pos = self._generate_spatial_pos(B, J, h, w, D, memory.device)
        
        return memory, memory_pos
    
    def _generate_spatial_pos(
        self,
        B: int,
        J: int,
        h: int,
        w: int,
        D: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate spatial positional embeddings."""
        # 2D sinusoidal positional encoding
        y_embed = torch.arange(h, device=device).float()
        x_embed = torch.arange(w, device=device).float()
        
        y_embed = y_embed / h
        x_embed = x_embed / w
        
        dim_t = torch.arange(D // 4, device=device).float()
        dim_t = 10000 ** (4 * dim_t / D)
        
        # Create encoding
        pos_y = y_embed.unsqueeze(1).unsqueeze(1) / dim_t  # (h, 1, D//4)
        pos_x = x_embed.unsqueeze(0).unsqueeze(2) / dim_t  # (1, w, D//4)
        
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).flatten(-2)
        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).flatten(-2)
        
        pos = torch.cat([
            pos_y.expand(-1, w, -1),
            pos_x.expand(h, -1, -1)
        ], dim=-1)  # (h, w, D)
        
        pos = pos.flatten(0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, h*w, D)
        pos = pos.expand(B, J, -1, -1)  # (B, J, h*w, D)
        pos = pos.flatten(1, 2)  # (B, J*h*w, D)
        
        # Duplicate for human and object
        pos = torch.cat([pos, pos], dim=1)  # (B, 2*J*h*w, D)
        
        return pos
    
    def forward(
        self,
        human_features: torch.Tensor,
        object_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Interaction Transformer.
        
        Args:
            human_features: (B, J, d_model, h, w) or (B*J, d_model, h, w)
            object_features: (B, J, d_model, h, w) or (B*J, d_model, h, w)
            
        Returns:
            Dictionary with:
                - 'human_queries': (B, 87, d_model) updated body queries
                - 'object_queries': (B, K, d_model) updated object queries
        """
        # Handle input shape
        if human_features.dim() == 4:
            # (B*J, d_model, h, w) -> (B, J, d_model, h, w)
            BJ, D, h, w = human_features.shape
            J = self.num_views
            B = BJ // J
            human_features = human_features.view(B, J, D, h, w)
            object_features = object_features.view(B, J, D, h, w)
        else:
            B = human_features.shape[0]
        
        # Prepare memory
        memory, memory_pos = self._prepare_memory(human_features, object_features)
        
        # Initialize queries
        body_q = self.body_queries.expand(B, -1, -1)  # (B, 87, D)
        object_q = self.object_queries.expand(B, -1, -1)  # (B, K, D)
        
        # Concatenate queries
        queries = torch.cat([body_q, object_q], dim=1)  # (B, 87+K, D)
        query_pos = torch.cat([
            self.body_pos_embed.expand(B, -1, -1),
            self.object_pos_embed.expand(B, -1, -1)
        ], dim=1)  # (B, 87+K, D)
        
        # Apply transformer layers
        for layer in self.layers:
            queries = layer(queries, memory, query_pos, memory_pos)
        
        # Split outputs
        human_out = queries[:, :self.num_body_points, :]  # (B, 87, D)
        object_out = queries[:, self.num_body_points:, :]  # (B, K, D)
        
        # Project outputs
        human_out = self.human_out_proj(human_out)
        object_out = self.object_out_proj(object_out)
        
        return {
            'human_queries': human_out,
            'object_queries': object_out
        }


class PointInteractionTransformer(nn.Module):
    """
    Interaction transformer for point cloud memories (human/object point features).
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_body_points: int = 87,
        num_object_queries: int = 4
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_body_points = num_body_points
        self.num_object_queries = num_object_queries

        self.body_queries = nn.Parameter(torch.randn(1, num_body_points, d_model) * 0.02)
        self.object_queries = nn.Parameter(torch.randn(1, num_object_queries, d_model) * 0.02)

        self.body_pos_embed = nn.Parameter(torch.randn(1, num_body_points, d_model) * 0.02)
        self.object_pos_embed = nn.Parameter(torch.randn(1, num_object_queries, d_model) * 0.02)

        self.type_embed = nn.Parameter(torch.randn(1, 2, 1, d_model) * 0.02)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )

        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.human_out_proj = nn.Linear(d_model, d_model)
        self.object_out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        human_features: torch.Tensor,
        human_xyz: torch.Tensor,
        object_features: torch.Tensor,
        object_xyz: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            human_features: (B, N_h, d_model)
            human_xyz: (B, N_h, 3)
            object_features: (B, N_o, d_model)
            object_xyz: (B, N_o, 3)
        """
        B = human_features.shape[0]

        human_pos = self.pos_mlp(human_xyz)
        object_pos = self.pos_mlp(object_xyz)

        # Keep positional signal in `memory_pos` only. Adding pos here and again in
        # CrossAttentionLayer(key = memory + memory_pos) effectively double-counts it.
        human_feat = human_features + self.type_embed[:, 0, :, :]
        object_feat = object_features + self.type_embed[:, 1, :, :]

        memory = torch.cat([human_feat, object_feat], dim=1)
        memory_pos = torch.cat([human_pos, object_pos], dim=1)

        body_q = self.body_queries.expand(B, -1, -1)
        object_q = self.object_queries.expand(B, -1, -1)
        queries = torch.cat([body_q, object_q], dim=1)
        query_pos = torch.cat([
            self.body_pos_embed.expand(B, -1, -1),
            self.object_pos_embed.expand(B, -1, -1)
        ], dim=1)

        for layer in self.layers:
            queries = layer(queries, memory, query_pos, memory_pos)

        human_out = queries[:, :self.num_body_points, :]
        object_out = queries[:, self.num_body_points:, :]

        return {
            'human_queries': self.human_out_proj(human_out),
            'object_queries': self.object_out_proj(object_out)
        }


class ContactPredictionHead(nn.Module):
    """
    Prediction heads for human contact classification and object coordinate regression.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_body_points: int = 87,
        num_object_queries: int = 4,
        hidden_dim: int = 128,
        object_softmax_tau: float = 0.5
    ):
        """
        Initialize prediction heads.
        
        Args:
            d_model: Input dimension
            num_body_points: Number of body contact points
            num_object_queries: Number of object queries
            hidden_dim: Hidden dimension for MLPs
        """
        super().__init__()
        
        self.num_body_points = num_body_points
        self.num_object_queries = num_object_queries
        self.object_softmax_tau = max(float(object_softmax_tau), 1e-6)
        self.object_attn_scale = d_model ** -0.5
        
        # Human contact classification head (binary)
        self.human_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # Object coordinate regression via attention over point features
        self.object_query_proj = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        human_queries: torch.Tensor,
        object_queries: torch.Tensor,
        object_feats: torch.Tensor,
        object_xyz: torch.Tensor,
        affordance_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict contact outputs.
        
        Args:
            human_queries: (B, 87, d_model) body query embeddings
            object_queries: (B, K, d_model) object query embeddings
            object_feats: (B, N, d_model) object point features
            object_xyz: (B, N, 3) object point coordinates
            affordance_mask: (B, N) object affordance mask/probabilities
            
        Returns:
            - 'human_contact': (B, 87) contact probabilities
            - 'object_coords': (B, K, 3) predicted coordinates
        """
        # Human contact prediction
        human_logits = self.human_head(human_queries).squeeze(-1)  # (B, 87)
        human_probs = torch.sigmoid(human_logits)
        
        # Object coordinate prediction from object query/object feature attention
        queries = self.object_query_proj(object_queries)  # (B, K, D)
        # Scaled dot-product attention for numerical stability.
        logits = torch.matmul(queries, object_feats.transpose(1, 2)) * self.object_attn_scale  # (B, K, N)
        weights = torch.softmax(logits / self.object_softmax_tau, dim=-1)
        object_coords = torch.matmul(weights, object_xyz)  # (B, K, 3)
        
        return {
            'human_contact': human_probs,
            'human_logits': human_logits,
            'object_coords': object_coords
        }
