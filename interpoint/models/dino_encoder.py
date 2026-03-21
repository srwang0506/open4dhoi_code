"""
DINOv2 Encoder for Multi-View Visual Feature Extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class DinoEncoder(nn.Module):
    """
    DINOv2 Visual Encoder for multi-view feature extraction.
    
    Encodes rendered views of human mesh and object point cloud
    into feature maps for downstream contact reasoning.
    
    Architecture:
    - Input: Rendered views (B*J, 3, H, W)
    - Backbone: DINOv2 ViT (frozen or fine-tuned)
    - Output: Feature maps (B*J, D, h, w)
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        d_tr: int = 256,
        pretrained: bool = True,
        freeze: bool = False,
        use_cls_token: bool = False,
        output_stride: int = 14
    ):
        """
        Initialize DINOv2 encoder.
        
        Args:
            model_name: Model name (supports HF or timm naming)
            d_tr: Target feature dimension
            pretrained: Whether to use pretrained weights
            freeze: Whether to freeze encoder weights
            use_cls_token: Whether to include CLS token in output
            output_stride: Patch size / output stride (14 for ViT-S/14)
        """
        super().__init__()
        
        self.d_tr = d_tr
        self.use_cls_token = use_cls_token
        self.output_stride = output_stride
        
        # Load DINOv2 model
        self.backbone, self.hidden_dim = self._load_backbone(
            model_name, pretrained
        )
        
        # Freeze if requested
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Project to d_tr if needed
        if self.hidden_dim != d_tr:
            self.proj = nn.Sequential(
                nn.Conv2d(self.hidden_dim, d_tr, 1),
                nn.BatchNorm2d(d_tr),
                nn.ReLU(inplace=True)
            )
        else:
            self.proj = nn.Identity()
    
    def _load_backbone(
        self,
        model_name: str,
        pretrained: bool
    ) -> Tuple[nn.Module, int]:
        """Load DINOv2 backbone."""
        # Try different loading methods
        
        # Method 1: torch.hub (official)
        try:
            if 'small' in model_name.lower() or 'vits' in model_name.lower():
                backbone = torch.hub.load(
                    'facebookresearch/dinov2',
                    'dinov2_vits14',
                    pretrained=pretrained
                )
                hidden_dim = 384
            elif 'base' in model_name.lower() or 'vitb' in model_name.lower():
                backbone = torch.hub.load(
                    'facebookresearch/dinov2',
                    'dinov2_vitb14',
                    pretrained=pretrained
                )
                hidden_dim = 768
            elif 'large' in model_name.lower() or 'vitl' in model_name.lower():
                backbone = torch.hub.load(
                    'facebookresearch/dinov2',
                    'dinov2_vitl14',
                    pretrained=pretrained
                )
                hidden_dim = 1024
            else:
                backbone = torch.hub.load(
                    'facebookresearch/dinov2',
                    'dinov2_vits14',
                    pretrained=pretrained
                )
                hidden_dim = 384
            
            return backbone, hidden_dim
        except Exception as e:
            print(f"Warning: Could not load from torch.hub: {e}")
        
        # Method 2: timm
        if TIMM_AVAILABLE:
            try:
                backbone = timm.create_model(
                    'vit_small_patch14_dinov2.lvd142m',
                    pretrained=pretrained,
                    num_classes=0  # Remove classification head
                )
                hidden_dim = 384
                return backbone, hidden_dim
            except Exception as e:
                print(f"Warning: Could not load from timm: {e}")
        
        # Method 3: Fallback to simple ViT
        print("Using fallback ViT encoder")
        backbone = SimpleViT(
            image_size=256,
            patch_size=14,
            dim=384,
            depth=12,
            heads=6
        )
        hidden_dim = 384
        
        return backbone, hidden_dim
    
    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Extract visual features from rendered views.
        
        Args:
            x: (B*J, 3, H, W) rendered view images
            return_intermediate: Whether to return intermediate features
            
        Returns:
            features: (B*J, d_tr, h, w) feature maps
        """
        B, C, H, W = x.shape
        
        # Calculate output spatial dimensions
        h = H // self.output_stride
        w = W // self.output_stride
        
        # Get features from backbone
        if hasattr(self.backbone, 'forward_features'):
            # timm-style
            features = self.backbone.forward_features(x)
        elif hasattr(self.backbone, 'get_intermediate_layers'):
            # DINOv2 official style
            features = self.backbone.get_intermediate_layers(
                x, n=1, reshape=True
            )[0]
        else:
            # Fallback
            features = self.backbone(x)
        
        # Handle different output formats
        if features.dim() == 3:
            # (B, N, D) -> (B, D, h, w)
            if self.use_cls_token:
                features = features[:, :, :]  # Keep all tokens
                N = features.shape[1] - 1
            else:
                features = features[:, 1:, :]  # Remove CLS token
                N = features.shape[1]
            
            # Reshape to spatial
            features = features.permute(0, 2, 1).reshape(B, -1, h, w)
        elif features.dim() == 4:
            # Already (B, D, h, w)
            pass
        
        # Project to d_tr
        features = self.proj(features)  # (B, d_tr, h, w)
        
        return features
    
    def forward_with_pos(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features with positional information.
        
        Args:
            x: (B*J, 3, H, W) rendered views
            
        Returns:
            features: (B*J, d_tr, h, w) feature maps
            pos_embed: (B*J, d_tr, h, w) positional embeddings
        """
        features = self.forward(x)
        
        B, D, h, w = features.shape
        
        # Generate sinusoidal positional embeddings
        pos_embed = self._get_positional_embedding(h, w, D, x.device)
        pos_embed = pos_embed.unsqueeze(0).expand(B, -1, -1, -1)
        
        return features, pos_embed
    
    def _get_positional_embedding(
        self,
        h: int,
        w: int,
        dim: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate 2D sinusoidal positional embeddings."""
        y_embed = torch.arange(h, device=device).float()
        x_embed = torch.arange(w, device=device).float()
        
        y_embed = y_embed.unsqueeze(1).expand(h, w)
        x_embed = x_embed.unsqueeze(0).expand(h, w)
        
        # Normalize to [0, 1]
        y_embed = y_embed / h
        x_embed = x_embed / w
        
        # Create sin/cos embeddings
        dim_t = torch.arange(dim // 2, device=device).float()
        dim_t = 10000 ** (2 * dim_t / dim)
        
        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t
        
        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).flatten(-2)
        
        pos = torch.cat([pos_y, pos_x], dim=-1).permute(2, 0, 1)  # (D, h, w)
        
        return pos[:dim]


class SimpleViT(nn.Module):
    """
    Simple Vision Transformer as fallback encoder.
    """
    
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 14,
        dim: int = 384,
        depth: int = 12,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.dim = dim
        
        num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, dim) * 0.02
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, dim, h, w)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, N, dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Transformer
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer block."""
    
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x
