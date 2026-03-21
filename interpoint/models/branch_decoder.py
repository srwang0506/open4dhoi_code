"""
Branch Decoder modules for Human and Object contact prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class ConvBlock(nn.Module):
    """Convolutional block with BN and ReLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
        use_relu: bool = True
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        ]
        
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsampling block with skip connection."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        scale_factor: int = 2
    ):
        super().__init__()
        
        self.up = nn.Upsample(
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=False
        )
        
        self.conv = nn.Sequential(
            ConvBlock(in_channels + skip_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.up(x)
        
        if skip is not None:
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:],
                    mode='bilinear', align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)


class BranchDecoder(nn.Module):
    """
    Branch Decoder for contact prediction.
    
    Takes encoder features and semantic embedding, produces:
    1. Auxiliary 2D mask for supervision
    2. Enhanced feature maps for downstream transformer
    
    Architecture:
    - Input: Base features F_base + Semantic embedding E
    - Process: FPN-style decoder with skip connections
    - Output: 2D mask + Enhanced features
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 256,
        out_channels: int = 256,
        num_layers: int = 3,
        mask_channels: int = 1,
        input_resolution: Tuple[int, int] = (16, 16),
        output_resolution: Tuple[int, int] = (256, 256),
        use_skip_connections: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize branch decoder.
        
        Args:
            in_channels: Input feature channels
            hidden_dim: Hidden dimension
            out_channels: Output feature channels
            num_layers: Number of decoder layers
            mask_channels: Number of mask output channels
            input_resolution: Input feature map resolution (h, w)
            output_resolution: Output mask resolution (H, W)
            use_skip_connections: Whether to use skip connections
            dropout: Dropout rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.use_skip_connections = use_skip_connections
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        
        # Semantic embedding fusion
        self.semantic_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Initial fusion layer
        self.fusion = nn.Sequential(
            ConvBlock(in_channels + hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim)
        )
        
        # Decoder layers (upsampling)
        self.decoder_layers = nn.ModuleList()
        self.skip_convs = nn.ModuleList() if use_skip_connections else None
        
        current_dim = hidden_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else out_channels
            skip_dim = current_dim if use_skip_connections else 0
            
            self.decoder_layers.append(
                UpBlock(current_dim, skip_dim, out_dim, scale_factor=2)
            )
            
            if use_skip_connections:
                self.skip_convs.append(
                    ConvBlock(current_dim, current_dim)
                )
            
            current_dim = out_dim
        
        # Mask prediction head
        self.mask_head = nn.Sequential(
            ConvBlock(out_channels, hidden_dim // 2),
            nn.Conv2d(hidden_dim // 2, mask_channels, 1)
        )
        
        # Feature output projection
        self.feature_proj = nn.Sequential(
            ConvBlock(out_channels, out_channels),
            nn.Dropout2d(dropout)
        )
    
    def forward(
        self,
        features: torch.Tensor,
        semantic_emb: torch.Tensor,
        return_mask: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: (B*J, in_channels, h, w) encoder features
            semantic_emb: (B, hidden_dim) semantic embedding
            return_mask: Whether to compute and return mask
            
        Returns:
            Dictionary with:
                - 'features': (B*J, out_channels, h, w) enhanced features
                - 'mask': (B*J, 1, H, W) 2D contact mask (if return_mask)
        """
        BJ, C, h, w = features.shape
        
        # Expand semantic embedding to match features
        # Assume B*J features where semantic_emb is (B, D)
        # Need to repeat J times
        B = semantic_emb.shape[0]
        J = BJ // B
        
        semantic_emb = self.semantic_proj(semantic_emb)  # (B, hidden_dim)
        semantic_emb = semantic_emb.unsqueeze(1).expand(-1, J, -1)  # (B, J, D)
        semantic_emb = semantic_emb.reshape(BJ, -1)  # (BJ, D)
        
        # Tile semantic embedding spatially
        semantic_spatial = semantic_emb.unsqueeze(-1).unsqueeze(-1)
        semantic_spatial = semantic_spatial.expand(-1, -1, h, w)
        
        # Fuse features with semantic embedding
        x = torch.cat([features, semantic_spatial], dim=1)
        x = self.fusion(x)  # (BJ, hidden_dim, h, w)
        
        # Store intermediate features for skip connections
        skip_features = [x]
        
        # Decoder with upsampling
        for i, decoder_layer in enumerate(self.decoder_layers):
            if self.use_skip_connections and i < len(skip_features):
                skip = self.skip_convs[i](skip_features[-(i+1)])
                # Resize skip if needed
                if skip.shape[2:] != x.shape[2:]:
                    skip = F.interpolate(
                        skip, size=x.shape[2:],
                        mode='bilinear', align_corners=False
                    )
            else:
                skip = None
            
            x = decoder_layer(x, skip)
        
        # Enhanced features (at intermediate resolution)
        enhanced_features = self.feature_proj(x)
        
        # Resize to match input resolution for transformer
        if enhanced_features.shape[2:] != (h, w):
            feat_out = F.interpolate(
                enhanced_features, size=(h, w),
                mode='bilinear', align_corners=False
            )
        else:
            feat_out = enhanced_features
        
        result = {'features': feat_out}
        
        # Mask prediction
        if return_mask:
            mask = self.mask_head(enhanced_features)
            
            # Resize to output resolution
            if mask.shape[2:] != self.output_resolution:
                mask = F.interpolate(
                    mask, size=self.output_resolution,
                    mode='bilinear', align_corners=False
                )
            
            result['mask'] = mask
        
        return result


class HumanBranch(nn.Module):
    """
    Human-specific decoder branch.
    
    Specialized for human body contact regions with anatomical priors.
    """
    
    def __init__(
        self,
        d_tr: int = 256,
        num_layers: int = 3,
        input_resolution: Tuple[int, int] = (16, 16),
        output_resolution: Tuple[int, int] = (256, 256),
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.decoder = BranchDecoder(
            in_channels=d_tr,
            hidden_dim=d_tr,
            out_channels=d_tr,
            num_layers=num_layers,
            mask_channels=1,
            input_resolution=input_resolution,
            output_resolution=output_resolution,
            use_skip_connections=True,
            dropout=dropout
        )
    
    def forward(
        self,
        features: torch.Tensor,
        semantic_emb: torch.Tensor,
        return_mask: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B*J, d_tr, h, w) base visual features
            semantic_emb: (B, d_tr) human semantic embedding E_human
            
        Returns:
            - 'features': (B*J, d_tr, h, w) enhanced human features
            - 'mask': (B*J, 1, H, W) human contact mask
        """
        return self.decoder(features, semantic_emb, return_mask)


class ObjectBranch(nn.Module):
    """
    Object-specific decoder branch.
    
    Specialized for object affordance/contact regions.
    """
    
    def __init__(
        self,
        d_tr: int = 256,
        num_layers: int = 3,
        input_resolution: Tuple[int, int] = (16, 16),
        output_resolution: Tuple[int, int] = (256, 256),
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.decoder = BranchDecoder(
            in_channels=d_tr,
            hidden_dim=d_tr,
            out_channels=d_tr,
            num_layers=num_layers,
            mask_channels=1,
            input_resolution=input_resolution,
            output_resolution=output_resolution,
            use_skip_connections=True,
            dropout=dropout
        )
    
    def forward(
        self,
        features: torch.Tensor,
        semantic_emb: torch.Tensor,
        return_mask: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B*J, d_tr, h, w) base visual features
            semantic_emb: (B, d_tr) object semantic embedding E_object
            
        Returns:
            - 'features': (B*J, d_tr, h, w) enhanced object features
            - 'mask': (B*J, 1, H, W) object affordance mask
        """
        return self.decoder(features, semantic_emb, return_mask)


class DualBranchDecoder(nn.Module):
    """
    Combined dual-branch decoder for both human and object.
    
    Allows weight sharing in early layers while specializing
    in later layers for each branch.
    """
    
    def __init__(
        self,
        d_tr: int = 256,
        num_shared_layers: int = 1,
        num_branch_layers: int = 2,
        input_resolution: Tuple[int, int] = (16, 16),
        output_resolution: Tuple[int, int] = (256, 256),
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Shared initial processing
        self.shared_conv = nn.Sequential(
            ConvBlock(d_tr * 2, d_tr),  # Features + semantic
            ConvBlock(d_tr, d_tr)
        )
        
        # Human branch
        self.human_branch = HumanBranch(
            d_tr=d_tr,
            num_layers=num_branch_layers,
            input_resolution=input_resolution,
            output_resolution=output_resolution,
            dropout=dropout
        )
        
        # Object branch
        self.object_branch = ObjectBranch(
            d_tr=d_tr,
            num_layers=num_branch_layers,
            input_resolution=input_resolution,
            output_resolution=output_resolution,
            dropout=dropout
        )
    
    def forward(
        self,
        features: torch.Tensor,
        human_emb: torch.Tensor,
        object_emb: torch.Tensor,
        return_mask: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for both branches.
        
        Args:
            features: (B*J, d_tr, h, w) base visual features
            human_emb: (B, d_tr) human semantic embedding
            object_emb: (B, d_tr) object semantic embedding
            
        Returns:
            Dictionary with human and object features/masks
        """
        # Human branch
        human_out = self.human_branch(features, human_emb, return_mask)
        
        # Object branch
        object_out = self.object_branch(features, object_emb, return_mask)
        
        return {
            'human_features': human_out['features'],
            'object_features': object_out['features'],
            'human_mask': human_out.get('mask'),
            'object_mask': object_out.get('mask')
        }
