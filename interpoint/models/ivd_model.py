"""
InterActVLM-Discrete (IVD) Main Model
Integrates all components for HOI contact prediction
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List

from .vlm_module import VLMModule, LightweightVLM
from .pointnet2_encoder import PointNetv2Encoder
from .pointnet2_decoder import PointNet2FeatureDecoder, TemplatePointProvider, AffordanceDecoder
from .interaction_transformer import PointInteractionTransformer, ContactPredictionHead
from .losses import IVDLoss


class IVDModel(nn.Module):
    """
    InterActVLM-Discrete: Point Cloud + Image Fusion Model

    Architecture:
    1. VLM Module: Extracts semantic embeddings E_human, E_object from RGB image
    2. PointNet++ Encoder: Encodes human/object point clouds into point features
    3. Point Interaction Transformer: Cross-modal reasoning with learnable queries
    4. Prediction Heads: Binary classification for human + index classification for object
    5. Affordance Decoders: Per-point contact labels for human/object point clouds

    Input:
        - RGB image (B, 3, 224, 224)
        - Human point cloud (B, N_h, 3)
        - Object point cloud (B, N_o, 3)

    Output:
        - Human contact predictions (B, 87)
        - Object point index predictions (B, K)
        - Human affordance logits (B, 10475)
        - Object affordance logits (B, N_o)
    """
    
    def __init__(
        self,
        # Model dimensions
        d_tr: int = 256,
        num_body_points: int = 87,
        num_object_queries: int = 87,
        # VLM config
        vlm_model_name: str = "llava-hf/llava-1.5-7b-hf",
        use_lightweight_vlm: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        freeze_vlm: bool = True,

        # Transformer config
        transformer_num_layers: int = 6,
        transformer_num_heads: int = 8,
        transformer_dim_ff: int = 1024,
        transformer_dropout: float = 0.1,

        # Training config
        use_aux_loss: bool = True,
        object_softmax_tau: float = 0.5,

        # Device
        device: str = 'cuda',
        vlm_device_map: str = 'auto'  # Use 'cuda:0' for single GPU
    ):
        """Initialize IVD model."""
        super().__init__()
        
        self.d_tr = d_tr
        self.num_body_points = num_body_points
        self.num_object_queries = num_object_queries
        self.use_aux_loss = use_aux_loss
        self.device = device
        
        # Stage 1: VLM Module
        if use_lightweight_vlm:
            self.vlm = LightweightVLM(
                model_name="openai/clip-vit-base-patch32",
                d_tr=d_tr,
                device=device
            )
        else:
            self.vlm = VLMModule(
                model_name=vlm_model_name,
                d_tr=d_tr,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                freeze_backbone=freeze_vlm,
                device=device,
                device_map=vlm_device_map  # Pass device_map for single-GPU support
            )
        
        # Stage 2: PointNet++ Encoder (object only)
        self.object_pc_encoder = PointNetv2Encoder(with_decoder=False)

        self.point_feat_proj = nn.Linear(256, d_tr)
        self.object_sem_film = nn.Linear(d_tr, d_tr * 2)

        # Stage 3: Point Interaction Transformer
        self.transformer = PointInteractionTransformer(
            d_model=d_tr,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            dim_feedforward=transformer_dim_ff,
            dropout=transformer_dropout,
            num_body_points=num_body_points,
            num_object_queries=num_object_queries
        )
        
        # Stage 5: Prediction Heads
        self.prediction_head = ContactPredictionHead(
            d_model=d_tr,
            num_body_points=num_body_points,
            num_object_queries=num_object_queries,
            object_softmax_tau=object_softmax_tau,
        )

        # Stage 6: Point affordance decoders
        self.object_point_decoder = PointNet2FeatureDecoder()
        self.human_point_provider = TemplatePointProvider(
            num_points=10475,
            d_model=d_tr,
            context_dim=d_tr
        )
        self.human_affordance = AffordanceDecoder(d_model=d_tr, context_dim=d_tr)
        self.object_affordance = AffordanceDecoder(d_model=d_tr, context_dim=d_tr)
        
        # Loss function
        self.loss_fn = IVDLoss(
            lambda_human=1.0,
            lambda_object=1.0,
            lambda_aux_mask=0.5 if use_aux_loss else 0.0
        )
    
    def forward(
        self,
        rgb_image: torch.Tensor,
        object_points: Optional[torch.Tensor],
        return_aux: bool = True,
        text_prompts: Optional[List[str]] = None,
        compute_human: bool = True,
        compute_object: bool = True,
        compute_transformer: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full model.
        
        Args:
            rgb_image: (B, 3, 224, 224) original scene image
            object_points: (B, N_o, 3) object point cloud
            return_aux: Whether to return auxiliary mask predictions
            text_prompts: Optional text prompts for VLM
            
        Returns:
            Dictionary with:
                - 'human_contact': (B, 87) contact probabilities (if transformer computed)
                - 'human_logits': (B, 87) raw logits (if transformer computed)
                - 'object_coords': (B, K, 3) predicted coordinates (if transformer computed)
                - 'human_affordance': (B, 10475) per-point contact probabilities (if compute_human)
                - 'object_affordance': (B, N_o) per-point contact probabilities (if compute_object)
        """
        # Stage 1: VLM semantic extraction
        E_human, E_object = self.vlm(rgb_image, text_prompts)  # (B, d_tr) each
        
        outputs = {}
        human_feat = human_xyz = None
        object_feat = object_xyz = None

        if compute_object and object_points is not None:
            # Stage 2: PointNet++ encoding (object only)
            _, object_encode = self.object_pc_encoder(object_points)
            object_dec = self.object_point_decoder(object_encode)
            object_xyz = object_dec['xyz']
            object_feat = self.point_feat_proj(object_dec['features'])
            object_feat = self._fuse_semantic(object_feat, E_object, self.object_sem_film)

            object_aff = self.object_affordance(object_feat, E_object)
            outputs['object_affordance_logits'] = object_aff['logits']
            outputs['object_affordance'] = torch.sigmoid(object_aff['logits'])

        if compute_human:
            human_points = self.human_point_provider(E_human)
            human_feat = human_points['features']
            human_xyz = human_points['xyz']
            human_feat = self._fuse_semantic(human_feat, E_human, self.object_sem_film)

            human_aff = self.human_affordance(human_feat, E_human)
            outputs['human_affordance_logits'] = human_aff['logits']
            outputs['human_affordance'] = torch.sigmoid(human_aff['logits'])

        if compute_transformer and (human_feat is not None) and (object_feat is not None):
            # Stage 3: Point Interaction Transformer
            transformer_out = self.transformer(human_feat, human_xyz, object_feat, object_xyz)
            human_queries = transformer_out['human_queries']  # (B, 87, d_tr)
            object_queries = transformer_out['object_queries']  # (B, K, d_tr)

            # Stage 5: Prediction
            predictions = self.prediction_head(
                human_queries,
                object_queries,
                object_feat,
                object_xyz,
            )

            outputs['human_contact'] = predictions['human_contact']
            outputs['human_logits'] = predictions['human_logits']
            outputs['object_coords'] = predictions['object_coords']
            outputs['human_queries'] = human_queries
            outputs['object_queries'] = object_queries

        return outputs

    @staticmethod
    def _fuse_semantic(point_feat: torch.Tensor, sem_emb: torch.Tensor, film: nn.Module) -> torch.Tensor:
        gamma_beta = film(sem_emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return point_feat * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        compute_aux: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            predictions: Model outputs
            targets: Ground truth targets
            compute_aux: Whether to compute auxiliary mask loss
            
        Returns:
            Dictionary with individual and total losses
        """
        return self.loss_fn(predictions, targets, compute_aux=compute_aux)
    
    def predict(
        self,
        rgb_image: torch.Tensor,
        object_points: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Inference-time prediction.
        
        Args:
            rgb_image: (B, 3, 224, 224) input image
            object_points: (B, N_o, 3) object point cloud
            threshold: Binary classification threshold
            
        Returns:
            - 'human_contact_binary': (B, 87) binary predictions
            - 'human_contact_prob': (B, 87) probabilities
            - 'object_coords': (B, K, 3) predicted coordinates
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(rgb_image, object_points, return_aux=False)
        
        human_prob = outputs['human_contact']
        human_binary = (human_prob > threshold).float()
        return {
            'human_contact_binary': human_binary,
            'human_contact_prob': human_prob,
            'object_coords': outputs['object_coords'],
            'human_affordance': outputs['human_affordance'],
            'object_affordance': outputs['object_affordance']
        }
    
    def get_trainable_parameters(self) -> Dict[str, List[nn.Parameter]]:
        """
        Get grouped trainable parameters for different learning rates.
        
        Returns:
            Dictionary with parameter groups
        """
        params = {
            'vlm': [],
            'encoder': [],
            'decoder': [],
            'transformer': [],
            'head': []
        }
        
        # VLM parameters
        if hasattr(self.vlm, 'get_trainable_params'):
            params['vlm'] = self.vlm.get_trainable_params()
        else:
            params['vlm'] = list(self.vlm.parameters())
        
        # Encoder parameters
        params['encoder'] = [p for p in self.object_pc_encoder.parameters() if p.requires_grad]

        # Decoder parameters
        params['decoder'] = list(self.object_point_decoder.parameters()) + \
                           list(self.human_point_provider.parameters()) + \
                           list(self.human_affordance.parameters()) + \
                           list(self.object_affordance.parameters())
        
        # Transformer parameters
        params['transformer'] = list(self.transformer.parameters())
        
        # Head parameters
        params['head'] = list(self.prediction_head.parameters())
        
        return params
    
    def freeze_vlm(self):
        """Freeze VLM parameters."""
        for param in self.vlm.parameters():
            param.requires_grad = False
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.object_pc_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


def build_model(config: dict) -> IVDModel:
    """
    Build IVD model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized IVD model
    """
    model = IVDModel(
        d_tr=config.get('d_tr', 256),
        num_body_points=config.get('num_body_points', 87),
        num_object_queries=config.get('num_object_queries', 87),
        vlm_model_name=config.get('vlm_model_name', "llava-hf/llava-1.5-7b-hf"),
        use_lightweight_vlm=config.get('use_lightweight_vlm', False),
        lora_r=config.get('lora_r', 16),
        lora_alpha=config.get('lora_alpha', 32),
        freeze_vlm=config.get('freeze_vlm', True),
        transformer_num_layers=config.get('transformer_num_layers', 6),
        transformer_num_heads=config.get('transformer_num_heads', 8),
        transformer_dim_ff=config.get('transformer_dim_ff', 1024),
        transformer_dropout=config.get('transformer_dropout', 0),
        use_aux_loss=config.get('use_aux_loss', True),
        object_softmax_tau=config.get('object_softmax_tau', 0.5),
        device=config.get('device', 'cuda'),
        vlm_device_map=config.get('vlm_device_map', 'auto')  # 'cuda:0' for single GPU
    )

    return model
