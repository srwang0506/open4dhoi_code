"""
Loss functions for InterActVLM-Discrete (IVD)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DiceLoss(nn.Module):
    """Dice loss for mask prediction."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: (B, 1, H, W) predicted mask (after sigmoid)
            target: (B, 1, H, W) ground truth mask
            
        Returns:
            Scalar loss
        """
        pred = pred.flatten(1)
        target = target.flatten(1)
        
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal loss.
        
        Args:
            pred: (B, N) predicted logits
            target: (B, N) binary targets
            
        Returns:
            Scalar loss
        """
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        pred_prob = torch.sigmoid(pred)
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ChamferLoss(nn.Module):
    """Chamfer distance loss for point set prediction."""
    
    def __init__(self, bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bidirectional
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Chamfer distance.
        
        Args:
            pred: (B, K, 3) predicted points
            target: (B, K, 3) target points
            mask: (B, K) optional mask for valid points
            
        Returns:
            Scalar loss
        """
        dist = torch.cdist(pred, target)  # (B, K, K)

        if mask is None:
            min_p2t = dist.min(dim=2)[0]  # (B, K)
            loss_p2t = min_p2t.mean()
            if self.bidirectional:
                min_t2p = dist.min(dim=1)[0]  # (B, K)
                loss_t2p = min_t2p.mean()
                return (loss_p2t + loss_t2p) / 2
            return loss_p2t

        valid = mask.bool()
        valid_count = valid.sum()
        if valid_count.item() == 0:
            return dist.sum() * 0.0

        # Only distances between valid points should participate in nearest-neighbor matching.
        valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)  # (B, K, K)
        inf = torch.finfo(dist.dtype).max
        dist_valid = dist.masked_fill(~valid_pair, inf)

        min_p2t = dist_valid.min(dim=2)[0]  # (B, K)
        loss_p2t = min_p2t[valid].sum() / (valid_count + 1e-8)

        if self.bidirectional:
            min_t2p = dist_valid.min(dim=1)[0]  # (B, K)
            loss_t2p = min_t2p[valid].sum() / (valid_count + 1e-8)
            return (loss_p2t + loss_t2p) / 2
        return loss_p2t


class IVDLoss(nn.Module):
    """
    Combined loss function for InterActVLM-Discrete.
    
    Combines:
    1. L_human: Binary classification loss for 87 body points
    2. L_object: Regression loss for object contact coordinates
    3. L_aux_mask: Auxiliary mask prediction loss (for early training)
    """
    def __init__(
        self,
        lambda_human: float = 1.0,
        lambda_object: float = 1.0,
        lambda_aux_mask: float = 0.4,
        lambda_object_repulsion: float = 0.0,
        repulsion_sigma: float = 0.05,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_chamfer: bool = False
    ):
        """
        Initialize IVD loss.
        
        Args:
            lambda_human: Weight for human contact loss
            lambda_object: Weight for object coordinate loss
            lambda_aux_mask: Weight for auxiliary mask loss
            use_focal: Whether to use focal loss for human contacts
            use_chamfer: Whether to use chamfer loss for object coords
            focal_alpha: Focal loss alpha
            focal_gamma: Focal loss gamma
        """
        super().__init__()
        
        self.lambda_human = lambda_human
        self.lambda_object = lambda_object
        self.lambda_aux_mask = lambda_aux_mask
        self.lambda_object_repulsion = lambda_object_repulsion
        self.repulsion_sigma = repulsion_sigma
        
        # Human contact loss
        self.human_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        # Object coordinate loss
        if use_chamfer:
            self.object_loss = ChamferLoss(bidirectional=True)
        else:
            self.object_loss = None
        
        # Auxiliary mask loss
        self.mask_bce = nn.BCEWithLogitsLoss()
        self.mask_dice = DiceLoss()
    
    def compute_human_loss(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute human contact classification loss.
        
        Args:
            pred_logits: (B, 87) predicted logits
            target: (B, 87) binary ground truth
            
        Returns:
            Scalar loss
        """
        return self.human_loss(pred_logits, target)
    
    def compute_object_loss(
        self,
        pred_logits: torch.Tensor,
        target_coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute object coordinate regression loss.
        
        Args:
            pred_logits: (B, K, 3) predicted coordinates
            target_coords: (B, K, 3) target coordinates
            mask: (B, K) optional mask for valid contacts
            
        Returns:
            Scalar loss
        """
        if isinstance(self.object_loss, ChamferLoss):
            return self.object_loss(pred_logits, target_coords, mask)
        # Default regression: SmoothL1 on valid contact indices only.
        # `mask` is expected to come from human_labels > 0 (shape: B x K).
        if mask is None:
            return F.smooth_l1_loss(pred_logits, target_coords, reduction='mean')
        mask = mask.float()
        per_coord = F.smooth_l1_loss(pred_logits, target_coords, reduction='none')  # (B, K, 3)
        per_kp = per_coord.sum(dim=-1)  # (B, K)
        return (per_kp * mask).sum() / (mask.sum() + 1e-8)

    def compute_object_repulsion_loss(
        self,
        pred_coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encourage predicted points to be spread out (avoid collapse).
        """
        if pred_coords is None:
            return torch.tensor(0.0, device=pred_coords.device if pred_coords is not None else 'cpu')
        B, K, _ = pred_coords.shape
        if K <= 1:
            return torch.tensor(0.0, device=pred_coords.device)

        dist = torch.cdist(pred_coords, pred_coords)  # (B, K, K)
        # ignore diagonal
        dist = dist + torch.eye(K, device=pred_coords.device).unsqueeze(0) * 1e6
        rep = torch.exp(- (dist ** 2) / (2 * (self.repulsion_sigma ** 2)))
        if mask is not None:
            m = mask.float()
            w = m.unsqueeze(2) * m.unsqueeze(1)
            rep = rep * w
            denom = w.sum() + 1e-8
            return rep.sum() / denom
        return rep.mean()

    # Affordance consistency losses removed: object coords are constrained by mask in forward.
    
    def compute_mask_loss(
        self,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary mask loss (Dice + BCE).
        
        Args:
            pred_mask: (B, 1, H, W) predicted mask logits
            target_mask: (B, 1, H, W) ground truth mask
            
        Returns:
            Scalar loss
        """
        if target_mask.device != pred_mask.device:
            target_mask = target_mask.to(pred_mask.device)
        if target_mask.dtype != pred_mask.dtype:
            target_mask = target_mask.to(dtype=pred_mask.dtype)

        bce_loss = self.mask_bce(pred_mask, target_mask)
        
        pred_sigmoid = torch.sigmoid(pred_mask)
        dice_loss = self.mask_dice(pred_sigmoid, target_mask)
        
        return bce_loss + dice_loss
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        compute_aux: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Args:
            predictions: Dictionary with:
                - 'human_logits': (B, 87) predicted contact logits
                - 'human_contact': (B, 87) predicted contact probabilities
                - 'object_coords': (B, K, 3) predicted coordinates
                - 'human_mask': (B*J, 1, H, W) auxiliary human mask (optional)
                - 'object_mask': (B*J, 1, H, W) auxiliary object mask (optional)
            targets: Dictionary with:
                - 'human_labels': (B, 87) binary contact labels
                - 'object_coords': (B, K, 3) target coordinates
                - 'human_mask_gt': (B*J, 1, H, W) ground truth human mask (optional)
                - 'object_mask_gt': (B*J, 1, H, W) ground truth object mask (optional)
            compute_aux: Whether to compute auxiliary mask loss
            
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        
        # Human contact loss
        if 'human_logits' in predictions and 'human_labels' in targets:
            losses['human_loss'] = self.compute_human_loss(
                predictions['human_logits'],
                targets['human_labels']
            )
        else:
            losses['human_loss'] = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        # Object index loss
        if 'object_coords' in predictions and 'object_coords' in targets:
            losses['object_loss'] = self.compute_object_loss(
                predictions['object_coords'],
                targets['object_coords'],
                targets.get('object_mask', None)
            )
        else:
            losses['object_loss'] = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        if self.lambda_object_repulsion > 0 and 'object_coords' in predictions:
            losses['object_repulsion_loss'] = self.compute_object_repulsion_loss(
                predictions['object_coords'],
                targets.get('object_mask', None)
            )
        else:
            losses['object_repulsion_loss'] = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        # Auxiliary mask losses
        if compute_aux:
            if 'human_mask' in predictions and 'human_mask_gt' in targets:
                losses['human_mask_loss'] = self.compute_mask_loss(
                    predictions['human_mask'],
                    targets['human_mask_gt']
                )
            else:
                losses['human_mask_loss'] = torch.tensor(0.0, device=next(iter(predictions.values())).device)
            
            if 'object_mask' in predictions and 'object_mask_gt' in targets:
                losses['object_mask_loss'] = self.compute_mask_loss(
                    predictions['object_mask'],
                    targets['object_mask_gt']
                )
            else:
                losses['object_mask_loss'] = torch.tensor(0.0, device=next(iter(predictions.values())).device)
            
            aux_loss = losses['human_mask_loss'] + losses['object_mask_loss']
        else:
            aux_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
            losses['human_mask_loss'] = aux_loss
            losses['object_mask_loss'] = aux_loss

        # Total loss
        losses['total_loss'] = (
            self.lambda_human * losses['human_loss'] +
            self.lambda_object * losses['object_loss'] +
            self.lambda_object_repulsion * losses['object_repulsion_loss'] +
            self.lambda_aux_mask * aux_loss
        )
        
        return losses


class WeightedBCELoss(nn.Module):
    """Weighted BCE loss for imbalanced binary classification."""
    
    def __init__(self, pos_weight: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, N) predicted logits
            target: (B, N) binary targets
        """
        pos_weight = torch.tensor([self.pos_weight], device=pred.device)
        return F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=pos_weight
        )


def compute_contact_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute contact prediction accuracy metrics.
    
    Args:
        pred: (B, 87) predicted probabilities
        target: (B, 87) binary targets
        threshold: Classification threshold
        
    Returns:
        Dictionary with accuracy metrics
    """
    pred_binary = (pred > threshold).float()
    
    # Overall accuracy
    accuracy = (pred_binary == target).float().mean().item()
    
    # Per-point accuracy
    per_point_acc = (pred_binary == target).float().mean(dim=0)
    
    # Precision, Recall for positive class
    tp = ((pred_binary == 1) & (target == 1)).float().sum()
    fp = ((pred_binary == 1) & (target == 0)).float().sum()
    fn = ((pred_binary == 0) & (target == 1)).float().sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'accuracy': accuracy,
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'per_point_acc_mean': per_point_acc.mean().item()
    }
