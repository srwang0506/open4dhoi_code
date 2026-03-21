"""
Evaluation metrics for InterActVLM-Discrete (IVD)
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    average_precision_score
)


class ContactMetrics:
    """
    Evaluation metrics for human-object interaction contact prediction.

    Computes metrics for:
    - Human contact classification (87 binary labels for SMPL-X keypoints)
    - Object contact regression (3D coordinates)
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize metrics calculator.
        
        Args:
            threshold: Threshold for binary classification
        """
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.human_preds = []
        self.human_labels = []
        self.human_probs = []
        
        self.object_preds = []
        self.object_labels = []
    
    def update(
        self,
        human_pred: torch.Tensor,
        human_label: torch.Tensor,
        object_pred: Optional[torch.Tensor] = None,
        object_label: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with new predictions.

        Args:
            human_pred: (B, 87) predicted contact probabilities
            human_label: (B, 87) ground truth binary labels
            object_pred: (B, K, 3) predicted object contact coordinates
            object_label: (B, K, 3) ground truth object contact coordinates
        """
        self.human_probs.append(human_pred.detach().cpu())
        self.human_preds.append((human_pred > self.threshold).float().detach().cpu())
        self.human_labels.append(human_label.detach().cpu())
        
        if object_pred is not None and object_label is not None:
            self.object_preds.append(object_pred.detach().cpu())
            self.object_labels.append(object_label.detach().cpu())
    
    def compute_human_metrics(self) -> Dict[str, float]:
        """
        Compute classification metrics for human contact prediction.
        
        Returns:
            Dictionary of metrics
        """
        preds = torch.cat(self.human_preds, dim=0).numpy()
        labels = torch.cat(self.human_labels, dim=0).numpy()
        probs = torch.cat(self.human_probs, dim=0).numpy()
        
        # Flatten for per-point metrics
        preds_flat = preds.flatten()
        labels_flat = labels.flatten()
        probs_flat = probs.flatten()
        
        # Accuracy
        accuracy = accuracy_score(labels_flat, preds_flat)
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_flat, preds_flat, average='binary', zero_division=0
        )
        
        # AUC-ROC (if we have both classes)
        try:
            if len(np.unique(labels_flat)) > 1:
                auc_roc = roc_auc_score(labels_flat, probs_flat)
            else:
                auc_roc = 0.0
        except:
            auc_roc = 0.0
        
        # Average Precision
        try:
            if labels_flat.sum() > 0:
                avg_precision = average_precision_score(labels_flat, probs_flat)
            else:
                avg_precision = 0.0
        except:
            avg_precision = 0.0
        
        # Per-sample accuracy
        per_sample_acc = (preds == labels).all(axis=1).mean()
        
        # Mean per-point accuracy
        per_point_acc = (preds == labels).mean(axis=0)
        mean_per_point_acc = per_point_acc.mean()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc,
            'avg_precision': avg_precision,
            'per_sample_acc': per_sample_acc,
            'mean_per_point_acc': mean_per_point_acc
        }
    
    def compute_object_metrics(self) -> Dict[str, float]:
        """
        Compute regression metrics for object contact prediction.
        
        Returns:
            Dictionary of metrics
        """
        if len(self.object_preds) == 0:
            return {}
        
        preds = torch.cat(self.object_preds, dim=0).numpy()  # (N, K, 3)
        labels = torch.cat(self.object_labels, dim=0).numpy()  # (N, K, 3)
        
        # L2 distance (Euclidean error)
        l2_dist = np.sqrt(((preds - labels) ** 2).sum(axis=-1))  # (N, K)
        mean_l2 = l2_dist.mean()
        
        # Per-query L2
        per_query_l2 = l2_dist.mean(axis=0)  # (K,)
        
        # Chamfer Distance (if K > 1)
        if preds.shape[1] > 1:
            chamfer = self._compute_chamfer_distance(preds, labels)
        else:
            chamfer = mean_l2
        
        # Accuracy at different thresholds
        acc_5cm = (l2_dist < 0.05).mean()
        acc_10cm = (l2_dist < 0.10).mean()
        acc_20cm = (l2_dist < 0.20).mean()
        
        return {
            'mean_l2': mean_l2,
            'chamfer': chamfer,
            'acc_5cm': acc_5cm,
            'acc_10cm': acc_10cm,
            'acc_20cm': acc_20cm,
            'per_query_l2': per_query_l2.tolist()
        }
    
    def _compute_chamfer_distance(
        self,
        pred: np.ndarray,
        label: np.ndarray
    ) -> float:
        """
        Compute Chamfer distance between predicted and ground truth points.
        
        Args:
            pred: (N, K, 3) predicted points
            label: (N, K, 3) ground truth points
            
        Returns:
            Mean Chamfer distance
        """
        N, K, _ = pred.shape
        
        total_chamfer = 0.0
        for i in range(N):
            p = pred[i]  # (K, 3)
            l = label[i]  # (K, 3)
            
            # Distance from pred to nearest label
            dist_p2l = np.sqrt(((p[:, None, :] - l[None, :, :]) ** 2).sum(axis=-1))
            min_p2l = dist_p2l.min(axis=1).mean()
            
            # Distance from label to nearest pred
            dist_l2p = np.sqrt(((l[:, None, :] - p[None, :, :]) ** 2).sum(axis=-1))
            min_l2p = dist_l2p.min(axis=1).mean()
            
            total_chamfer += (min_p2l + min_l2p) / 2
        
        return total_chamfer / N
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        human_metrics = self.compute_human_metrics()
        for k, v in human_metrics.items():
            metrics[f'human_{k}'] = v
        
        object_metrics = self.compute_object_metrics()
        for k, v in object_metrics.items():
            if not isinstance(v, list):
                metrics[f'object_{k}'] = v
        
        return metrics
    
    def __repr__(self) -> str:
        return f"ContactMetrics(threshold={self.threshold})"


def compute_per_part_metrics(
    pred: torch.Tensor,
    label: torch.Tensor,
    part_groups: Dict[str, list],
    threshold: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics per body part group.
    
    Args:
        pred: (B, 74) predicted contact probabilities
        label: (B, 74) ground truth labels
        part_groups: Dictionary mapping part names to keypoint indices
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics per body part
    """
    pred_np = pred.detach().cpu().numpy()
    label_np = label.detach().cpu().numpy()
    pred_binary = (pred_np > threshold).astype(float)
    
    results = {}
    
    for part_name, indices in part_groups.items():
        if len(indices) == 0:
            continue
        
        part_pred = pred_binary[:, indices].flatten()
        part_label = label_np[:, indices].flatten()
        part_probs = pred_np[:, indices].flatten()
        
        acc = accuracy_score(part_label, part_pred)
        
        try:
            if len(np.unique(part_label)) > 1:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    part_label, part_pred, average='binary', zero_division=0
                )
            else:
                precision, recall, f1 = 0.0, 0.0, 0.0
        except:
            precision, recall, f1 = 0.0, 0.0, 0.0
        
        results[part_name] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_points': len(indices)
        }
    
    return results
