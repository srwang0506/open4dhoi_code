import torch
import torch.nn as nn
import torch.nn.functional as F


class IndexPredictionAdapter(nn.Module):
    """
    Adapter that exposes index-logit outputs for object contact prediction.
    It wraps the base IVD model without changing the original project modules.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, rgb: torch.Tensor, object_points: torch.Tensor):
        E_human, E_object = self.base.vlm(rgb, None)

        _, object_encode = self.base.object_pc_encoder(object_points)
        object_dec = self.base.object_point_decoder(object_encode)
        object_xyz = object_dec['xyz']
        object_feat = self.base.point_feat_proj(object_dec['features'])
        object_feat = self.base._fuse_semantic(object_feat, E_object, self.base.object_sem_film)

        human_points = self.base.human_point_provider(E_human)
        human_feat = human_points['features']
        human_xyz = human_points['xyz']
        human_feat = self.base._fuse_semantic(human_feat, E_human, self.base.object_sem_film)

        tr_out = self.base.transformer(human_feat, human_xyz, object_feat, object_xyz)
        human_queries = tr_out['human_queries']
        object_queries = tr_out['object_queries']

        human_logits = self.base.prediction_head.human_head(human_queries).squeeze(-1)
        q = self.base.prediction_head.object_query_proj(object_queries)
        object_logits = torch.matmul(q, object_feat.transpose(1, 2)) * self.base.prediction_head.object_attn_scale

        return {
            'human_logits': human_logits,
            'object_logits': object_logits,
            'object_xyz': object_xyz,
        }


class IndexPredictionLoss(nn.Module):
    def __init__(
        self,
        human_loss_fn: nn.Module,
        cls_tau: float = 1.0,
        soft_sigma: float = 0.06,
        lambda_human: float = 1.0,
        lambda_object: float = 1.0,
    ):
        super().__init__()
        self.human_loss_fn = human_loss_fn
        self.cls_tau = cls_tau
        self.soft_sigma = soft_sigma
        self.lambda_human = lambda_human
        self.lambda_object = lambda_object

    def _build_soft_targets(self, gt_coords: torch.Tensor, object_xyz: torch.Tensor) -> torch.Tensor:
        d = torch.cdist(gt_coords, object_xyz)  # (B,K,N)
        if self.soft_sigma <= 0:
            nn_idx = d.argmin(dim=-1)
            return F.one_hot(nn_idx, num_classes=object_xyz.shape[1]).float()
        logits = - (d ** 2) / (2.0 * self.soft_sigma * self.soft_sigma)
        return torch.softmax(logits, dim=-1)

    def _compute_object_cls_loss(self, object_logits: torch.Tensor, soft_targets: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(object_logits / max(self.cls_tau, 1e-6), dim=-1)
        per_query = -(soft_targets * log_probs).sum(dim=-1)
        denom = object_mask.sum() + 1e-8
        return (per_query * object_mask).sum() / denom

    def forward(self, outputs: dict, human_labels: torch.Tensor, gt_coords: torch.Tensor):
        object_mask = (human_labels > 0.5).float()

        human_loss = self.human_loss_fn(outputs['human_logits'], human_labels)
        soft_targets = self._build_soft_targets(gt_coords, outputs['object_xyz'])
        object_loss = self._compute_object_cls_loss(outputs['object_logits'], soft_targets, object_mask)

        total = self.lambda_human * human_loss + self.lambda_object * object_loss
        return {
            'total_loss': total,
            'human_loss': human_loss,
            'object_cls_loss': object_loss,
        }

    @torch.no_grad()
    def aux_metrics(self, object_logits: torch.Tensor):
        w = torch.softmax(object_logits / max(self.cls_tau, 1e-6), dim=-1)
        ent = -(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum(dim=-1) / torch.log(
            torch.tensor(w.shape[-1], device=w.device, dtype=w.dtype)
        )
        arg = w.argmax(dim=-1)
        uniq = [torch.unique(arg[b]).numel() for b in range(arg.shape[0])]
        return {
            'object_entropy': ent.mean().item(),
            'object_argmax_unique': float(sum(uniq)) / max(1, len(uniq)),
        }
