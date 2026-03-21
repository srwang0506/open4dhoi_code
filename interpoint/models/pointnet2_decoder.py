import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2 import PointNetSetAbstractionMsg, PointNetFeaturePropagation

class PointNet2KPDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.fp4 = PointNetFeaturePropagation(in_channel=512+512+256+256, mlp=[256,256])
        self.fp3 = PointNetFeaturePropagation(in_channel=128+128+256, mlp=[256,256])
        self.fp2 = PointNetFeaturePropagation(in_channel=32+64+256,  mlp=[256,256])
        self.fp1 = PointNetFeaturePropagation(in_channel=256,        mlp=[256,256,256])

        self.bn = nn.BatchNorm1d(256)



    def forward(self, context_feat,encode_feature):


        l0_xyz,l1_xyz, l2_xyz, l1_points, l2_points=encode_feature


        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        l0_points = self.bn(l0_points)
        l0_points = l0_points.transpose(1, 2)

        context_norm=F.normalize(context_feat,p=2,dim=-1)
        point_norm= F.normalize(l0_points,p=2,dim=-1)
        # print('context_norm',context_norm.shape)
        # print('point_norm',point_norm.shape)
        cross_feature=torch.einsum('bmd,bnd->bmn', context_norm, point_norm)



        return cross_feature


class PointNet2FeatureDecoder(nn.Module):
    def __init__(self, in_dim: int = 256, context_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        self.fp4 = PointNetFeaturePropagation(in_channel=512+512+256+256, mlp=[256,256])
        self.fp3 = PointNetFeaturePropagation(in_channel=128+128+256, mlp=[256,256])
        self.fp2 = PointNetFeaturePropagation(in_channel=32+64+256,  mlp=[256,256])
        self.fp1 = PointNetFeaturePropagation(in_channel=256,        mlp=[256,256,256])

        self.bn = nn.BatchNorm1d(256)
        self.context_proj = nn.Linear(context_dim, in_dim) if context_dim != in_dim else nn.Identity()

    def forward(self, encode_feature):
        l0_xyz, l1_xyz, l2_xyz, l1_points, l2_points = encode_feature

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        l0_points = self.bn(l0_points)  # (B, C, N)
        return {
            'features': l0_points.transpose(1, 2),
            'xyz': l0_xyz.transpose(1, 2)
        }


class TemplatePointProvider(nn.Module):
    def __init__(
        self,
        num_points: int,
        d_model: int,
        context_dim: int,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_points = num_points
        self.d_model = d_model

        self.template_feat = nn.Parameter(torch.randn(1, num_points, d_model) * 0.02)
        self.template_xyz = nn.Parameter(torch.randn(1, num_points, 3) * 0.02)

        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model * 2)
        )

    def forward(self, context_feat: torch.Tensor):
        B = context_feat.shape[0]
        feat = self.template_feat.expand(B, -1, -1)
        xyz = self.template_xyz.expand(B, -1, -1)

        gamma_beta = self.context_proj(context_feat)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        feat = feat * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        return {
            'features': feat,
            'xyz': xyz
        }


class AffordanceDecoder(nn.Module):
    def __init__(self, d_model: int, context_dim: int, dropout: float = 0.3):
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model * 2)
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        self.eps = 1e-5

    def forward(self, point_feat: torch.Tensor, context_feat: torch.Tensor):
        """
        Args:
            point_feat: (B, N, d_model)
            context_feat: (B, C)
        """
        gamma_beta = self.context_proj(context_feat)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        mean = point_feat.mean(dim=1, keepdim=True)
        var = point_feat.var(dim=1, keepdim=True, unbiased=False)
        feat_norm = (point_feat - mean) / torch.sqrt(var + self.eps)
        feat = feat_norm * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        logits = self.mlp(feat).squeeze(-1)
        return {
            'logits': logits,
            'features': feat
        }
