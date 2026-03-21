from .pointnet2 import PointNetSetAbstractionMsg,PointNetFeaturePropagation
# import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class PointNetv2Encoder(nn.Module):
    def __init__(
        self, num_class=1, in_channel=None, with_decoder=True, out_dim=128,
        sa1_params=None, sa2_params=None, sa3_params=None, sa4_params=None,
        fp1_params=None, fp2_params=None, fp3_params=None, fp4_params=None
    ):
        super(PointNetv2Encoder, self).__init__()

        self.in_channel = 3 if in_channel is None else in_channel
        #self.in_channel = 3
        self.with_decoder = with_decoder
        self.out_dim = out_dim

        if sa1_params is None:
            sa1_params = dict()
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=sa1_params.get("npoint", 1024), radius_list=sa1_params.get("radius_list", [0.05, 0.1]),
            nsample_list=sa1_params.get("nsample_list", [16, 32]), in_channel=self.in_channel,
            mlp_list=sa1_params.get("mlp_list", [[16, 16, 32], [32, 32, 64]])
        )

        if sa2_params is None:
            sa2_params = dict()
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=sa2_params.get("npoint", 256), radius_list=sa2_params.get("radius_list", [0.1, 0.2]),
            nsample_list=sa2_params.get("nsample_list", [16, 32]), in_channel=sa2_params.get("in_channel", 96),
            mlp_list=sa2_params.get("mlp_list", [[64, 64, 128], [64, 96, 128]])
        )

        if sa3_params is None:
            sa3_params = dict()
        self.sa3 = PointNetSetAbstractionMsg(
            npoint=sa3_params.get("npoint", 64), radius_list=sa3_params.get("radius_list", [0.2, 0.4]),
            nsample_list=sa3_params.get("nsample_list", [16, 32]), in_channel=sa3_params.get("in_channel", 256),
            mlp_list=sa3_params.get("mlp_list", [[128, 196, 256], [128, 196, 256]])
        )

        if sa4_params is None:
            sa4_params = dict()
        self.sa4 = PointNetSetAbstractionMsg(
            npoint=sa4_params.get("npoint", 16), radius_list=sa4_params.get("radius_list", [0.4, 0.8]),
            nsample_list=sa4_params.get("nsample_list", [16, 32]), in_channel=sa4_params.get("in_channel", 512),
            mlp_list=sa4_params.get("mlp_list", [[256, 256, 512], [256, 384, 512]])
        )

        if fp4_params is None:
            fp4_params = dict()
        self.fp4 = PointNetFeaturePropagation(
            in_channel=fp4_params.get("in_channel", 512+512+256+256), mlp=fp4_params.get("mlp", [256, 256])
        )

        if fp3_params is None:
            fp3_params = dict()
        self.fp3 = PointNetFeaturePropagation(
            in_channel=fp3_params.get("in_channel", 128+128+256), mlp=fp3_params.get("mlp", [256, 256])
        )

        if fp2_params is None:
            fp2_params = dict()
        self.fp2 = PointNetFeaturePropagation(
            in_channel=fp2_params.get("in_channel", 32+64+256), mlp=fp2_params.get("mlp", [256, 128])
        )

        if fp1_params is None:
            fp1_params = dict()
        self.fp1 = PointNetFeaturePropagation(
            in_channel=fp1_params.get("in_channel", 128), mlp=fp1_params.get("mlp", [128, 128, 128])
        )

        if self.with_decoder:
            self.decoder = self.create_decoder(256, 1)
        self.Linear = nn.Linear(256, 128 - 3)
    @staticmethod
    def create_decoder(in_dim, out_dim):
        return nn.Sequential(
            nn.Conv1d(in_dim, in_dim, 1),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(in_dim, out_dim, 1)
        )

    def forward(self, xyz):
        xyz = xyz.transpose(2, 1)
        l0_points = xyz
        #l0_points=None
        #print('l0_points',l0_points.shape)
        l0_xyz = xyz[:, :3, :]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print('l1_xyz',l1_xyz.shape)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        point_feature=[l0_xyz,l1_xyz, l2_xyz, l1_points, l2_points]

        feature=torch.cat((self.Linear(l2_points.transpose(1, 2)),l2_xyz.transpose(1, 2)),dim=-1)
        # x = x.permute(0, 2, 1)
        return feature,point_feature
