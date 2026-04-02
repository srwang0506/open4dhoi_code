import copy
import os
import sys

import numpy as np
import open3d as o3d
import smplx
import torch


def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # type: ignore
    except Exception:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


_model_type = "smplx"
_model_folder = resource_path("video_optimizer/smpl_models/SMPLX_NEUTRAL.npz")
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smplx.create(
    _model_folder,
    model_type=_model_type,
    gender="neutral",
    num_betas=10,
    num_expression_coeffs=10,
    use_pca=False,
    flat_hand_mean=True,
).to(_device)


def apply_initial_transform_to_mesh(mesh: o3d.geometry.TriangleMesh, t: np.ndarray):
    mesh_copy = copy.deepcopy(mesh)
    verts = np.asarray(mesh_copy.vertices)
    transformed_verts = verts + t
    mesh_copy.vertices = o3d.utility.Vector3dVector(transformed_verts)
    return mesh_copy


def apply_initial_transform_to_points(points: np.ndarray, t: np.ndarray) -> np.ndarray:
    return points + t
