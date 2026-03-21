import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import open3d as o3d


def _load_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _first_center(obj_poses: Dict[str, Any]) -> np.ndarray:
    center = obj_poses.get("center")
    if center is None:
        raise KeyError("obj_poses.json missing key: center")

    # Old format: center is a list of per-frame [x,y,z]
    if isinstance(center, list) and center and isinstance(center[0], list):
        return np.asarray(center[0], dtype=np.float32)

    # New/single-frame format: center is a single [x,y,z]
    if isinstance(center, list) and len(center) == 3 and all(isinstance(x, (int, float)) for x in center):
        return np.asarray(center, dtype=np.float32)

    raise ValueError(f"Unsupported center format in obj_poses.json: {type(center)}")


def build_obj_init(
    obj_org_path: Path,
    obj_poses_path: Path,
    merged_path: Path,
    out_path: Path,
) -> None:
    if not obj_org_path.exists():
        raise FileNotFoundError(f"obj_org.obj not found: {obj_org_path}")
    if not obj_poses_path.exists():
        raise FileNotFoundError(f"obj_poses.json not found: {obj_poses_path}")
    if not merged_path.exists():
        raise FileNotFoundError(f"kp_record_merged.json not found: {merged_path}")

    obj_poses = _load_json(obj_poses_path)
    merged = _load_json(merged_path)

    scale_pose = float(obj_poses.get("scale", 1.0))
    scale_user = float(merged.get("object_scale", 1.0))
    t0 = _first_center(obj_poses)

    mesh = o3d.io.read_triangle_mesh(str(obj_org_path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"Failed to load mesh vertices from: {obj_org_path}")

    verts = np.asarray(mesh.vertices, dtype=np.float32)

    # Match optimize.py semantics:
    # 1) scale by obj_poses.scale
    # 2) recenter by mean
    # 3) apply merged object_scale around centroid (centroid is ~0 after recenter)
    # 4) translate by first-frame center
    verts_scaled = verts * scale_pose
    center_obj = verts_scaled.mean(axis=0)
    verts_centered = (verts_scaled - center_obj) * scale_user
    verts_init = verts_centered + t0

    mesh_out = o3d.geometry.TriangleMesh()
    mesh_out.vertices = o3d.utility.Vector3dVector(verts_init.astype(np.float64))
    mesh_out.triangles = mesh.triangles
    if mesh.has_vertex_colors():
        mesh_out.vertex_colors = mesh.vertex_colors
    if mesh.has_vertex_normals():
        mesh_out.vertex_normals = mesh.vertex_normals
    mesh_out.compute_vertex_normals()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_triangle_mesh(str(out_path), mesh_out)
    if not ok:
        raise RuntimeError(f"Failed to write obj_init.obj to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True, help="session folder containing obj_org.obj / align/obj_poses.json / kp_record_merged.json")
    parser.add_argument("--obj_org", default="obj_org.obj")
    parser.add_argument("--obj_poses", default=os.path.join("align", "obj_poses.json"))
    parser.add_argument("--merged", default="kp_record_merged.json")
    parser.add_argument("--out", default="obj_init.obj")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    build_obj_init(
        obj_org_path=video_dir / args.obj_org,
        obj_poses_path=video_dir / args.obj_poses,
        merged_path=video_dir / args.merged,
        out_path=video_dir / args.out,
    )
    print(f"Saved: {video_dir / args.out}")


if __name__ == "__main__":
    main()
