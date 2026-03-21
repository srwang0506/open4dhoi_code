import json
import os
import numpy as np
import torch
from tqdm import tqdm

from .optimizer_part import VideoBodyObjectOptimizer
from .hoi_solver import HOISolver
from .kp_common import (
    model,
    resource_path,
    apply_initial_transform_to_mesh,
    apply_initial_transform_to_points,
)
from copy import deepcopy
from typing import Optional
import open3d as o3d


def save_ls_check_mesh(
    video_dir: str,
    frame_idx: int,
    body_params,
    hand_poses,
    icp_transform_matrix,
    obj_org,
    start_frame: int
):
    """
    Save a quick 3D mesh check after least squares solve.

    Args:
        video_dir: Path to video directory
        frame_idx: Absolute frame index to check
        body_params: Body parameters dict
        hand_poses: Hand poses dict
        icp_transform_matrix: ICP transformation matrix for the frame
        obj_org: Original object mesh
        start_frame: Start frame index
    """
    check_dir = os.path.join(video_dir, "ls_check")
    os.makedirs(check_dir, exist_ok=True)

    local_idx = frame_idx - start_frame
    if local_idx < 0 or local_idx >= len(icp_transform_matrix):
        print(f"[WARN] Frame {frame_idx} out of range for LS check, skipping")
        return

    print(f"[INFO] Saving LS check mesh for frame {frame_idx} to {check_dir}")

    # 1. Generate human mesh using SMPLX
    try:
        # Get model device
        model_device = next(model.parameters()).device

        body_pose = body_params['body_pose'][frame_idx].reshape(1, -1)
        betas = body_params['betas'][frame_idx].reshape(1, 10)
        global_orient = body_params['global_orient'][frame_idx].reshape(1, 3)
        transl = body_params['transl'][frame_idx].reshape(1, 3)

        # Get hand poses
        frame_key = str(frame_idx)
        if frame_key in hand_poses:
            left_hand_data = hand_poses[frame_key].get('left_hand', [[0.0, 0.0, 0.0]] * 15)
            right_hand_data = hand_poses[frame_key].get('right_hand', [[0.0, 0.0, 0.0]] * 15)
        else:
            left_hand_data = [[0.0, 0.0, 0.0]] * 15
            right_hand_data = [[0.0, 0.0, 0.0]] * 15

        # Flatten hand poses to (45,) arrays
        left_hand = np.array(left_hand_data, dtype=np.float32).reshape(-1)
        right_hand = np.array(right_hand_data, dtype=np.float32).reshape(-1)

        # Convert to torch tensors on the same device as model
        if not isinstance(body_pose, torch.Tensor):
            body_pose = torch.tensor(body_pose, dtype=torch.float32, device=model_device)
        else:
            body_pose = body_pose.to(model_device)

        if not isinstance(betas, torch.Tensor):
            betas = torch.tensor(betas, dtype=torch.float32, device=model_device)
        else:
            betas = betas.to(model_device)

        if not isinstance(global_orient, torch.Tensor):
            global_orient = torch.tensor(global_orient, dtype=torch.float32, device=model_device)
        else:
            global_orient = global_orient.to(model_device)

        if not isinstance(transl, torch.Tensor):
            transl = torch.tensor(transl, dtype=torch.float32, device=model_device)
        else:
            transl = transl.to(model_device)

        left_hand_pose = torch.tensor(left_hand, dtype=torch.float32, device=model_device).reshape(1, -1)
        right_hand_pose = torch.tensor(right_hand, dtype=torch.float32, device=model_device).reshape(1, -1)

        # Generate SMPLX mesh with hand poses
        with torch.no_grad():
            output = model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                return_verts=True,
            )

        human_vertices = output.vertices[0].cpu().numpy()
        human_faces = model.faces

        # Save human mesh
        human_mesh = o3d.geometry.TriangleMesh()
        human_mesh.vertices = o3d.utility.Vector3dVector(human_vertices)
        human_mesh.triangles = o3d.utility.Vector3iVector(human_faces)
        human_mesh.compute_vertex_normals()

        human_path = os.path.join(check_dir, f"human_frame_{frame_idx:05d}.obj")
        o3d.io.write_triangle_mesh(human_path, human_mesh)
        print(f"  ✓ Saved human mesh: {human_path}")

    except Exception as e:
        print(f"[ERROR] Failed to save human mesh: {e}")
        import traceback
        traceback.print_exc()

    # 2. Transform and save object mesh
    try:
        icp_mat = icp_transform_matrix[local_idx]
        R = icp_mat[:3, :3]
        t = icp_mat[:3, 3]

        obj_vertices = np.asarray(obj_org.vertices, dtype=np.float32)
        obj_faces = np.asarray(obj_org.triangles)

        # Apply transformation: v' = R @ v + t
        obj_vertices_transformed = obj_vertices @ R.T + t

        # Save object mesh
        obj_mesh = o3d.geometry.TriangleMesh()
        obj_mesh.vertices = o3d.utility.Vector3dVector(obj_vertices_transformed)
        obj_mesh.triangles = o3d.utility.Vector3iVector(obj_faces)

        if obj_org.has_vertex_colors():
            obj_mesh.vertex_colors = obj_org.vertex_colors

        obj_mesh.compute_vertex_normals()

        obj_path = os.path.join(check_dir, f"object_frame_{frame_idx:05d}.obj")
        o3d.io.write_triangle_mesh(obj_path, obj_mesh)
        print(f"  ✓ Saved object mesh: {obj_path}")

        # Print some debug info
        print(f"  [DEBUG] Human translation: {transl.cpu().numpy()}")
        print(f"  [DEBUG] Object center (t): {t}")
        print(f"  [DEBUG] Distance between human and object: {np.linalg.norm(transl.cpu().numpy()[0] - t):.4f}")

    except Exception as e:
        print(f"[ERROR] Failed to save object mesh: {e}")
        import traceback
        traceback.print_exc()


def kp_use_new(
    output,
    hand_poses,
    body_poses,
    global_body_poses,
    obj_orgs,
    sampled_orgs,
    centers_depth=None,
    human_part=None,
    K=None,
    start_frame=0,
    end_frame=0,
    video_dir=None,
    is_static_object=False,
    kp_record_path: str = None,
    save_ls_meshes: bool = False,
    ls_mesh_dir: Optional[str] = None,
    use_least_squares_only: bool = False,
    best_frame_override: int = None
):
    if not save_ls_meshes:
        ls_mesh_dir = None
    if kp_record_path is None or not os.path.exists(kp_record_path):
        raise FileNotFoundError(f"kp_record_path not found: {kp_record_path}")

    # Default handling for optional parameters
    if centers_depth is None:
        centers_depth = np.zeros((end_frame - start_frame, 3), dtype=np.float32)
    if human_part is None:
        human_part = {}

    with open(kp_record_path, "r", encoding="utf-8") as f:
        merged = json.load(f)

    body_params = body_poses
    global_body_params = global_body_poses
    seq_length = end_frame - start_frame

    # Determine best_frame for static objects
    best_frame = 0
    if is_static_object:
        if best_frame_override is not None:
            # User specified best_frame (absolute frame index), convert to relative index
            best_frame = best_frame_override - start_frame
            if best_frame < 0 or best_frame >= seq_length:
                print(f"[WARN] best_frame_override={best_frame_override} out of range [{start_frame}, {end_frame}), using 0")
                best_frame = 0
            else:
                print(f"[INFO] Using user-specified best_frame: {best_frame_override} (relative: {best_frame})")
        else:
            # Auto-detect: find frame with most annotations
            max_count = -1
            for i in range(seq_length):
                frame_id = start_frame + i
                key = f"{frame_id:05d}"
                annotation = merged.get(key, {"2D_keypoint": []})
                num_2d = len(annotation.get("2D_keypoint", []))
                num_3d = 0
                for k in annotation.keys():
                    if k == "2D_keypoint":
                        continue
                    if k not in human_part:
                        continue
                    num_3d += 1
                total = num_2d + num_3d
                if total > max_count:
                    max_count = total
                    best_frame = i
            print(f"[INFO] Auto-detected best_frame: {start_frame + best_frame} (relative: {best_frame}, annotations: {max_count})")

    object_points_idx = []
    body_points_idx = []
    pairs_2d = []
    object_points = []
    image_points = []
    body_kp_name = []
    hoi_solver = HOISolver(model_folder=resource_path('video_optimizer/smpl_models/SMPLX_NEUTRAL.npz'))

    for i in tqdm(range(seq_length)):
        frame_id = start_frame + i
        key = f"{frame_id:05d}"
        annotation = merged.get(key, {"2D_keypoint": []})

        if annotation.get("2D_keypoint"):
            current_idx = best_frame if is_static_object else i
            point_indices = [p[0] for p in annotation["2D_keypoint"]]
            image_coords = [np.array(p[1]) for p in annotation["2D_keypoint"]]
            object_verts = np.array(deepcopy(obj_orgs[current_idx].vertices))[point_indices]
            depth_idx = min(current_idx, len(centers_depth) - 1)
            transformed_verts = apply_initial_transform_to_points(
                object_verts, centers_depth[depth_idx]
            )

            object_points.append(transformed_verts.astype(np.float32))
            image_points.append(np.array(image_coords, dtype=np.float32))
        else:
            object_points.append(np.array([]))
            image_points.append(np.array([]))
        object_idx = np.zeros((len(human_part), 2))
        for k, annot_index in annotation.items():
            if k == "2D_keypoint":
                continue
            if k not in human_part:
                continue
            body_kp_name.append(k)
            human_part_index = list(human_part.keys()).index(k)
            object_idx[human_part_index] = [annot_index, 1]

        pairs_2d.append(annotation.get("2D_keypoint", []))
        body_idx = [v['index'] for v in human_part.values()]
        object_points_idx.append(object_idx)
        body_points_idx.append(body_idx)

    hoi_interval = 1
    if is_static_object:
        frames_to_optimize = [best_frame]
    else:
        frames_to_optimize = list(range(0, seq_length, hoi_interval))
        if frames_to_optimize[-1] != seq_length - 1:
            frames_to_optimize.append(seq_length - 1)

    optimized_results = {}
    icp_transform_matrix = []
    joint_mapping = json.load(open(resource_path('video_optimizer/data/joint_reflect.json')))

    for i in frames_to_optimize:
        frame_idx = i + start_frame

        # Bounds check: skip if frame index is out of range
        if frame_idx >= len(body_params["body_pose"]):
            print(f"[SKIP] Frame {frame_idx} exceeds body_params length ({len(body_params['body_pose'])})")
            continue

        obj_src_idx = best_frame if is_static_object else i
        depth_idx = min(obj_src_idx, len(centers_depth) - 1)
        obj_init = apply_initial_transform_to_mesh(
            obj_orgs[obj_src_idx], centers_depth[depth_idx]
        )
        obj_init_sample = apply_initial_transform_to_mesh(
            sampled_orgs[obj_src_idx], centers_depth[depth_idx]
        )
        result = hoi_solver.solve_hoi(
            obj_init,
            obj_init_sample,
            body_params,
            global_body_params,
            i,
            start_frame,
            end_frame,
            hand_poses,
            object_points_idx,
            body_points_idx,
            object_points,
            image_points,
            joint_mapping,
            K=K.cpu().numpy() if hasattr(K, "cpu") else K,
            save_meshes=save_ls_meshes,
            debug_out_dir=ls_mesh_dir,
            debug_prefix=f"frame_{i + start_frame:05d}"
        )
        body_params['global_orient'][frame_idx] = result['global_orient'].detach().cpu()
        body_params['body_pose'][frame_idx] = result['body_pose'].detach().cpu()
        icp_transform_matrix.append(result['icp_transform_matrix'])

    if is_static_object:
        if len(icp_transform_matrix) > 0:
            icp_transform_matrix = [icp_transform_matrix[0] for _ in range(seq_length)]
        first_frame_obj = obj_orgs[best_frame]
        first_frame_sampled = sampled_orgs[best_frame]
        for i in range(seq_length):
            obj_orgs[i] = first_frame_obj
            sampled_orgs[i] = first_frame_sampled

    # If use_least_squares_only is True, skip Adam optimization
    if use_least_squares_only:
        print("[INFO] use_least_squares_only=True, skipping Adam optimization")
        # Return results from least squares solver only
        # R_finals and t_finals need to be extracted from icp_transform_matrix
        R_finals = []
        t_finals = []
        for mat in icp_transform_matrix:
            R_finals.append(mat[:3, :3])
            t_finals.append(mat[:3, 3])

        # Convert hand_poses from dict format to list format to match optimizer output
        hand_poses_list = []
        for i in range(seq_length):
            frame_idx = start_frame + i
            frame_key = str(frame_idx)
            if frame_key in hand_poses:
                hand_poses_list.append({
                    "left_hand": hand_poses[frame_key]["left_hand"],
                    "right_hand": hand_poses[frame_key]["right_hand"]
                })
            else:
                # Fallback: zero hand pose
                hand_poses_list.append({
                    "left_hand": [[0.0, 0.0, 0.0]] * 15,
                    "right_hand": [[0.0, 0.0, 0.0]] * 15
                })

        # Save a quick check mesh for frame 20 (if it exists in the range)
        check_frame = 20
        if start_frame <= check_frame < end_frame:
            obj_src_idx = best_frame if is_static_object else (check_frame - start_frame)
            save_ls_check_mesh(
                video_dir=video_dir,
                frame_idx=check_frame,
                body_params=body_params,
                hand_poses=hand_poses,
                icp_transform_matrix=icp_transform_matrix,
                obj_org=obj_orgs[obj_src_idx],
                start_frame=start_frame
            )
        else:
            print(f"[INFO] Frame 20 not in range [{start_frame}, {end_frame}), skipping LS check")

        optimized_params = None
        return body_params, hand_poses_list, R_finals, t_finals, optimized_params

    optimizer_args = {
        "body_params": body_params,
        "global_body_params": global_body_params,
        "hand_params": hand_poses,
        "object_points_idx": object_points_idx,
        "body_points_idx": body_points_idx,
        "body_kp_name": body_kp_name,
        "pairs_2d": pairs_2d,
        "object_meshes": obj_orgs,
        "sampled_obj_meshes": sampled_orgs,
        "icp_transform_matrix": icp_transform_matrix,
        "smpl_model": model,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "video_dir": video_dir,
        "lr": 0.1,
        "is_static_object": is_static_object,
        "best_frame": best_frame if is_static_object else None,
    }
    optimizer = VideoBodyObjectOptimizer(**optimizer_args)
    optimizer.optimize(steps=30, print_every=5)
    optimized_params = optimizer.get_optimized_parameters()
    optimizer.create_visualization_video(
        os.path.join(video_dir, "optimized_frames"),
        K=K,
        video_path=os.path.join(video_dir, "optimize_video.mp4"),
        clear=False
    )
    body_params, hand_poses, R_finals, t_finals = optimizer.get_optimize_result()
    return body_params, hand_poses, R_finals, t_finals, optimized_params
