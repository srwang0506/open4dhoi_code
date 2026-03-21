import os
import sys
import cv2
import json
import argparse
import traceback
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import open3d as o3d
from io import BytesIO
from threading import Lock
from copy import deepcopy

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent

try:
    import yaml
except ImportError:
    yaml = None
try:
    from solver.kp_use_new import kp_use_new
    print("[Solver] kp_use_new module loaded successfully")
except Exception as e:
    import traceback
    print("=" * 60)
    print("[Solver] ERROR: Could not import kp_use_new")
    print(f"[Solver] Exception type: {type(e).__name__}")
    print(f"[Solver] Exception message: {e}")
    print("[Solver] Full traceback:")
    traceback.print_exc()
    print("=" * 60)
    kp_use_new = None

# Add CoTracker to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'co-tracker'))

try:
    import torch
    import smplx
    from cotracker.predictor import CoTrackerOnlinePredictor
    COTRACKER_AVAILABLE = True
    print("CoTracker and SMPLX imported successfully")
except ImportError as e:
    COTRACKER_AVAILABLE = False
    print(f"Dependency missing: {e}")

app = Flask(__name__, static_folder='static')

# ========== Configuration ==========

def _load_config():
    """Load constants from config.yaml (use defaults if missing)."""
    defaults = {
        'annotator': {'name': 'anonymous'},
        'mesh': {
            'obj_decimation_target_faces': 30000,
            'obj_decimate_if_vertices_above': 5000,
        },
        'server': {'host': '0.0.0.0', 'port': 5010},
        'video': {'default_fps': 30},
        'ivd_model': {
            'enabled': True,
            'checkpoint': '',
            'device': 'cuda',
            'threshold': 0.3,
            'use_lightweight_vlm': False,
        },
    }
    config_path = Path(__file__).resolve().parent / 'config.yaml'
    if yaml is not None and config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                for k, v in defaults.items():
                    if k in loaded and isinstance(loaded[k], dict):
                        defaults[k] = {**defaults[k], **loaded[k]}
                    elif k in loaded:
                        defaults[k] = loaded[k]
        except Exception as e:
            print(f"Warning: could not load config.yaml: {e}")
    return defaults

CONFIG = _load_config()

# ========== Global State ==========

VIDEO_PATH = ""
OBJ_PATH = ""
CAP = None
CAP_LOCK = Lock()
MESH_DATA = None
VIDEO_FRAMES = []
VIDEO_FRAMES_ENCODED = []
VIDEO_FPS = CONFIG['video']['default_fps']
VIDEO_TOTAL_FRAMES = 0
COTRACKER_MODEL = None
COTRACKER_LOCK = Lock()
TRACKED_POINTS = {}
SCENE_DATA = None
DATA_DIR = None  # Root directory containing session folders


# ========== Session Discovery ==========

def _scan_sessions(data_dir):
    """Scan data_dir for valid annotation sessions.

    A valid session folder must contain video.mp4 and obj_org.obj.
    Supports both flat layout (data_dir/session/) and categorized
    layout (data_dir/category/session/).
    """
    sessions = []
    data_path = Path(data_dir)
    if not data_path.exists():
        return sessions

    for entry in sorted(data_path.iterdir()):
        if not entry.is_dir():
            continue
        # Check if this entry is itself a session folder
        if (entry / 'video.mp4').exists():
            sessions.append({
                'name': entry.name,
                'path': str(entry),
                'category': '',
            })
            continue
        # Otherwise scan sub-directories (category/session layout)
        for session_dir in sorted(entry.iterdir()):
            if not session_dir.is_dir():
                continue
            if (session_dir / 'video.mp4').exists():
                sessions.append({
                    'name': session_dir.name,
                    'path': str(session_dir),
                    'category': entry.name,
                })
    return sessions


# ========== Video Session Loading ==========

def _load_video_session(video_dir: str) -> bool:
    """Load a video session from the given directory.

    Sets up VIDEO_PATH, OBJ_PATH, loads all frames into memory,
    initializes CoTracker, builds SCENE_DATA and MESH_DATA.
    """
    global VIDEO_PATH, OBJ_PATH, CAP, MESH_DATA, SCENE_DATA
    global VIDEO_FRAMES, VIDEO_FRAMES_ENCODED, VIDEO_FPS, VIDEO_TOTAL_FRAMES

    video_dir = str(video_dir)
    VIDEO_PATH = os.path.join(video_dir, "video.mp4")
    OBJ_PATH = os.path.join(video_dir, "obj_org.obj")

    # Release previous video handle
    if CAP is not None:
        try:
            CAP.release()
        except Exception:
            pass
    CAP = None
    VIDEO_FRAMES = []
    VIDEO_FRAMES_ENCODED = []
    VIDEO_FPS = CONFIG['video']['default_fps']
    VIDEO_TOTAL_FRAMES = 0
    MESH_DATA = None
    SCENE_DATA = None

    if not os.path.exists(VIDEO_PATH):
        print(f"Warning: Video not found at {VIDEO_PATH}")
        return False

    CAP = cv2.VideoCapture(VIDEO_PATH)
    if not CAP.isOpened():
        print(f"Failed to open video: {VIDEO_PATH}")
        CAP = None
        return False

    print(f"Video loaded: {VIDEO_PATH}")
    load_video_frames()
    init_cotracker()

    # Initialize Scene Data
    SCENE_DATA = SceneData(video_dir)
    SCENE_DATA.load()

    if os.path.exists(OBJ_PATH) and SCENE_DATA is not None and SCENE_DATA.obj_mesh_org is not None:
        MESH_DATA = load_mesh(SCENE_DATA.obj_mesh_org)
    else:
        print(f"Warning: Mesh not found or obj_mesh_org is None for video_dir {video_dir}")

    return True


def preprocess_obj_sample(obj_org, object_poses, seq_length):
    """Preprocess object mesh for all frames."""
    centers = np.array(object_poses.get('center', []))
    if len(centers) == 0:
        centers = np.zeros((seq_length, 3))
    elif len(centers) < seq_length:
        last_center = centers[-1] if len(centers) > 0 else np.zeros(3)
        centers = np.vstack([centers, np.tile(last_center, (seq_length - len(centers), 1))])

    # Extract initial translation from obj_poses
    t_raw = object_poses.get('t', None)
    if t_raw is not None:
        if isinstance(t_raw, list) and t_raw and isinstance(t_raw[0], list):
            t0 = np.array(t_raw[0], dtype=np.float64)
        elif isinstance(t_raw, list) and len(t_raw) == 3:
            t0 = np.array(t_raw, dtype=np.float64)
        else:
            t0 = np.zeros(3)
    else:
        t0 = np.zeros(3)

    obj_orgs = []
    center_objs = []
    scale = object_poses.get('scale', 1.0)

    for i in range(seq_length):
        obj_pcd = deepcopy(obj_org)
        if 'rotation' in object_poses and i < len(object_poses['rotation']):
            rotation_matrix = np.array(object_poses['rotation'][i])
            if rotation_matrix.shape == (3, 3):
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rotation_matrix
                obj_pcd.transform(transform_matrix)

        new_overts = np.asarray(obj_pcd.vertices)
        new_overts *= scale
        new_overts = new_overts - np.mean(new_overts, axis=0)
        new_overts = new_overts + t0
        center_objs.append(np.mean(new_overts, axis=0))
        obj_pcd.vertices = o3d.utility.Vector3dVector(new_overts)
        obj_orgs.append(obj_pcd)

    return obj_orgs, centers, center_objs


# ========== SceneData ==========

class SceneData:
    def __init__(self, video_dir):
        self.video_dir = video_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.smplx_model = None
        self.motion_data = None
        self.obj_poses = None
        self.obj_mesh_org = None
        self.obj_mesh_raw = None
        self.obj_orgs = []
        self.obj_orgs_world = []
        self.t_finals = None
        self.R_finals = []
        self.hand_poses = None
        self.total_frames = 0
        self.loaded = False
        self.obj_orgs_base_vertices = None
        self.object_scale_factor = 1.0

    def update_world_meshes(self):
        """Update cached world-space meshes based on current local meshes and transforms."""
        self.obj_orgs_world = []
        for i, obj in enumerate(self.obj_orgs):
            world_obj = deepcopy(obj)
            verts = np.asarray(world_obj.vertices)

            if len(self.R_finals) > i:
                R = self.R_finals[i]
            else:
                R = np.eye(3)

            if self.t_finals is not None and i < len(self.t_finals):
                t = np.array(self.t_finals[i])
            else:
                t = np.zeros(3)

            verts_transformed = np.matmul(verts, R.T) + t
            world_obj.vertices = o3d.utility.Vector3dVector(verts_transformed)
            self.obj_orgs_world.append(world_obj)

    def load(self):
        try:
            # Load SMPL-X Model
            model_path = os.path.join(app.root_path, 'asset', 'data', 'SMPLX_NEUTRAL.npz')
            if not os.path.exists(model_path):
                model_path = os.path.join(os.path.dirname(__file__), 'asset', 'data', 'SMPLX_NEUTRAL.npz')
            if os.path.exists(model_path):
                self.smplx_model = smplx.create(model_path, model_type='smplx',
                                                gender='neutral', num_betas=10,
                                                num_expression_coeffs=10,
                                                use_pca=False, flat_hand_mean=True).to(self.device)
            else:
                print(f"SMPL-X model not found at {model_path}")

            # Load Motion Data (prefer result_hand.pt)
            motion_path = os.path.join(self.video_dir, 'motion', 'result_hand.pt')
            fallback_motion_path = os.path.join(self.video_dir, 'motion', 'result.pt')
            if os.path.exists(motion_path):
                self.motion_data = torch.load(motion_path, map_location=self.device)
            elif os.path.exists(fallback_motion_path):
                self.motion_data = torch.load(fallback_motion_path, map_location=self.device)
                print(f"Motion data not found at {motion_path}, using fallback {fallback_motion_path}")
            else:
                self.motion_data = None
                print(f"Motion data not found at {motion_path} or {fallback_motion_path}")

            if self.motion_data is not None:
                params = self.motion_data.get('smpl_params_incam', {})
                if 'body_pose' in params:
                    self.total_frames = len(params['body_pose'])

                # Extract hand poses from motion data
                left_hand = params.get('left_hand_pose')
                right_hand = params.get('right_hand_pose')
                self.hand_poses = {}
                if left_hand is not None and right_hand is not None:
                    t_len = min(len(left_hand), len(right_hand))
                    for i in range(t_len):
                        lh = left_hand[i]
                        rh = right_hand[i]
                        if torch.is_tensor(lh):
                            lh = lh.cpu().numpy().tolist()
                        if torch.is_tensor(rh):
                            rh = rh.cpu().numpy().tolist()
                        self.hand_poses[str(i)] = {
                            "left_hand": lh,
                            "right_hand": rh,
                        }
                    if t_len > 0:
                        print(f"Loaded hand poses from result_hand.pt: {t_len} frames")
                if not self.hand_poses:
                    # Fallback: zero hand poses
                    zero_count = self.total_frames if self.total_frames > 0 else 0
                    for i in range(zero_count):
                        self.hand_poses[str(i)] = {
                            "left_hand": [[0.0, 0.0, 0.0]] * 15,
                            "right_hand": [[0.0, 0.0, 0.0]] * 15,
                        }
                    print(f"Hand poses missing; using zeros for {zero_count} frames")
            else:
                self.hand_poses = {}

            # Load Object Poses (try align/ then output/)
            obj_pose_paths = [
                os.path.join(self.video_dir, 'align', 'obj_poses.json'),
                os.path.join(self.video_dir, 'output', 'obj_poses.json')
            ]
            obj_pose_path = None
            for path in obj_pose_paths:
                if os.path.exists(path):
                    obj_pose_path = path
                    break

            if obj_pose_path:
                with open(obj_pose_path, 'r') as f:
                    self.obj_poses = json.load(f)
                print(f"Loaded object poses from {obj_pose_path}")
            else:
                print(f"Object poses not found in: {obj_pose_paths}")

            # Load Object Mesh
            obj_mesh_path = os.path.join(self.video_dir, 'obj_org.obj')
            if os.path.exists(obj_mesh_path):
                self.obj_mesh_raw = o3d.io.read_triangle_mesh(obj_mesh_path)
                self.obj_mesh_org = deepcopy(self.obj_mesh_raw)
                # Simplify for performance
                target_faces = CONFIG['mesh']['obj_decimation_target_faces']
                vert_threshold = CONFIG['mesh']['obj_decimate_if_vertices_above']
                if len(self.obj_mesh_org.vertices) > vert_threshold:
                    self.obj_mesh_org = self.obj_mesh_org.simplify_quadric_decimation(
                        target_number_of_triangles=target_faces)
            else:
                print(f"Object mesh not found at {obj_mesh_path}")

            # Preprocess object meshes for all frames
            if not self.obj_mesh_org:
                print("Warning: obj_mesh_org is None, cannot preprocess object meshes")
            if not self.obj_poses:
                print("Warning: obj_poses is None, cannot preprocess object meshes")
            if self.total_frames == 0:
                print("Warning: total_frames is 0, cannot preprocess object meshes")

            if self.obj_mesh_org and self.obj_poses and self.total_frames > 0:
                try:
                    self.obj_orgs, self.t_finals, _ = preprocess_obj_sample(
                        self.obj_mesh_org, self.obj_poses, self.total_frames
                    )
                    self.obj_orgs_base_vertices = [
                        np.asarray(obj.vertices).copy() for obj in self.obj_orgs
                    ]
                    self.R_finals = [np.eye(3) for _ in range(self.total_frames)]

                    if 'rotation' not in self.obj_poses:
                        self.obj_poses['rotation'] = [np.eye(3).tolist() for _ in range(self.total_frames)]

                    self.update_world_meshes()
                    print(f"Preprocessed {len(self.obj_orgs)} object meshes for {self.total_frames} frames")
                except Exception as e:
                    print(f"Error preprocessing object meshes: {e}")
                    traceback.print_exc()
                    self.obj_orgs = []
            else:
                print(f"Object preprocessing skipped: obj_mesh_org={self.obj_mesh_org is not None}, "
                      f"obj_poses={self.obj_poses is not None}, total_frames={self.total_frames}")
                self.obj_orgs = []

            if self.smplx_model and self.motion_data:
                self.loaded = True
                print(f"Scene data loaded (SMPL-X + Motion, {self.total_frames} frames)")
            else:
                print("Scene data incomplete (Missing model or motion file)")

        except Exception as e:
            print(f"Error loading scene data: {e}")
            traceback.print_exc()

    def apply_object_scale(self, scale_factor: float):
        """Rescale object meshes for all frames around their centers."""
        if scale_factor <= 0:
            raise ValueError("scale_factor must be > 0")
        if not self.obj_orgs:
            raise RuntimeError("No object meshes loaded to scale")

        if (self.obj_orgs_base_vertices is None or
                len(self.obj_orgs_base_vertices) != len(self.obj_orgs)):
            self.obj_orgs_base_vertices = [
                np.asarray(obj.vertices).copy() for obj in self.obj_orgs
            ]

        n_frames = min(self.total_frames, len(self.obj_orgs_base_vertices), len(self.obj_orgs))
        for frame_idx in range(n_frames):
            base_vertices = self.obj_orgs_base_vertices[frame_idx]
            if base_vertices.size == 0:
                continue
            center = np.mean(base_vertices, axis=0)
            vertices_final = (base_vertices - center) * scale_factor + center
            self.obj_orgs[frame_idx].vertices = o3d.utility.Vector3dVector(vertices_final)

        self.update_world_meshes()
        self.object_scale_factor = float(scale_factor)

    def get_hand_focus_view(self, frame_idx):
        if not self.loaded:
            return None, "Scene data not loaded"

        if frame_idx < 0 or (self.total_frames > 0 and frame_idx >= self.total_frames):
            return None, f"Frame index {frame_idx} out of range"

        try:
            params = self.motion_data['smpl_params_incam']

            body_pose = params['body_pose'][frame_idx:frame_idx+1]
            betas = params['betas'][frame_idx:frame_idx+1]
            global_orient = params['global_orient'][frame_idx:frame_idx+1]
            transl = params['transl'][frame_idx:frame_idx+1]

            if isinstance(body_pose, torch.Tensor):
                if body_pose.dim() == 3 and body_pose.shape[-1] == 3:
                    body_pose = body_pose.reshape(1, -1)
                elif body_pose.dim() == 2 and body_pose.shape[0] == 1:
                    pass
                elif body_pose.dim() == 1:
                    body_pose = body_pose.unsqueeze(0)

            if isinstance(betas, torch.Tensor):
                if betas.dim() == 1:
                    betas = betas.unsqueeze(0)

            if isinstance(global_orient, torch.Tensor):
                if global_orient.dim() == 1:
                    global_orient = global_orient.unsqueeze(0)
                elif global_orient.dim() == 3:
                    global_orient = global_orient.squeeze(1)

            if isinstance(transl, torch.Tensor):
                if transl.dim() == 1:
                    transl = transl.unsqueeze(0)

            left_hand_pose = None
            right_hand_pose = None

            if self.hand_poses and str(frame_idx) in self.hand_poses:
                hand_data = self.hand_poses[str(frame_idx)]
                if 'left_hand' in hand_data and hand_data['left_hand'] is not None:
                    left_hand_array = np.array(hand_data['left_hand'])
                    if left_hand_array.size > 0:
                        left_hand_pose = torch.from_numpy(
                            left_hand_array.reshape(-1, 3)[None, ...]
                        ).float().to(self.device)
                if 'right_hand' in hand_data and hand_data['right_hand'] is not None:
                    right_hand_array = np.array(hand_data['right_hand'])
                    if right_hand_array.size > 0:
                        right_hand_pose = torch.from_numpy(
                            right_hand_array.reshape(-1, 3)[None, ...]
                        ).float().to(self.device)

            if isinstance(body_pose, torch.Tensor):
                body_pose = body_pose.to(self.device)
            if isinstance(betas, torch.Tensor):
                betas = betas.to(self.device)
            if isinstance(global_orient, torch.Tensor):
                global_orient = global_orient.to(self.device)
            if isinstance(transl, torch.Tensor):
                transl = transl.to(self.device)

            zero_pose = torch.zeros((1, 3), dtype=torch.float32, device=self.device)

            if left_hand_pose is None:
                left_hand_pose = torch.zeros((1, 15, 3), dtype=torch.float32, device=self.device)
            if right_hand_pose is None:
                right_hand_pose = torch.zeros((1, 15, 3), dtype=torch.float32, device=self.device)

            output = self.smplx_model(
                betas=betas,
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                jaw_pose=zero_pose,
                leye_pose=zero_pose,
                reye_pose=zero_pose,
                expression=torch.zeros((1, 10), dtype=torch.float32, device=self.device),
                transl=transl,
                return_verts=True
            )

            human_verts = output.vertices[0].detach().cpu().numpy()

            if len(self.obj_orgs_world) > 0 and frame_idx < len(self.obj_orgs_world):
                world_obj = self.obj_orgs_world[frame_idx]
                obj_verts = np.asarray(world_obj.vertices)
            else:
                self.update_world_meshes()
                if len(self.obj_orgs_world) > 0 and frame_idx < len(self.obj_orgs_world):
                    world_obj = self.obj_orgs_world[frame_idx]
                    obj_verts = np.asarray(world_obj.vertices)
                else:
                    return None, "Object mesh not available for this frame"

            h_verts = human_verts.tolist()
            h_faces = self.smplx_model.faces.tolist()
            o_verts = obj_verts.tolist()
            o_faces = np.asarray(world_obj.triangles).tolist()

            camera = None

            return {
                'human': {'vertices': h_verts, 'faces': h_faces},
                'object': {'vertices': o_verts, 'faces': o_faces},
                'camera': camera
            }, None
        except Exception as e:
            print(f"Error generating focus view: {e}")
            traceback.print_exc()
            return None, str(e)

    def get_frame_meshes(self, frame_idx):
        if not self.loaded:
            return None, None

        if frame_idx < 0 or (self.total_frames > 0 and frame_idx >= self.total_frames):
            print(f"Frame index {frame_idx} out of range [0, {self.total_frames})")
            return None, None

        try:
            params = self.motion_data['smpl_params_incam']

            body_pose = params['body_pose'][frame_idx:frame_idx+1]
            betas = params['betas'][frame_idx:frame_idx+1]
            global_orient = params['global_orient'][frame_idx:frame_idx+1]
            transl = params['transl'][frame_idx:frame_idx+1]

            if isinstance(body_pose, torch.Tensor):
                if body_pose.dim() == 3 and body_pose.shape[-1] == 3:
                    body_pose = body_pose.reshape(1, -1)
                elif body_pose.dim() == 2 and body_pose.shape[0] == 1:
                    pass
                elif body_pose.dim() == 1:
                    body_pose = body_pose.unsqueeze(0)

            if isinstance(betas, torch.Tensor):
                if betas.dim() == 1:
                    betas = betas.unsqueeze(0)

            if isinstance(global_orient, torch.Tensor):
                if global_orient.dim() == 1:
                    global_orient = global_orient.unsqueeze(0)
                elif global_orient.dim() == 3:
                    global_orient = global_orient.squeeze(1)

            if isinstance(transl, torch.Tensor):
                if transl.dim() == 1:
                    transl = transl.unsqueeze(0)

            left_hand_pose = None
            right_hand_pose = None

            if self.hand_poses and str(frame_idx) in self.hand_poses:
                hand_data = self.hand_poses[str(frame_idx)]
                if 'left_hand' in hand_data and hand_data['left_hand'] is not None:
                    left_hand_array = np.array(hand_data['left_hand'])
                    if left_hand_array.size > 0:
                        left_hand_pose = torch.from_numpy(
                            left_hand_array.reshape(-1, 3)[None, ...]
                        ).float().to(self.device)
                if 'right_hand' in hand_data and hand_data['right_hand'] is not None:
                    right_hand_array = np.array(hand_data['right_hand'])
                    if right_hand_array.size > 0:
                        right_hand_pose = torch.from_numpy(
                            right_hand_array.reshape(-1, 3)[None, ...]
                        ).float().to(self.device)

            if isinstance(body_pose, torch.Tensor):
                body_pose = body_pose.to(self.device)
            if isinstance(betas, torch.Tensor):
                betas = betas.to(self.device)
            if isinstance(global_orient, torch.Tensor):
                global_orient = global_orient.to(self.device)
            if isinstance(transl, torch.Tensor):
                transl = transl.to(self.device)

            zero_pose = torch.zeros((1, 3), dtype=torch.float32, device=self.device)

            if left_hand_pose is None:
                left_hand_pose = torch.zeros((1, 15, 3), dtype=torch.float32, device=self.device)
            if right_hand_pose is None:
                right_hand_pose = torch.zeros((1, 15, 3), dtype=torch.float32, device=self.device)

            output = self.smplx_model(
                betas=betas,
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                jaw_pose=zero_pose,
                leye_pose=zero_pose,
                reye_pose=zero_pose,
                expression=torch.zeros((1, 10), dtype=torch.float32, device=self.device),
                transl=transl,
                return_verts=True
            )

            human_verts = output.vertices[0].detach().cpu().numpy()
            human_faces = self.smplx_model.faces.tolist()

            obj_verts = []
            obj_faces = []

            if len(self.obj_orgs_world) > 0 and frame_idx < len(self.obj_orgs_world):
                world_obj = self.obj_orgs_world[frame_idx]
                obj_verts = np.asarray(world_obj.vertices).tolist()
                obj_faces = np.asarray(world_obj.triangles).tolist()
            else:
                print(f"Object mesh not available for frame {frame_idx}")
                return None, None

            return (human_verts.tolist(), human_faces), (obj_verts, obj_faces)

        except Exception as e:
            print(f"Error generating frame meshes for frame {frame_idx}: {e}")
            traceback.print_exc()
            return None, None


# ========== Mesh Utility ==========

def load_mesh(obj_mesh):
    """Convert Open3D mesh to Plotly-compatible dict."""
    vertices = np.asarray(obj_mesh.vertices).tolist()
    faces = np.asarray(obj_mesh.triangles).tolist()

    # Extract vertex colors (if present)
    vertex_colors = None
    if obj_mesh.has_vertex_colors():
        colors = np.asarray(obj_mesh.vertex_colors)
        vertex_colors = ['rgb({},{},{})'.format(
            int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)
        ) for c in colors]
        print(f"Mesh loaded: {len(vertices)} vertices, {len(faces)} faces, with vertex colors")
    else:
        print(f"Mesh loaded: {len(vertices)} vertices, {len(faces)} faces, no vertex colors")

    return {'vertices': vertices, 'faces': faces, 'vertex_colors': vertex_colors}


# ========== Basic Routes ==========

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/asset/<path:filename>')
def serve_asset(filename):
    return send_from_directory('asset', filename)

@app.route('/api/metadata')
def get_metadata():
    global CAP, VIDEO_TOTAL_FRAMES, VIDEO_FPS
    with CAP_LOCK:
        if CAP is None:
            return jsonify({'error': 'Video not loaded'}), 500

        total_frames = VIDEO_TOTAL_FRAMES or int(CAP.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = VIDEO_FPS or CAP.get(cv2.CAP_PROP_FPS)

    return jsonify({
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'fps': fps,
        'has_mesh': MESH_DATA is not None,
        'video_name': os.path.basename(VIDEO_PATH),
        'obj_name': os.path.basename(OBJ_PATH)
    })


@app.get("/api/annotator")
def api_annotator():
    """Get the configured annotator name."""
    name = CONFIG.get('annotator', {}).get('name', 'anonymous')
    return jsonify({"ok": True, "annotator": name or "anonymous"})


# ========== Auth Stubs (no-op, frontend expects these) ==========

@app.get("/api/me")
def api_me():
    """Stub: always report logged in as the configured annotator."""
    name = CONFIG.get('annotator', {}).get('name', 'anonymous')
    return jsonify({"ok": True, "logged_in": True, "user": {"username": name, "display_name": name}})


@app.post("/api/login")
def api_login():
    """Stub: accept any login."""
    data = request.get_json(silent=True) or {}
    username = data.get("username", "anonymous")
    return jsonify({"ok": True, "token": "stub", "username": username})


@app.post("/api/logout")
def api_logout():
    """Stub: no-op logout."""
    return jsonify({"ok": True})


@app.route('/api/hoi_tasks')
def get_hoi_tasks():
    """Stub: return sessions as tasks for frontend compatibility."""
    if DATA_DIR is None:
        return jsonify({'tasks': [], 'total': 0})
    sessions = _scan_sessions(DATA_DIR)
    tasks = []
    for i, s in enumerate(sessions):
        tasks.append({
            'id': i,
            'object_category': s.get('category', ''),
            'file_path': s.get('path', ''),
            'session_folder': s.get('path', ''),
            'annotation_progress': 2.0,
            '_locked': False,
            '_locked_by': None,
            '_locked_by_me': False,
        })
    return jsonify({'tasks': tasks, 'total': len(tasks)})


@app.route('/api/hoi_start', methods=['POST'])
def hoi_start():
    """Stub: load session from path provided by frontend."""
    payload = request.get_json(silent=True) or {}
    session_folder = payload.get('session_folder', '')
    if not session_folder or not os.path.isdir(session_folder):
        return jsonify({'error': 'Invalid session_folder'}), 400
    ok = _load_video_session(session_folder)
    if not ok:
        return jsonify({'error': f'Failed to load session at {session_folder}'}), 500
    return jsonify({
        'status': 'success',
        'video_dir': session_folder,
        'record': {
            'session_folder': session_folder,
            '_locked': True,
            '_locked_by_me': True,
        },
    })


@app.route('/api/hoi_finish', methods=['POST'])
def hoi_finish():
    """Stub: no-op finish."""
    return jsonify({'ok': True, 'message': 'done'})


# ========== Session Management Routes ==========

@app.route('/api/sessions')
def list_sessions():
    """List all available annotation sessions from the data directory."""
    if DATA_DIR is None:
        return jsonify({'sessions': [], 'error': 'No data directory configured'}), 400
    sessions = _scan_sessions(DATA_DIR)
    return jsonify({'sessions': sessions, 'total': len(sessions)})


@app.route('/api/load_session', methods=['POST'])
def load_session():
    """Load a specific session for annotation."""
    data = request.get_json(silent=True) or {}
    session_path = data.get('path')
    if not session_path or not os.path.isdir(session_path):
        return jsonify({'error': 'Invalid session path'}), 400
    ok = _load_video_session(session_path)
    if not ok:
        return jsonify({'error': f'Failed to load session at {session_path}'}), 500
    return jsonify({'status': 'success', 'path': session_path})


@app.route('/api/reannotate', methods=['POST'])
def reannotate():
    """Clear annotations and reload the current session."""
    global SCENE_DATA

    if SCENE_DATA is None:
        return jsonify({'error': 'No session loaded'}), 400

    video_dir = SCENE_DATA.video_dir
    deleted_files = []

    kp_dir = os.path.join(video_dir, 'kp_record')
    if os.path.exists(kp_dir):
        import shutil
        try:
            shutil.rmtree(kp_dir)
            deleted_files.append(kp_dir)
        except Exception as e:
            print(f"Failed to delete kp_record dir: {e}")

    kp_merged = os.path.join(video_dir, 'kp_record_merged.json')
    if os.path.exists(kp_merged):
        try:
            os.remove(kp_merged)
            deleted_files.append(kp_merged)
        except Exception as e:
            print(f"Failed to delete kp_record_merged.json: {e}")

    ok = _load_video_session(video_dir)
    if not ok:
        return jsonify({'error': f'Failed to reload session at {video_dir}'}), 500

    return jsonify({'status': 'success', 'deleted_files': deleted_files})


# ========== Frame Delivery ==========

@app.route('/api/frame/<int:frame_idx>')
def get_frame(frame_idx):
    global VIDEO_FRAMES_ENCODED, VIDEO_TOTAL_FRAMES

    frame_idx = max(0, min(frame_idx, VIDEO_TOTAL_FRAMES - 1))

    # Use pre-encoded frames for fast response
    if VIDEO_FRAMES_ENCODED and len(VIDEO_FRAMES_ENCODED) > frame_idx:
        frame_data = VIDEO_FRAMES_ENCODED[frame_idx]
        io_buf = BytesIO(frame_data)
        response = send_file(io_buf, mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'public, max-age=31536000'
        response.headers['ETag'] = f'frame-{frame_idx}'
        return response

    # Fallback to real-time encoding
    global CAP
    with CAP_LOCK:
        if CAP is None:
            return "Video not loaded", 404
        CAP.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = CAP.read()

    if not ret:
        return "Frame read error", 500

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
    ret, buffer = cv2.imencode('.jpg', frame, encode_params)
    io_buf = BytesIO(buffer)

    response = send_file(io_buf, mimetype='image/jpeg')
    response.headers['Cache-Control'] = 'public, max-age=3600'
    response.headers['ETag'] = str(frame_idx)
    return response


# ========== Mesh Data Routes ==========

@app.route('/api/mesh')
def get_mesh():
    if MESH_DATA is None:
        return jsonify({'error': 'Mesh not loaded'}), 404

    vertices = MESH_DATA['vertices']
    faces = MESH_DATA['faces']
    vertex_colors = MESH_DATA.get('vertex_colors')

    x, y, z = zip(*vertices)
    i, j, k = zip(*faces) if faces else ([], [], [])

    result = {
        'x': x, 'y': y, 'z': z,
        'i': i, 'j': j, 'k': k
    }

    if vertex_colors:
        result['vertex_colors'] = vertex_colors

    return jsonify(result)

@app.route('/api/scene_data/<int:frame_idx>')
def get_scene_data(frame_idx):
    if SCENE_DATA is None or not SCENE_DATA.loaded:
        return jsonify({'error': 'Scene data not loaded'}), 404

    human, obj = SCENE_DATA.get_frame_meshes(frame_idx)

    if human is None or obj is None:
        error_msg = 'Failed to generate meshes'
        if human is None:
            error_msg += ' (human mesh)'
        if obj is None:
            error_msg += ' (object mesh)'
        return jsonify({'error': error_msg}), 500

    h_verts, h_faces = human
    o_verts, o_faces = obj

    hx, hy, hz = zip(*h_verts) if h_verts else ([], [], [])
    hi, hj, hk = zip(*h_faces) if h_faces else ([], [], [])

    ox, oy, oz = zip(*o_verts) if o_verts else ([], [], [])
    oi, oj, ok = zip(*o_faces) if o_faces else ([], [], [])

    # Get object vertex colors (if present)
    vertex_colors = MESH_DATA.get('vertex_colors') if MESH_DATA else None

    result = {
        'human': {
            'x': list(hx), 'y': list(hy), 'z': list(hz),
            'i': list(hi), 'j': list(hj), 'k': list(hk)
        },
        'object': {
            'x': list(ox), 'y': list(oy), 'z': list(oz),
            'i': list(oi), 'j': list(oj), 'k': list(ok)
        }
    }

    if vertex_colors:
        result['object']['vertex_colors'] = vertex_colors

    return jsonify(result)

@app.route('/api/focus_hand/<int:frame_idx>')
def focus_hand(frame_idx):
    if SCENE_DATA is None or not SCENE_DATA.loaded:
        return jsonify({'error': 'Scene data not loaded'}), 404

    data, error = SCENE_DATA.get_hand_focus_view(frame_idx)
    if data is None:
        return jsonify({'error': f'Failed to generate focus view: {error}'}), 500

    h_verts = data['human']['vertices']
    h_faces = data['human']['faces']
    if h_verts and len(h_verts) > 0:
        hx, hy, hz = zip(*h_verts)
    else:
        hx, hy, hz = [], [], []

    if h_faces and len(h_faces) > 0:
        hi, hj, hk = zip(*h_faces)
    else:
        hi, hj, hk = [], [], []

    o_verts = data['object']['vertices']
    o_faces = data['object']['faces']
    if o_verts and len(o_verts) > 0:
        ox, oy, oz = zip(*o_verts)
    else:
        ox, oy, oz = [], [], []

    if o_faces and len(o_faces) > 0:
        oi, oj, ok = zip(*o_faces)
    else:
        oi, oj, ok = [], [], []

    return jsonify({
        'human': {
            'x': list(hx), 'y': list(hy), 'z': list(hz),
            'i': list(hi), 'j': list(hj), 'k': list(hk)
        },
        'object': {
            'x': list(ox), 'y': list(oy), 'z': list(oz),
            'i': list(oi), 'j': list(oj), 'k': list(ok)
        },
        'camera': data['camera']
    })


@app.route('/api/set_scale', methods=['POST'])
def set_scale():
    """Set a new global object scale for scene visualization."""
    global SCENE_DATA

    if SCENE_DATA is None or not SCENE_DATA.loaded:
        return jsonify({'error': 'Scene data not loaded'}), 404

    data = request.get_json(silent=True) or {}
    scale_factor = data.get('scale_factor')

    try:
        scale_factor = float(scale_factor)
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid scale_factor'}), 400

    if scale_factor <= 0:
        return jsonify({'error': 'scale_factor must be > 0'}), 400

    try:
        SCENE_DATA.apply_object_scale(scale_factor)
    except Exception as e:
        print(f"Error applying object scale: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    return jsonify({'status': 'success', 'scale_factor': scale_factor})


# ========== CoTracker 2D Tracking ==========

@app.route('/api/track_2d', methods=['POST'])
def track_2d():
    global COTRACKER_MODEL, VIDEO_FRAMES, COTRACKER_LOCK

    if not COTRACKER_AVAILABLE or COTRACKER_MODEL is None:
        return jsonify({'error': 'CoTracker not available'}), 500

    data = request.json
    try:
        start_frame = int(data.get('frame_idx', 0))
    except (TypeError, ValueError):
        start_frame = 0
    x = data.get('x')
    y = data.get('y')

    if x is None or y is None:
        return jsonify({'error': 'Missing coordinates'}), 400

    if not VIDEO_FRAMES:
        return jsonify({'error': 'Video frames not loaded in memory'}), 500

    with COTRACKER_LOCK:
        try:
            device = next(COTRACKER_MODEL.parameters()).device

            queries = [[0.0, x, y]]
            queries_tensor = torch.tensor(queries, dtype=torch.float32).to(device)

            video_sequence = VIDEO_FRAMES[start_frame:]
            if not video_sequence:
                return jsonify({'error': 'No frames to track'}), 400

            window_frames = [video_sequence[0]]

            def _process_step(window_frames, is_first_step, queries=None):
                step = COTRACKER_MODEL.step
                frames_to_use = window_frames[-step * 2:] if len(window_frames) >= step * 2 else window_frames
                if len(frames_to_use) == 0:
                    return None, None

                video_chunk = (
                    torch.tensor(
                        np.stack(frames_to_use), device=device
                    )
                    .float()
                    .permute(0, 3, 1, 2)[None]
                )

                return COTRACKER_MODEL(
                    video_chunk,
                    is_first_step=is_first_step,
                    queries=queries,
                    grid_size=0,
                    grid_query_frame=0,
                )

            is_first_step = True
            pred_tracks_list = []
            pred_visibility_list = []

            step = COTRACKER_MODEL.step

            for i in range(1, len(video_sequence)):
                window_frames.append(video_sequence[i])

                if (i % step == 0) or (i == len(video_sequence) - 1):
                    pred_tracks, pred_visibility = _process_step(
                        window_frames,
                        is_first_step,
                        queries=queries_tensor[None] if is_first_step else None,
                    )

                    if pred_tracks is not None:
                        pred_tracks_list.append(pred_tracks)
                        pred_visibility_list.append(pred_visibility)

                    is_first_step = False

            if not pred_tracks_list:
                return jsonify({'error': 'Tracking failed to produce results'}), 500

            final_tracks = pred_tracks_list[-1][0].permute(1, 0, 2).cpu().numpy()
            tracks = final_tracks[0]

            result = {}
            for i in range(len(tracks)):
                abs_frame = start_frame + i
                if abs_frame < start_frame:
                    continue
                result[abs_frame] = tracks[i].tolist()

            return jsonify({'status': 'success', 'tracks': result})

        except Exception as e:
            print(f"Tracking error: {e}")
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500


@app.route('/api/track_2d_multi', methods=['POST'])
def track_2d_multi():
    """Track multiple 2D points in a single CoTracker run."""
    global COTRACKER_MODEL, VIDEO_FRAMES, COTRACKER_LOCK

    if not COTRACKER_AVAILABLE or COTRACKER_MODEL is None:
        return jsonify({'error': 'CoTracker not available'}), 500

    data = request.json or {}

    try:
        start_frame = int(data.get('frame_idx', 0))
    except (TypeError, ValueError):
        start_frame = 0

    points = data.get('points') or []
    if not isinstance(points, list) or len(points) == 0:
        return jsonify({'error': 'No points provided for tracking'}), 400

    if not VIDEO_FRAMES:
        return jsonify({'error': 'Video frames not loaded in memory'}), 500

    queries = []
    obj_indices = []
    for p in points:
        try:
            x = float(p.get('x'))
            y = float(p.get('y'))
        except (TypeError, ValueError):
            continue
        obj_idx = p.get('obj_idx')
        obj_indices.append(obj_idx)
        queries.append([0.0, x, y])

    if len(queries) == 0:
        return jsonify({'error': 'No valid points to track'}), 400

    with COTRACKER_LOCK:
        try:
            device = next(COTRACKER_MODEL.parameters()).device

            queries_tensor = torch.tensor(queries, dtype=torch.float32).to(device)

            video_sequence = VIDEO_FRAMES[start_frame:]
            if not video_sequence:
                return jsonify({'error': 'No frames to track'}), 400

            window_frames = [video_sequence[0]]

            def _process_step(window_frames, is_first_step, queries=None):
                step = COTRACKER_MODEL.step
                frames_to_use = window_frames[-step * 2:] if len(window_frames) >= step * 2 else window_frames
                if len(frames_to_use) == 0:
                    return None, None

                video_chunk = (
                    torch.tensor(
                        np.stack(frames_to_use), device=device
                    )
                    .float()
                    .permute(0, 3, 1, 2)[None]
                )

                return COTRACKER_MODEL(
                    video_chunk,
                    is_first_step=is_first_step,
                    queries=queries,
                    grid_size=0,
                    grid_query_frame=0,
                )

            is_first_step = True
            pred_tracks_list = []
            pred_visibility_list = []

            step = COTRACKER_MODEL.step

            for i in range(1, len(video_sequence)):
                window_frames.append(video_sequence[i])

                if (i % step == 0) or (i == len(video_sequence) - 1):
                    pred_tracks, pred_visibility = _process_step(
                        window_frames,
                        is_first_step,
                        queries=queries_tensor[None] if is_first_step else None,
                    )

                    if pred_tracks is not None:
                        pred_tracks_list.append(pred_tracks)
                        pred_visibility_list.append(pred_visibility)

                    is_first_step = False

            if not pred_tracks_list:
                return jsonify({'error': 'Tracking failed to produce results'}), 500

            final_tracks = pred_tracks_list[-1][0].permute(1, 0, 2).cpu().numpy()

            result = {}
            num_points, num_frames, _ = final_tracks.shape
            for i in range(num_points):
                obj_idx = obj_indices[i]
                key = str(obj_idx)
                tracks_i = final_tracks[i]
                frame_dict = {}
                for t in range(num_frames):
                    abs_frame = start_frame + t
                    frame_dict[abs_frame] = tracks_i[t].tolist()
                result[key] = frame_dict

            return jsonify({'status': 'success', 'tracks': result})

        except Exception as e:
            print(f"Multi-point tracking error: {e}")
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500


# ========== Annotation Saving ==========

@app.route('/api/save_annotation', methods=['POST'])
def save_annotation():
    """Save annotation for a single frame."""
    global SCENE_DATA, VIDEO_PATH

    data = request.get_json(silent=True) or {}

    try:
        frame_idx = int(data.get('frame', 0))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid frame index'}), 400

    human_keypoints = data.get('human_keypoints', {}) or {}
    tracks = data.get('tracks', {}) or {}

    # Build kp_record structure
    kp_record = {}

    # 3D keypoints: joint_name -> object point index
    for joint_name, info in human_keypoints.items():
        if not isinstance(info, dict):
            continue
        idx = info.get('index')
        if idx is None:
            continue
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            continue
        kp_record[joint_name] = idx

    # 2D keypoints for this frame
    two_d_list = []
    for obj_idx_str, track in tracks.items():
        try:
            obj_idx = int(obj_idx_str)
        except (TypeError, ValueError):
            obj_idx = obj_idx_str

        if not isinstance(track, dict):
            continue

        pt = track.get(str(frame_idx)) or track.get(frame_idx)
        if not pt or len(pt) < 2:
            continue

        try:
            x = float(pt[0])
            y = float(pt[1])
        except (TypeError, ValueError):
            continue

        two_d_list.append([obj_idx, [x, y]])

    kp_record['2D_keypoint'] = two_d_list

    video_dir = None
    if SCENE_DATA is not None and getattr(SCENE_DATA, 'video_dir', None):
        video_dir = SCENE_DATA.video_dir
    elif VIDEO_PATH:
        video_dir = os.path.dirname(VIDEO_PATH)

    if not video_dir:
        return jsonify({'error': 'Video directory not available; cannot save annotations'}), 500

    kp_dir = os.path.join(video_dir, 'kp_record')
    os.makedirs(kp_dir, exist_ok=True)

    out_path = os.path.join(kp_dir, f"{frame_idx:05d}.json")
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(kp_record, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving annotation for frame {frame_idx} to {out_path}: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    return jsonify({'status': 'success', 'frame': frame_idx})


# ========== Annotation Merging ==========

@app.route('/api/save_merged_annotations', methods=['POST'])
def save_merged_annotations():
    """Merge per-frame kp_record JSONs into a single kp_record_merged.json."""
    global SCENE_DATA, VIDEO_PATH

    payload = request.get_json(silent=True) or {}

    video_dir = None
    if SCENE_DATA is not None and getattr(SCENE_DATA, 'video_dir', None):
        video_dir = SCENE_DATA.video_dir
    elif VIDEO_PATH:
        video_dir = os.path.dirname(VIDEO_PATH)

    if not video_dir:
        return jsonify({'error': 'Video directory not available; cannot merge annotations'}), 500

    joint_keyframes = payload.get('joint_keyframes') or {}
    visibility_keyframes = payload.get('visibility_keyframes') or {}
    tracks = payload.get('tracks') or {}
    total_frames = payload.get('total_frames')

    last_frame_param = payload.get('last_frame')
    try:
        last_frame = int(last_frame_param) if last_frame_param is not None else None
    except (TypeError, ValueError):
        last_frame = None

    def is_visible_at_frame(obj_idx_str: str, frame_idx: int) -> bool:
        kfs = visibility_keyframes.get(str(obj_idx_str)) or []
        if not kfs:
            return True
        result = True
        for kf in kfs:
            try:
                f = int(kf.get('frame', 0))
            except Exception:
                continue
            if f <= frame_idx:
                result = bool(kf.get('visible', True))
            else:
                break
        return result

    def joint_for_obj_at_frame(obj_idx_str: str, frame_idx: int):
        kfs = joint_keyframes.get(str(obj_idx_str)) or []
        if not kfs:
            return None
        result = None
        for kf in kfs:
            try:
                f = int(kf.get('frame', 0))
            except Exception:
                continue
            if f <= frame_idx:
                result = kf.get('joint')
            else:
                break
        return result

    merged = {}
    invalid_frames = []
    first_annotated_frame = None

    # Load valid human body part keys for DoF validation
    valid_human_part_keys = None
    valid_human_part_keys_lower = {}
    try:
        part_kp_path = os.path.join(app.root_path, 'solver', 'data', 'part_kp.json')
        if os.path.exists(part_kp_path):
            with open(part_kp_path, 'r', encoding='utf-8') as f:
                _hp = json.load(f) or {}
            if isinstance(_hp, dict):
                valid_human_part_keys = set(_hp.keys())
                # Case-insensitive name mapping
                valid_human_part_keys_lower = {k.lower(): k for k in _hp.keys()}
    except Exception:
        valid_human_part_keys = None
        valid_human_part_keys_lower = {}

    try:
        if joint_keyframes or tracks:
            # Build from in-memory annotations provided by the frontend
            if total_frames is None:
                candidate_frames = []
                for obj_idx_str, tr in tracks.items():
                    for f_str in tr.keys():
                        try:
                            candidate_frames.append(int(f_str))
                        except Exception:
                            continue
                if candidate_frames:
                    total_frames = max(candidate_frames) + 1
            if total_frames is None:
                return jsonify({'error': 'total_frames is required when using in-memory annotations'}), 400

            total_frames = int(total_frames)

            if last_frame is not None:
                max_frame = min(int(last_frame), total_frames - 1)
            else:
                max_frame = total_frames - 1

            all_obj_indices = set()
            for k in joint_keyframes.keys():
                all_obj_indices.add(str(k))
            for k in tracks.keys():
                all_obj_indices.add(str(k))

            for frame_idx in range(0, max_frame + 1):
                frame_kp = {}

                joint_map = {}
                for obj_idx_str in all_obj_indices:
                    if not is_visible_at_frame(obj_idx_str, frame_idx):
                        continue
                    joint_name = joint_for_obj_at_frame(obj_idx_str, frame_idx)
                    if joint_name:
                        try:
                            obj_idx_int = int(obj_idx_str)
                        except Exception:
                            obj_idx_int = obj_idx_str
                        # Normalize joint name case (match part_kp.json)
                        normalized_name = valid_human_part_keys_lower.get(
                            joint_name.lower(), joint_name
                        ) if valid_human_part_keys_lower else joint_name
                        joint_map[str(normalized_name)] = obj_idx_int

                for joint_name, obj_idx_val in joint_map.items():
                    frame_kp[joint_name] = obj_idx_val

                two_d_list = []
                for obj_idx_str, tr in tracks.items():
                    pt = tr.get(str(frame_idx)) or tr.get(frame_idx)
                    if not pt or len(pt) < 2:
                        continue
                    try:
                        x = float(pt[0])
                        y = float(pt[1])
                    except (TypeError, ValueError):
                        continue
                    try:
                        obj_idx_int = int(obj_idx_str)
                    except Exception:
                        obj_idx_int = obj_idx_str
                    two_d_list.append([obj_idx_int, [x, y]])

                frame_kp['2D_keypoint'] = two_d_list

                num_2d = len(two_d_list)
                if valid_human_part_keys is None:
                    num_3d = len([k for k in frame_kp.keys() if k != '2D_keypoint'])
                else:
                    num_3d = len([k for k in frame_kp.keys() if (
                        k != '2D_keypoint' and
                        (k in valid_human_part_keys or k.lower() in valid_human_part_keys_lower)
                    )])
                has_annotation = (num_2d > 0) or (num_3d > 0)

                if has_annotation and first_annotated_frame is None:
                    first_annotated_frame = frame_idx

                dof = 3 * num_3d + 2 * num_2d
                if first_annotated_frame is not None and frame_idx >= first_annotated_frame and dof < 6:
                    invalid_frames.append((frame_idx, dof, num_3d, num_2d))

                merged[f"{frame_idx:05d}"] = frame_kp

        else:
            # Fallback: merge existing per-frame kp_record JSONs on disk
            kp_dir = os.path.join(video_dir, 'kp_record')
            if not os.path.isdir(kp_dir):
                return jsonify({'error': 'kp_record folder not found; nothing to merge'}), 400

            frame_indices = []
            for fname in os.listdir(kp_dir):
                if not fname.endswith('.json'):
                    continue
                stem = os.path.splitext(fname)[0]
                if len(stem) != 5:
                    continue
                try:
                    idx = int(stem)
                except ValueError:
                    continue
                frame_indices.append(idx)

            if not frame_indices:
                return jsonify({'error': 'No per-frame kp_record JSON files found to merge'}), 400

            max_frame = max(frame_indices)

            if last_frame is not None:
                max_frame = min(max_frame, int(last_frame))

            kp_dir = os.path.join(video_dir, 'kp_record')

            for frame_idx in range(0, max_frame + 1):
                fname = f"{frame_idx:05d}.json"
                fpath = os.path.join(kp_dir, fname)
                if not os.path.exists(fpath):
                    if first_annotated_frame is not None:
                        invalid_frames.append((frame_idx, 0, 0, 0))
                    continue

                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                num_2d = len(data.get('2D_keypoint', []) or [])
                num_3d = len([k for k in data.keys() if k != '2D_keypoint'])
                has_annotation = (num_2d > 0) or (num_3d > 0)

                if has_annotation and first_annotated_frame is None:
                    first_annotated_frame = frame_idx

                dof = 3 * num_3d + 2 * num_2d
                if first_annotated_frame is not None and frame_idx >= first_annotated_frame and dof < 6:
                    invalid_frames.append((frame_idx, dof, num_3d, num_2d))

                merged[f"{frame_idx:05d}"] = data

        if first_annotated_frame is None:
            return jsonify({'error': 'No 2D or 3D annotations found in any frame'}), 400

        if invalid_frames:
            msg_lines = ["Frames with insufficient DoF (need >= 6):"]
            for frame_idx, dof, n3, n2 in invalid_frames[:10]:
                msg_lines.append(f"Frame {frame_idx}: DoF={dof} (3D={n3}x3, 2D={n2}x2)")
            if len(invalid_frames) > 10:
                msg_lines.append(f"... and {len(invalid_frames) - 10} more frames")
            return jsonify({'error': '\n'.join(msg_lines)}), 400

        # Attach global metadata
        if SCENE_DATA is not None:
            try:
                object_scale = float(getattr(SCENE_DATA, 'object_scale_factor', 1.0))
            except Exception:
                object_scale = 1.0
        else:
            object_scale = 1.0

        is_static_object = bool(payload.get('is_static_object', False))

        merged['object_scale'] = object_scale
        merged['is_static_object'] = is_static_object
        merged['start_frame_index'] = first_annotated_frame

        out_path = os.path.join(video_dir, 'kp_record_merged.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

        return jsonify({
            'status': 'success',
            'path': out_path,
            'first_annotated_frame': first_annotated_frame,
            'last_frame_index': max_frame
        })

    except Exception as e:
        print(f"Error merging kp_record files: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Failed to merge kp_record files: {e}'}), 500


# ========== Optimization ==========

@app.route('/api/run_optimization', methods=['POST'])
def run_optimization():
    global SCENE_DATA, CAP

    if SCENE_DATA is None or not SCENE_DATA.loaded:
        return jsonify({'status': 'error', 'message': 'Scene data not loaded'})

    data = request.get_json(silent=True) or {}
    try:
        current_frame = int(data.get('frame_idx', 0))
    except Exception:
        current_frame = 0
    try:
        last_frame_req = data.get('last_frame', current_frame)
        last_frame = int(last_frame_req)
    except Exception:
        last_frame = current_frame

    video_dir = SCENE_DATA.video_dir
    kp_record_path = os.path.join(video_dir, 'kp_record_merged.json')

    if not os.path.exists(kp_record_path):
        return jsonify({'status': 'error', 'message': 'No merged annotations found. Please save annotations first.'})

    with open(kp_record_path, 'r') as f:
        merged = json.load(f)

    try:
        start_frame = merged.get("start_frame_index", 0)
    except Exception:
        start_frame = 0

    is_static_object = merged.get("is_static_object", False)

    end_frame = min(SCENE_DATA.total_frames, max(last_frame, current_frame) + 1)
    if end_frame <= start_frame:
        end_frame = min(SCENE_DATA.total_frames, start_frame + 1)

    part_kp_path = os.path.join(app.root_path, 'solver', 'data', 'part_kp.json')
    if not os.path.exists(part_kp_path):
        return jsonify({'status': 'error', 'message': 'part_kp.json not found'})

    with open(part_kp_path, 'r') as f:
        human_part = json.load(f)

    # Early validation: ensure sufficient DoF for solver
    try:
        valid_keys = set(human_part.keys()) if isinstance(human_part, dict) else set()
        with open(kp_record_path, 'r', encoding='utf-8') as f:
            merged_for_check = json.load(f) or {}

        insufficient = []
        for frame_idx in range(start_frame, end_frame):
            key = f"{frame_idx:05d}"
            ann = merged_for_check.get(key) or {}

            two_d = ann.get('2D_keypoint') or []
            n2 = len(two_d)
            n3 = 0
            ignored = []
            for k in ann.keys():
                if k == '2D_keypoint':
                    continue
                if k in valid_keys:
                    n3 += 1
                else:
                    ignored.append(k)

            dof = 2 * n2 + 3 * n3
            if dof < 6:
                insufficient.append((frame_idx, dof, n3, n2, ignored[:5]))

        if insufficient:
            lines = ["Insufficient effective DoF for optimization (need >= 6)."]
            for (fi, dof, n3, n2, ign) in insufficient[:5]:
                extra = f"; ignored_3d_keys={ign}" if ign else ""
                lines.append(f"Frame {fi}: DoF={dof} (valid3D={n3}x3, 2D={n2}x2){extra}")
            if len(insufficient) > 5:
                lines.append(f"... and {len(insufficient) - 5} more frames")
            return jsonify({'status': 'error', 'message': "\n".join(lines)})
    except Exception:
        pass

    # Prepare camera intrinsic matrix K
    width = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))
    focal_length = max(width, height)
    cx = width / 2
    cy = height / 2
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    body_params = SCENE_DATA.motion_data['smpl_params_incam']

    SCENE_DATA.update_world_meshes()
    sampled_orgs = SCENE_DATA.obj_orgs_world

    try:
        if kp_use_new is None:
            return jsonify({'status': 'error', 'message': 'kp_use_new module not loaded'})

        new_body_params, new_icp_transforms = kp_use_new(
            output=None,
            hand_poses=SCENE_DATA.hand_poses,
            body_poses=body_params,
            global_body_poses=body_params,
            sampled_orgs=sampled_orgs,
            human_part=human_part,
            K=torch.from_numpy(K),
            start_frame=start_frame,
            end_frame=end_frame,
            video_dir=video_dir,
            is_static_object=is_static_object,
            kp_record_path=kp_record_path
        )

        # Apply transforms to object meshes
        for i, transform in enumerate(new_icp_transforms):
            frame_idx = start_frame + i
            if frame_idx < len(SCENE_DATA.obj_orgs):
                mat_inc = transform.cpu().numpy() if hasattr(transform, 'cpu') else transform

                R_inc = mat_inc[:3, :3]
                t_inc = mat_inc[:3, 3]

                if SCENE_DATA.R_finals is not None and frame_idx < len(SCENE_DATA.R_finals):
                    R_old = SCENE_DATA.R_finals[frame_idx]
                    SCENE_DATA.R_finals[frame_idx] = R_inc @ R_old

                if SCENE_DATA.t_finals is not None and frame_idx < len(SCENE_DATA.t_finals):
                    t_old = SCENE_DATA.t_finals[frame_idx]
                    SCENE_DATA.t_finals[frame_idx] = R_inc @ t_old + t_inc

        SCENE_DATA.update_world_meshes()

        # Delete kp_record_merged.json after optimization to allow re-annotation
        if os.path.exists(kp_record_path):
            try:
                os.remove(kp_record_path)
            except Exception as e:
                print(f"Warning: Failed to delete {kp_record_path}: {e}")

        return jsonify({'status': 'success'})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})


# ========== IVD Model Auto-Prediction (Optional) ==========

IVD_PREDICTOR = None
IVD_LOCK = Lock()


def _get_ivd_predictor():
    """Get the IVD predictor instance."""
    global IVD_PREDICTOR
    if IVD_PREDICTOR is None:
        raise RuntimeError("IVD predictor not initialized. Call _init_ivd_predictor() first.")
    return IVD_PREDICTOR


def _init_ivd_predictor():
    """Initialize the IVD predictor at app startup (if enabled in config)."""
    global IVD_PREDICTOR
    if IVD_PREDICTOR is not None:
        return

    ivd_config = CONFIG.get('ivd_model', {})
    if not ivd_config.get('enabled', True):
        print("[IVD] IVD predictor disabled in config")
        return

    try:
        from ivd_predictor import IVDPredictor
        print("[IVD] Initializing IVD predictor...")

        checkpoint = ivd_config.get('checkpoint', '') or None
        device = ivd_config.get('device', 'cuda')
        use_lightweight = ivd_config.get('use_lightweight_vlm', False)

        IVD_PREDICTOR = IVDPredictor(
            checkpoint_path=checkpoint,
            device=device if torch.cuda.is_available() else 'cpu',
            use_lightweight_vlm=use_lightweight,
        )
        IVD_PREDICTOR._load_model()
        print("[IVD] IVD predictor initialized successfully")
    except Exception as e:
        print(f"[IVD] Warning: Failed to initialize IVD predictor: {e}")
        traceback.print_exc()


@app.route('/api/auto_predict', methods=['POST'])
def auto_predict():
    """Auto-predict human contact joints from current frame using IVD model."""
    global VIDEO_FRAMES, MESH_DATA, SCENE_DATA

    ivd_config = CONFIG.get('ivd_model', {})
    if not ivd_config.get('enabled', True):
        return jsonify({
            'ok': False,
            'error': 'IVD model is disabled in config. Set ivd_model.enabled=true to enable.'
        }), 503

    data = request.get_json(silent=True) or {}
    frame_idx = data.get('frame', 0)
    threshold = data.get('threshold', 0.5)
    top_k = data.get('top_k', None)

    if not VIDEO_FRAMES or frame_idx < 0 or frame_idx >= len(VIDEO_FRAMES):
        return jsonify({
            'ok': False,
            'error': f'Invalid frame index: {frame_idx}'
        }), 400

    if MESH_DATA is None:
        return jsonify({'ok': False, 'error': 'No mesh data loaded'}), 400

    if SCENE_DATA is None or not SCENE_DATA.loaded:
        return jsonify({'ok': False, 'error': 'Scene data not loaded'}), 400

    if IVD_PREDICTOR is None:
        return jsonify({
            'ok': False,
            'error': 'IVD predictor not initialized. The model may have failed to load at startup.'
        }), 503

    try:
        with IVD_LOCK:
            predictor = _get_ivd_predictor()

            rgb_frame = VIDEO_FRAMES[frame_idx]

            if frame_idx < len(SCENE_DATA.obj_orgs_world):
                obj_mesh = SCENE_DATA.obj_orgs_world[frame_idx]
                obj_vertices = np.asarray(obj_mesh.vertices)
            else:
                obj_vertices = np.array(MESH_DATA['vertices'])

            predictions = predictor.predict(
                rgb_frame=rgb_frame,
                object_vertices=obj_vertices,
                threshold=threshold,
                top_k=top_k,
            )

            mesh_center = np.mean(obj_vertices, axis=0)
            for pred in predictions:
                xyz = np.array(pred['xyz'])
                pred['xyz'] = (xyz + mesh_center).tolist()

                distances = np.linalg.norm(obj_vertices - np.array(pred['xyz']), axis=1)
                nearest_idx = int(np.argmin(distances))
                pred['vertex_idx'] = nearest_idx
                pred['vertex_distance'] = float(distances[nearest_idx])

            return jsonify({
                'ok': True,
                'predictions': predictions,
                'frame': frame_idx,
                'threshold': threshold,
            })

    except ImportError as e:
        return jsonify({'ok': False, 'error': f'IVD predictor not available: {e}'}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({'ok': False, 'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/ivd_joint_names')
def get_ivd_joint_names():
    """Get the list of joint names used by the IVD model."""
    try:
        with IVD_LOCK:
            predictor = _get_ivd_predictor()
            joint_names = predictor.get_joint_names()
            return jsonify({
                'ok': True,
                'joint_names': joint_names,
                'count': len(joint_names),
            })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/ivd_status')
def get_ivd_status():
    """Check IVD predictor status."""
    ivd_config = CONFIG.get('ivd_model', {})
    status = {
        'enabled': ivd_config.get('enabled', True),
        'predictor_initialized': IVD_PREDICTOR is not None,
        'predictor_loaded': False,
        'device': None,
        'checkpoint': None,
        'error': None,
    }

    if IVD_PREDICTOR is not None:
        try:
            status['predictor_loaded'] = IVD_PREDICTOR._loaded
            status['device'] = str(IVD_PREDICTOR.device)
            status['checkpoint'] = IVD_PREDICTOR.checkpoint_path
        except Exception as e:
            status['error'] = str(e)

    return jsonify(status)


# ========== Video Loading & CoTracker ==========

def load_video_frames():
    global VIDEO_FRAMES, VIDEO_FRAMES_ENCODED, CAP, VIDEO_FPS, VIDEO_TOTAL_FRAMES
    if CAP is None:
        return

    VIDEO_FPS = CAP.get(cv2.CAP_PROP_FPS)
    VIDEO_TOTAL_FRAMES = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Loading video frames into memory... FPS: {VIDEO_FPS}, Total frames: {VIDEO_TOTAL_FRAMES}")
    VIDEO_FRAMES = []
    VIDEO_FRAMES_ENCODED = []

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]

    CAP.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0

    while True:
        ret, frame = CAP.read()
        if not ret:
            break

        # Store raw RGB frame (for processing)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        VIDEO_FRAMES.append(frame_rgb)

        # Pre-encode as JPEG for fast response
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        VIDEO_FRAMES_ENCODED.append(buffer.tobytes())

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Loaded and encoded {frame_count}/{VIDEO_TOTAL_FRAMES} frames...")

    print(f"Successfully loaded {len(VIDEO_FRAMES)} frames")
    print(f"Total encoded size: {sum(len(f) for f in VIDEO_FRAMES_ENCODED) / 1024 / 1024:.2f} MB")


def init_cotracker():
    global COTRACKER_MODEL
    if not COTRACKER_AVAILABLE:
        return

    checkpoint_path = os.path.join(os.path.dirname(__file__), 'co-tracker/checkpoints/scaled_online.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading CoTracker from {checkpoint_path}")
        COTRACKER_MODEL = CoTrackerOnlinePredictor(checkpoint=checkpoint_path)
        if torch.cuda.is_available():
            COTRACKER_MODEL = COTRACKER_MODEL.to('cuda')
        print("CoTracker loaded")
    else:
        print(f"CoTracker checkpoint not found at {checkpoint_path}")


# ========== Main Entry Point ==========

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='4DHOI Annotator - Interactive 4D HOI Annotation Tool')
    parser.add_argument('--data_dir', type=str, default='demo',
                        help='Directory containing session folders to annotate')
    parser.add_argument('--video_dir', type=str, default=None,
                        help='Directly load a specific session directory')
    parser.add_argument('--host', type=str, default=None, help='Server host')
    parser.add_argument('--port', type=int, default=None, help='Server port')
    args = parser.parse_args()

    DATA_DIR = os.path.abspath(args.data_dir)
    print(f"Data directory: {DATA_DIR}")

    if args.video_dir:
        _load_video_session(args.video_dir)

    # Initialize IVD predictor (if enabled)
    _init_ivd_predictor()

    host = args.host or CONFIG['server']['host']
    port = args.port or CONFIG['server']['port']
    app.run(debug=True, host=host, port=port, use_reloader=False)
