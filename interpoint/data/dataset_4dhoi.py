"""
Dataset for 4D-HOI custom data aligned to IVD input format.
Contact masks are returned as None.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
import trimesh

from .transforms import IVDTransform
from utils.keypoints import KeypointManager


class Custom4DHOIAlignedDataset:
    def __init__(
        self,
        upload_records_path: str,
        data_root: str,
        split: str = "train",
        transform: Optional[IVDTransform] = None,
        num_object_points: int = 1024,
        num_human_points: int = 10475,
        test_split_ratio: float = 0.2,
        frame_interval: int = 10,
        seed: int = 42,
        skip_invalid: bool = True,
        split_file: Optional[str] = "/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/dataset_statics/split_progress4_seed42.json",
    ):
        self.upload_records_path = Path(upload_records_path)
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.num_object_points = num_object_points
        self.num_human_points = num_human_points
        self.frame_interval = frame_interval
        self.skip_invalid = skip_invalid
        self._obj_cache: Dict[str, Dict] = {}
        self.split_file = Path(split_file) if split_file else None
        self.keypoints = KeypointManager("data/part_kp.json")
        self.name_to_kp = self.keypoints.name_to_idx

        random.seed(seed)
        np.random.seed(seed)

        self.samples = self._build_frame_samples()
        if self.split_file is not None:
            self.samples = self._apply_split_file(self.samples)
        else:
            sessions_map = {}
            for sample in self.samples:
                session_folder = sample["session_folder"]
                sessions_map.setdefault(session_folder, []).append(sample)

            sessions_list = list(sessions_map.keys())
            random.shuffle(sessions_list)
            split_idx = int(len(sessions_list) * (1 - test_split_ratio))

            if split == "train":
                valid_sessions = sessions_list[:split_idx]
            else:
                valid_sessions = sessions_list[split_idx:]

            self.samples = [s for s in self.samples if s["session_folder"] in valid_sessions]

    def _build_frame_samples(self) -> List[Dict]:
        with open(self.upload_records_path, "r") as f:
            records = json.load(f)

        samples = []
        for record in records:
            if record.get("annotation_progress") != 4.0:
                continue

            session_folder = Path(record["session_folder"])
            if not session_folder.is_absolute():
                session_folder = self.data_root / session_folder

            kp_file = session_folder / "kp_record_new.json"
            obj_file = session_folder / "obj_init.obj"
            frames_dir = session_folder / "frames"
            if not (kp_file.exists() and obj_file.exists() and frames_dir.exists()):
                if self.skip_invalid:
                    continue
                else:
                    raise FileNotFoundError(session_folder)

            with open(kp_file, "r") as f:
                kp_data = json.load(f)

            frames = sorted([f.name for f in frames_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
            if len(frames) == 0:
                continue
            sampled_frames = frames[:: self.frame_interval]

            for frame_name in sampled_frames:
                frame_id = Path(frame_name).stem
                frame_data = kp_data.get(frame_id, {})
                keys = [k for k in frame_data.keys() if k != "2D_keypoint"]
                if len(keys) == 0:
                    continue
                invalid_key = any(k not in self.name_to_kp for k in keys)
                if invalid_key:
                    continue
                samples.append(
                    {
                        "session_folder": session_folder,
                        "file_name": record["file_name"],
                        "object_category": record.get("object_category", "unknown"),
                        "frame_name": frame_name,
                        "frame_id": frame_id,
                    }
                )
        return samples

    def _apply_split_file(self, samples: List[Dict]) -> List[Dict]:
        if self.split_file is None:
            return samples
        with open(self.split_file, "r") as f:
            split_data = json.load(f)
        splits = split_data.get("splits", {})
        records = split_data.get("records", {})
        if self.split not in splits:
            raise ValueError(f"Split '{self.split}' not found in {self.split_file}")

        split_ids = splits[self.split]
        valid_sessions = set()
        for rid in split_ids:
            rec = records.get(rid)
            if rec is None:
                continue
            session_folder = Path(rec["session_folder"])
            if not session_folder.is_absolute():
                session_folder = self.data_root / session_folder
            valid_sessions.add(session_folder)

        return [s for s in samples if s["session_folder"] in valid_sessions]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, img_path: Path, image_size: int = 224) -> np.ndarray:
        image = Image.open(img_path).convert("RGB")
        image = image.resize((image_size, image_size), Image.BILINEAR)
        return np.array(image)

    def _load_object_mesh(self, obj_path: Path) -> Dict:
        cache_key = str(obj_path)
        if cache_key in self._obj_cache:
            return self._obj_cache[cache_key]

        mesh = trimesh.load(obj_path, process=False)
        points = np.array(mesh.vertices, dtype=np.float32)
        default_color = np.tile(np.array([[0.82, 0.84, 0.88]], dtype=np.float32), (points.shape[0], 1))
        colors = default_color
        try:
            if hasattr(mesh, "visual") and mesh.visual is not None:
                vc = getattr(mesh.visual, "vertex_colors", None)
                if vc is not None:
                    vc = np.array(vc)
                    if vc.ndim == 2 and vc.shape[0] == points.shape[0]:
                        if vc.shape[1] >= 3:
                            colors = vc[:, :3].astype(np.float32)
                            if colors.max() > 1.0:
                                colors = colors / 255.0
        except Exception:
            colors = default_color

        centroid = points.mean(axis=0)
        points = points - centroid
        max_radius = np.max(np.linalg.norm(points, axis=1))
        origin_points = points
        if max_radius > 0:
            origin_points = points / max_radius

        if len(origin_points) > self.num_object_points:
            indices = np.random.choice(len(origin_points), self.num_object_points, replace=False)
            points = origin_points[indices]
            point_colors = colors[indices]
        elif len(origin_points) < self.num_object_points:
            pad = np.zeros((self.num_object_points - len(origin_points), 3), dtype=np.float32)
            pad_col = np.tile(np.array([[0.82, 0.84, 0.88]], dtype=np.float32), (self.num_object_points - len(origin_points), 1))
            points = np.concatenate([origin_points, pad], axis=0)
            point_colors = np.concatenate([colors, pad_col], axis=0)
        else:
            points = origin_points
            point_colors = colors

        result = {
            "points": points,
            "colors": point_colors.astype(np.float32),
            "centroid": centroid,
            "max_radius": float(max_radius),
            "original_vertices": origin_points,
            "original_colors": colors.astype(np.float32),
        }
        self._obj_cache[cache_key] = result
        return result

    def _load_keypoints_3d(self, kp_file: Path, frame_id: str) -> np.ndarray:
        with open(kp_file, "r") as f:
            kp_data = json.load(f)
        frame_data = kp_data.get(frame_id, {})
        object_indices = np.full((87,), -1, dtype=np.int64)
        for key, value in frame_data.items():
            if key == "2D_keypoint":
                continue
            idx = self.name_to_kp.get(key)
            if idx is None:
                raise KeyError(f"Unknown keypoint name '{key}' in {kp_file} frame {frame_id}")
            object_indices[idx] = int(value)
        return object_indices

    def _get_keypoint_contacts_binary(self, kp_file: Path, frame_id: str, num_keypoints: int = 87) -> np.ndarray:
        with open(kp_file, "r") as f:
            kp_data = json.load(f)
        frame_data = kp_data.get(frame_id, {})
        labels = np.zeros(num_keypoints, dtype=np.float32)
        for key in frame_data.keys():
            if key == "2D_keypoint":
                continue
            idx = self.name_to_kp.get(key)
            if idx is None or idx >= num_keypoints:
                raise KeyError(f"Unknown keypoint name '{key}' in {kp_file} frame {frame_id}")
            labels[idx] = 1.0
        return labels

    def __getitem__(self, idx: int) -> Dict:
        sample_info = self.samples[idx]
        session_folder = sample_info["session_folder"]
        frame_name = sample_info["frame_name"]
        frame_id = sample_info["frame_id"]

        img_path = session_folder / "frames" / frame_name
        rgb_image = self._load_image(img_path)

        obj_path = session_folder / "obj_init.obj"
        object_data = self._load_object_mesh(obj_path)

        kp_file = session_folder / "kp_record_new.json"
        human_labels = self._get_keypoint_contacts_binary(kp_file, frame_id)
        keypoint_3d_indices = self._load_keypoints_3d(kp_file, frame_id)

        if human_labels.sum() == 0:
            raise RuntimeError(f"No valid human labels in {kp_file} frame {frame_id}")

        object_indices = keypoint_3d_indices.copy()
        valid_indices = object_indices[object_indices >= 0]
        if len(valid_indices) == 0:
            raise RuntimeError(f"No valid 3D keypoint indices in {kp_file} frame {frame_id}")
        max_idx = object_data["original_vertices"].shape[0] - 1
        object_indices = np.where(object_indices >= 0, np.minimum(object_indices, max_idx), object_indices)

        object_coords = np.zeros((87, 3), dtype=np.float32)
        valid = (object_indices >= 0) & (object_indices < object_data["original_vertices"].shape[0])
        if not np.any(valid):
            raise RuntimeError(f"No valid object coords in {kp_file} frame {frame_id}")
        object_coords[valid] = object_data["original_vertices"][object_indices[valid]]

        sample = {
            "sample_id": f"{sample_info['file_name']}_{frame_id}",
            "rgb_image": rgb_image,
            "human_vertices": None,
            "object_points": object_data["points"],
            "object_colors": object_data["colors"],
            "human_labels": human_labels,
            "object_coords": object_coords,
            "human_contact_mask": None,
            "object_contact_mask": None,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
