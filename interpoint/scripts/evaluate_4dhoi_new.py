"""
Evaluate IVD model on 4D-HOI data.

Human: 87-way binary classification metrics.
Object: Map predicted/GT object coords to nearest points in normalized object point cloud (batch["object_points"]),
        then compute point-coverage recall + GT->pred mean min-dist.
Interaction: Compute pair_recall / pair_recall_joint based on mapped point cloud distances.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

default_collate_func = dataloader.default_collate


def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]
try:
    from scipy.spatial import cKDTree
    _HAS_CKDTREE = True
except Exception:
    _HAS_CKDTREE = False

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.dataset_4dhoi import Custom4DHOIAlignedDataset
from data.dataset import collate_fn as ivd_collate_fn
from data.transforms import get_val_transforms
from models import build_model
from utils.keypoints import KeypointManager


def compute_binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    tn = (~pred & ~gt).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
    }

def nearest_indices(points: np.ndarray, queries: np.ndarray) -> np.ndarray:
    if queries.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if _HAS_CKDTREE:
        tree = cKDTree(points)
        _, idx = tree.query(queries, k=1)
        return idx.astype(np.int64)
    pts = torch.from_numpy(points.astype(np.float32))
    q = torch.from_numpy(queries.astype(np.float32))
    dist = torch.cdist(q.unsqueeze(0), pts.unsqueeze(0)).squeeze(0)
    return torch.argmin(dist, dim=1).cpu().numpy().astype(np.int64)

def _min_distances_gt_to_pred(
    obj_verts: np.ndarray,
    gt_indices: np.ndarray,
    pred_indices: np.ndarray,
) -> np.ndarray:
    if gt_indices.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if pred_indices.size == 0:
        return np.full((len(gt_indices),), np.nan, dtype=np.float32)
    
    gt_xyz = obj_verts[gt_indices]
    pred_xyz = obj_verts[pred_indices]
    
    a = torch.from_numpy(gt_xyz).float()
    b = torch.from_numpy(pred_xyz).float()
    d = torch.cdist(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)
    return d.min(dim=1).values.cpu().numpy().astype(np.float32)


def _collect_used_keypoint_names_all(
    upload_records_path: str,
    data_root: str,
    name_to_idx: Dict[str, int],
) -> List[str]:
    with open(upload_records_path, "r") as f:
        records = json.load(f)

    used = set()
    for record in records:
        if record.get("annotation_progress") != 4.0:
            continue

        session_folder = Path(record.get("session_folder"))
        if not session_folder.is_absolute():
            session_folder = Path(data_root) / session_folder

        kp_file = session_folder / "kp_record_new.json"
        if not kp_file.exists():
            continue

        try:
            with open(kp_file, "r") as kf:
                kp_data = json.load(kf)
        except Exception:
            continue

        if not isinstance(kp_data, dict):
            continue

        for _, frame_data in kp_data.items():
            if not isinstance(frame_data, dict):
                continue
            for k in frame_data.keys():
                if k == "2D_keypoint":
                    continue
                if k in name_to_idx:
                    used.add(k)

    return sorted(list(used))


def _collect_used_keypoint_names_from_samples(
    samples: List[Dict],
    name_to_idx: Dict[str, int],
) -> List[str]:
    used = set()
    seen_sessions = set()

    for sample in samples:
        session_folder = sample.get("session_folder")
        if session_folder is None:
            continue
        sess_key = str(session_folder)
        if sess_key in seen_sessions:
            continue
        seen_sessions.add(sess_key)

        kp_file = Path(session_folder) / "kp_record_new.json"
        if not kp_file.exists():
            continue

        try:
            with open(kp_file, "r") as kf:
                kp_data = json.load(kf)
        except Exception:
            continue

        if not isinstance(kp_data, dict):
            continue

        for _, frame_data in kp_data.items():
            if not isinstance(frame_data, dict):
                continue
            for k in frame_data.keys():
                if k == "2D_keypoint":
                    continue
                if k in name_to_idx:
                    used.add(k)

    return sorted(list(used))


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    threshold_human=0.25,
    threshold_object=0.1,
    used_kp_indices: Optional[np.ndarray] = None,
):
    model.eval()

    human_preds = []
    human_gts = []

    # Object Metrics Accumulators
    obj_gt_points_total = 0
    obj_gt_points_covered = 0
    all_gt_to_pred_dists = []

    # Interaction Metrics Accumulators
    paired_total = 0
    matched_total = 0
    matched_joint_total = 0

    if used_kp_indices is not None:
        kp_mask = np.zeros((87,), dtype=bool)
        kp_mask[used_kp_indices] = True
    else:
        kp_mask = None

    for batch in tqdm(loader, desc="Evaluating"):
        rgb = batch["rgb_image"].to(device)
        obj_pts = batch["object_points"].to(device)
        human_labels = batch["human_labels"].cpu().numpy()
        object_coords = batch["object_coords"].cpu().numpy()

        outputs = model(rgb, obj_pts, return_aux=False)
        human_probs = outputs["human_contact"].detach().cpu().numpy()
        pred_object_coords = outputs["object_coords"].detach().cpu().numpy()

        pred_human_full = (human_probs >= threshold_human).astype(np.uint8)
        human_gts_full = (human_labels >= 0.5).astype(np.uint8)

        if kp_mask is not None:
            pred_human_bin = pred_human_full[:, kp_mask]
            human_gts_bin = human_gts_full[:, kp_mask]
        else:
            pred_human_bin = pred_human_full
            human_gts_bin = human_gts_full

        # Record for Human Metrics (possibly filtered)
        human_preds.append(pred_human_bin)
        human_gts.append(human_gts_bin)

        # Process per sample in batch
        for b in range(obj_pts.shape[0]):
            obj_points_np = obj_pts[b].cpu().numpy()
            gt_human_full = human_gts_full[b]
            pred_human_full_b = pred_human_full[b]

            if kp_mask is not None:
                kp_indices = np.where(kp_mask)[0]
                gt_contact_kp_indices = kp_indices[gt_human_full[kp_indices] == 1]
            else:
                gt_contact_kp_indices = np.where(gt_human_full == 1)[0]
            pred_vertex_indices = np.array([], dtype=np.int64)

            # --- Object Metrics ---
            if len(gt_contact_kp_indices) > 0:
                # Decouple object evaluation from human classification:
                # for each GT contact keypoint, evaluate prediction at the same keypoint index.
                pred_coords = pred_object_coords[b][gt_contact_kp_indices]
                gt_coords = object_coords[b][gt_contact_kp_indices]

                # Filter invalid coordinates in a pairwise way to keep index alignment.
                valid_mask = np.isfinite(pred_coords).all(axis=1) & np.isfinite(gt_coords).all(axis=1)
                pred_coords = pred_coords[valid_mask]
                gt_coords = gt_coords[valid_mask]

                # Map to point cloud surface
                if gt_coords.size > 0:
                    gt_vertex_indices = nearest_indices(obj_points_np, gt_coords)
                
                    if pred_coords.size > 0:
                        pred_vertex_indices = nearest_indices(obj_points_np, pred_coords)

                        # Compute distance on point cloud
                        min_dists_per_gt = _min_distances_gt_to_pred(
                            obj_points_np, gt_vertex_indices, pred_vertex_indices
                        )

                        covered = int((min_dists_per_gt < threshold_object).sum())
                        obj_gt_points_covered += covered
                        
                        # Remove invalid values then record distance
                        valid_dists = min_dists_per_gt[~np.isnan(min_dists_per_gt)]
                        all_gt_to_pred_dists.extend(valid_dists.tolist())

                    obj_gt_points_total += len(gt_vertex_indices)

                # --- Interaction Metrics ---
                for kpi in gt_contact_kp_indices:
                    paired_total += 1
                    obj_ok = False

                    if pred_vertex_indices.size > 0:
                        gt_coord = object_coords[b][kpi]
                        if np.isfinite(gt_coord).all():
                            # Map a single GT point
                            kpi_gt_vertex_idx = nearest_indices(obj_points_np, np.array([gt_coord]))[0]
                            
                            gt_xyz = obj_points_np[kpi_gt_vertex_idx]
                            pred_xyz = obj_points_np[pred_vertex_indices]

                            # Find nearest distance on point cloud
                            dists = np.linalg.norm(pred_xyz - gt_xyz, axis=1)
                            min_dist = dists.min()
                            
                            if min_dist < threshold_object:
                                obj_ok = True

                    if obj_ok:
                        matched_total += 1
                        if pred_human_full_b[kpi] == 1:
                            matched_joint_total += 1

    # Summarize Human Metrics
    human_preds = np.concatenate(human_preds, axis=0)
    human_gts = np.concatenate(human_gts, axis=0)
    human_metrics = compute_binary_metrics(human_preds.flatten(), human_gts.flatten())

    # Summarize Object Metrics
    obj_recall = obj_gt_points_covered / max(1, obj_gt_points_total)
    obj_mean_dist = np.mean(all_gt_to_pred_dists) if len(all_gt_to_pred_dists) > 0 else float("nan")
    
    object_metrics = {
        "recall_point_coverage": float(obj_recall),
        "mean_min_dist_gt_to_pred": float(obj_mean_dist),
        "distance_threshold": float(threshold_object)
    }

    # Summarize Interaction Metrics
    interaction_metrics = {
        "pair_recall": float(matched_total / max(1, paired_total)) if paired_total > 0 else float("nan"),
        "pair_recall_joint": float(matched_joint_total / max(1, paired_total)) if paired_total > 0 else float("nan"),
        "paired_total": int(paired_total),
    }

    return human_metrics, object_metrics, interaction_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload_records", type=str, default="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/4dhoi_autorecon/upload_records.json")
    parser.add_argument("--data_root", type=str, default='/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/4dhoi_autorecon')
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test_split_ratio", type=float, default=0.4)
    parser.add_argument("--frame_interval", type=int, default=10)
    parser.add_argument("--threshold_human", type=float, default=0.3)
    parser.add_argument("--threshold_object", type=float, default=0.05) # Object distance threshold
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--object_coord_mode",
        type=str,
        default="weighted",
        choices=["weighted", "direct"],
        help="Object coordinate head mode used by the checkpoint.",
    )
    parser.add_argument(
        "--keypoint_scan_scope",
        default="split",
        choices=["split", "all"],
        help=(
            "How to collect the set of keypoint names for human/object/pair evaluation. "
            "split: scan sessions in selected split only (recommended). "
            "all: scan all sessions in upload_records."
        ),
    )
    parser.add_argument("--smplx_kp_json", default="data/part_kp.json")
    parser.add_argument("--debug_keypoints", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = build_model({
        "d_tr": 256,
        "num_body_points": 87,
        "num_object_queries": 87,
        "object_coord_mode": args.object_coord_mode,
        "use_lightweight_vlm": False,
        "device": str(device),
    }).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    
    val_transform = get_val_transforms(image_size=224, render_size=256)
    dataset = Custom4DHOIAlignedDataset(
        upload_records_path=args.upload_records,
        data_root=args.data_root,
        split="test",
        transform=val_transform,
        test_split_ratio=args.test_split_ratio,
        frame_interval=args.frame_interval,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ivd_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    if not os.path.exists(args.smplx_kp_json):
        raise FileNotFoundError(f"smplx_kp_json not found: {args.smplx_kp_json}")
    keypoints = KeypointManager(args.smplx_kp_json)
    name_to_idx = keypoints.name_to_idx

    if args.keypoint_scan_scope == "all":
        keypoint_names = _collect_used_keypoint_names_all(args.upload_records, args.data_root, name_to_idx)
    else:
        keypoint_names = _collect_used_keypoint_names_from_samples(dataset.samples, name_to_idx)

    if len(keypoint_names) == 0:
        raise RuntimeError("No usable keypoint names found in kp_record_new.json")

    used_kp_indices = np.array(sorted([name_to_idx[k] for k in keypoint_names]), dtype=np.int64)

    human_metrics, object_metrics, interaction_metrics = evaluate(
        model, loader, device, 
        threshold_human=args.threshold_human, 
        threshold_object=args.threshold_object,
        used_kp_indices=used_kp_indices,
    )

    print("\n" + "="*40)
    print("Human Metrics (87 Keypoints Binary Classification):")
    for k, v in human_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "="*40)
    print(f"Object Metrics (Mapped to Dataloader Point Cloud, Threshold: {args.threshold_object}):")
    for k, v in object_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "="*40)
    print("Interaction Metrics:")
    for k, v in interaction_metrics.items():
        if "total" in k:
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.4f}")

    print("\n" + "="*40)
    print("Keypoint Evaluation Scope:")
    print(f"  num_keypoints_eval: {len(used_kp_indices)}")
    print(f"  keypoint_scan_scope: {args.keypoint_scan_scope}")
    if args.debug_keypoints:
        print("  used_keypoint_names:")
        for name in keypoint_names:
            print(f"    {name}")


if __name__ == "__main__":
    main()
