"""
Combined dataset loader for InterCap + IMHD.
Provides the same API as the original InterCap dataset module.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from utils.keypoints import KeypointManager
from utils.renderer import MultiViewRenderer
from .transforms import get_train_transforms, get_val_transforms, IVDTransform
from .dataset_intercap import IVDDataset as IntercapDataset
from .dataset_imhd import IMHDDataset


class MixedIVDDataset(Dataset):
    """
    Concatenate InterCap and IMHD datasets into one.
    """

    def __init__(
        self,
        intercap_data: Optional[str] = None,
        intercap_annot: Optional[str] = None,
        imhd_root: Optional[str] = None,
        split: str = "train",
        transform: Optional[IVDTransform] = None,
        keypoint_manager: Optional[KeypointManager] = None,
        renderer: Optional[MultiViewRenderer] = None,
        num_views: int = 4,
        render_on_fly: bool = True,
        num_object_points: int = 1024,
        num_object_queries: int = 87,
        cache_renders: bool = False,
        render_size: int = 256,
        load_contact_masks: bool = True,
        imhd_camera_id: int = 1,
        smplx_model_path: str = "smpl_models/SMPLX_NEUTRAL.npz",
        imhd_valid_indices_path: Optional[str] = None,
    ):
        datasets = []

        if intercap_data is not None:
            annot_path = intercap_annot or f"{intercap_data}/annotations"
            datasets.append(
                IntercapDataset(
                    intercap_data=intercap_data,
                    annot_data=annot_path,
                    split=split,
                    transform=transform,
                    keypoint_manager=keypoint_manager,
                    renderer=renderer,
                    num_views=num_views,
                    render_on_fly=render_on_fly,
                    num_object_points=num_object_points,
                    num_object_queries=num_object_queries,
                    cache_renders=cache_renders,
                    render_size=render_size,
                    load_contact_masks=load_contact_masks,
                )
            )

        if imhd_root is not None:
            imhd_split = "train" if split == "train" else "test"
            datasets.append(
                IMHDDataset(
                    imhd_root=imhd_root,
                    split=imhd_split,
                    transform=transform,
                    keypoint_manager=keypoint_manager,
                    num_object_points=num_object_points,
                    num_object_queries=num_object_queries,
                    load_contact_masks=load_contact_masks,
                    camera_id=imhd_camera_id,
                    smplx_model_path=smplx_model_path,
                    valid_indices_path=imhd_valid_indices_path,
                )
            )

        if not datasets:
            raise ValueError("At least one dataset root must be provided.")

        self._concat = ConcatDataset(datasets)

    def __len__(self) -> int:
        return len(self._concat)

    def __getitem__(self, idx: int) -> Dict:
        return self._concat[idx]


# Backward-compatible alias
IVDDataset = MixedIVDDataset


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function (same as InterCap)."""
    if any("object_coords" not in s for s in batch):
        missing = [s.get("sample_id", "<unknown>") for s in batch if "object_coords" not in s]
        raise KeyError(f"Missing object_coords for samples: {missing}")

    result: Dict = {}
    result["sample_id"] = [s["sample_id"] for s in batch]

    tensor_keys = [
        "rgb_image", "human_vertices",
        "object_points", "human_labels", "object_coords",
        "human_contact_mask", "object_contact_mask"
    ]

    for key in tensor_keys:
        if key not in batch[0]:
            continue
        values = [s[key] for s in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], np.ndarray):
            result[key] = torch.from_numpy(np.stack(values))

    return result


def create_dataloaders(
    intercap_data: Optional[str] = None,
    annot_data: Optional[str] = None,
    imhd_root: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 6,
    keypoint_manager: Optional[KeypointManager] = None,
    renderer: Optional[MultiViewRenderer] = None,
    image_size: int = 224,
    render_size: int = 256,
    load_contact_masks: bool = True,
    imhd_camera_id: int = 1,
    smplx_model_path: str = "smpl_models/SMPLX_NEUTRAL.npz",
    imhd_valid_indices_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders for InterCap + IMHD.
    If only one dataset is provided, it will be used alone.
    """
    train_transform = get_train_transforms(image_size, render_size)
    val_transform = get_val_transforms(image_size, render_size)

    train_dataset = MixedIVDDataset(
        intercap_data=intercap_data,
        intercap_annot=annot_data,
        imhd_root=imhd_root,
        split="train",
        transform=train_transform,
        keypoint_manager=keypoint_manager,
        renderer=renderer,
        render_on_fly=True,
        render_size=render_size,
        load_contact_masks=load_contact_masks,
        imhd_camera_id=imhd_camera_id,
        smplx_model_path=smplx_model_path,
        imhd_valid_indices_path=imhd_valid_indices_path,
    )

    val_dataset = MixedIVDDataset(
        intercap_data=intercap_data,
        intercap_annot=annot_data,
        imhd_root=imhd_root,
        split="val",
        transform=val_transform,
        keypoint_manager=keypoint_manager,
        renderer=renderer,
        render_on_fly=True,
        render_size=render_size,
        load_contact_masks=load_contact_masks,
        imhd_camera_id=imhd_camera_id,
        smplx_model_path=smplx_model_path,
        imhd_valid_indices_path=imhd_valid_indices_path,
    )

    test_dataset = MixedIVDDataset(
        intercap_data=intercap_data,
        intercap_annot=annot_data,
        imhd_root=imhd_root,
        split="test",
        transform=val_transform,
        keypoint_manager=keypoint_manager,
        renderer=renderer,
        render_on_fly=True,
        render_size=render_size,
        load_contact_masks=load_contact_masks,
        imhd_camera_id=imhd_camera_id,
        smplx_model_path=smplx_model_path,
        imhd_valid_indices_path=imhd_valid_indices_path,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader
