"""
Data transforms and augmentations for InterActVLM-Discrete (IVD)
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Callable
import torchvision.transforms as T
import torchvision.transforms.functional as TF

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


class IVDTransform:
    """
    Combined transforms for IVD training and evaluation.
    
    Handles:
    - RGB image augmentation
    - Rendered view normalization
    - Geometric consistency for 3D data
    """
    
    def __init__(
        self,
        image_size: int = 224,
        render_size: int = 256,
        is_train: bool = True,
        use_augmentation: bool = True,
        normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize transforms.
        
        Args:
            image_size: Target size for RGB images
            render_size: Target size for rendered views
            is_train: Whether in training mode
            use_augmentation: Whether to apply augmentations
            normalize_mean: ImageNet mean for normalization
            normalize_std: ImageNet std for normalization
        """
        self.image_size = image_size
        self.render_size = render_size
        self.is_train = is_train
        self.use_augmentation = use_augmentation and is_train
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup transform pipelines."""
        # RGB image transforms
        if self.use_augmentation:
            self.rgb_transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                # Do not flip RGB alone: labels/object coords are index-aligned and
                # must be transformed synchronously with geometry.
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.ToTensor(),
                T.Normalize(mean=self.normalize_mean, std=self.normalize_std)
            ])
        else:
            self.rgb_transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=self.normalize_mean, std=self.normalize_std)
            ])
        
        # Render view transforms (minimal augmentation to preserve geometry)
        self.render_transform = T.Compose([
            T.Resize((self.render_size, self.render_size)),
            T.ToTensor(),
            T.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])
    
    def transform_rgb(self, image: np.ndarray) -> torch.Tensor:
        """
        Transform RGB image.
        
        Args:
            image: (H, W, 3) numpy array or PIL Image
            
        Returns:
            Transformed tensor (3, image_size, image_size)
        """
        from PIL import Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self.rgb_transform(image)
    
    def transform_render(self, render: np.ndarray) -> torch.Tensor:
        """
        Transform rendered view.
        
        Args:
            render: (H, W, 3) numpy array or PIL Image
            
        Returns:
            Transformed tensor (3, render_size, render_size)
        """
        from PIL import Image
        if isinstance(render, np.ndarray):
            render = Image.fromarray(render)
        return self.render_transform(render)
    
    def transform_batch_renders(self, renders: torch.Tensor) -> torch.Tensor:
        """
        Transform batch of rendered views.
        
        Args:
            renders: (B*J, 3, H, W) rendered views
            
        Returns:
            Normalized renders (B*J, 3, render_size, render_size)
        """
        # Resize if needed
        if renders.shape[-1] != self.render_size:
            renders = TF.resize(renders, [self.render_size, self.render_size])
        
        # Normalize
        mean = torch.tensor(self.normalize_mean, device=renders.device).view(1, 3, 1, 1)
        std = torch.tensor(self.normalize_std, device=renders.device).view(1, 3, 1, 1)
        renders = (renders - mean) / std
        
        return renders
    
    def __call__(self, sample: Dict) -> Dict:
        """
        Apply transforms to a sample.
        
        Args:
            sample: Dictionary containing:
                - 'rgb_image': Original scene image
                - 'render_views': Multi-view renders (optional, can be generated)
                - 'human_labels': Binary contact labels
                - 'object_coords': Object contact coordinates
                
        Returns:
            Transformed sample
        """
        result = {}
        
        # Transform RGB image
        if 'rgb_image' in sample:
            result['rgb_image'] = self.transform_rgb(sample['rgb_image'])
        
        # Transform render views if provided
        if 'render_views' in sample:
            if isinstance(sample['render_views'], torch.Tensor):
                result['render_views'] = self.transform_batch_renders(sample['render_views'])
            else:
                # List of numpy arrays
                renders = [self.transform_render(r) for r in sample['render_views']]
                result['render_views'] = torch.stack(renders)
        
        # Copy labels (no transform needed)
        if 'human_labels' in sample:
            result['human_labels'] = torch.tensor(
                sample['human_labels'], dtype=torch.float32
            )
        
        if 'object_coords' in sample:
            result['object_coords'] = torch.tensor(
                sample['object_coords'], dtype=torch.float32
            )
        
        # Copy other fields
        for key in sample:
            if key not in result:
                result[key] = sample[key]
        
        return result


def get_train_transforms(
    image_size: int = 224,
    render_size: int = 256,
    use_augmentation: bool = True
) -> IVDTransform:
    """Get training transforms."""
    return IVDTransform(
        image_size=image_size,
        render_size=render_size,
        is_train=True,
        use_augmentation=use_augmentation
    )


def get_val_transforms(
    image_size: int = 224,
    render_size: int = 256
) -> IVDTransform:
    """Get validation/test transforms."""
    return IVDTransform(
        image_size=image_size,
        render_size=render_size,
        is_train=False,
        use_augmentation=False
    )


class SynchronizedAugmentation:
    """
    Synchronized augmentations for RGB image and corresponding 3D data.
    
    Ensures that flipping operations are applied consistently to:
    - RGB image
    - Human mesh
    - Object point cloud
    - Contact labels
    """
    
    def __init__(self, p_flip: float = 0.5):
        """
        Initialize synchronized augmentation.
        
        Args:
            p_flip: Probability of horizontal flip
        """
        self.p_flip = p_flip
        
        # Left-right correspondence for SMPL-X keypoints
        self.lr_pairs = [
            (0, 9),   # leftToeBase <-> rightToeBase
            (1, 2),   # leftShoulder_front <-> rightShoulder_front
            (3, 4),   # leftShoulder_back <-> rightShoulder_back
            (5, 8),   # leftNeck_front <-> rightNeck_front
            (6, 7),   # rightNeck_back <-> leftNeck_back
            (10, 12), # leftUpperLeg_inner <-> rightUpperLeg_inner
            (11, 13), # leftUpperLeg_outer <-> rightUpperLeg_outer
            # ... add more pairs as needed
        ]
    
    def flip_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Flip contact labels to match flipped image.
        
        Args:
            labels: (87,) binary contact labels
            
        Returns:
            Flipped labels (87,)
        """
        flipped = labels.copy()
        for left_idx, right_idx in self.lr_pairs:
            flipped[left_idx], flipped[right_idx] = labels[right_idx], labels[left_idx]
        return flipped
    
    def flip_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """
        Flip mesh vertices horizontally.
        
        Args:
            vertices: (V, 3) vertex positions
            
        Returns:
            Flipped vertices (V, 3)
        """
        flipped = vertices.copy()
        flipped[:, 0] = -flipped[:, 0]  # Flip X coordinate
        return flipped
    
    def flip_points(self, points: np.ndarray) -> np.ndarray:
        """
        Flip point cloud horizontally.
        
        Args:
            points: (N, 3) point coordinates
            
        Returns:
            Flipped points (N, 3)
        """
        flipped = points.copy()
        flipped[:, 0] = -flipped[:, 0]
        return flipped
    
    def __call__(self, sample: Dict) -> Dict:
        """
        Apply synchronized augmentations.
        
        Args:
            sample: Data sample dictionary
            
        Returns:
            Augmented sample
        """
        if np.random.random() > self.p_flip:
            return sample
        
        result = sample.copy()
        
        # Flip RGB image
        if 'rgb_image' in sample:
            result['rgb_image'] = np.fliplr(sample['rgb_image']).copy()
        
        # Flip human contact labels
        if 'human_labels' in sample:
            result['human_labels'] = self.flip_labels(sample['human_labels'])
        
        # Flip human vertices
        if 'human_vertices' in sample:
            result['human_vertices'] = self.flip_vertices(sample['human_vertices'])
        
        # Flip object points
        if 'object_points' in sample:
            result['object_points'] = self.flip_points(sample['object_points'])
        
        # Object labels are index-aligned and do not require geometric flipping.
        if 'object_coords' in sample:
            result['object_coords'] = self.flip_points(sample['object_coords'])
        
        return result
