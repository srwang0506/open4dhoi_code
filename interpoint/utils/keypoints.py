"""
Keypoint Manager for 87 predefined body contact points on SMPL-X mesh
"""

import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class KeypointManager:
    """
    Manages the 87 predefined anatomical keypoints on SMPL-X body mesh.

    Includes 74 base points (hands, arms, legs, torso, neck, shoulders)
    plus 13 additional points (face, knees).

    Handles:
    - Loading keypoint definitions from JSON
    - Mapping 3D ground truth contacts to binary labels
    - Vertex index extraction for mesh operations
    """
    
    # Default keypoint definition (from part_kp.json)
    KEYPOINT_NAMES = [
        "leftToeBase", "leftShoulder_front", "rightShoulder_front",
        "leftShoulder_back", "rightShoulder_back", "leftNeck_front",
        "rightNeck_back", "leftNeck_back", "rightNeck_front", "rightToeBase",
        "leftUpperLeg_inner", "leftUpperLeg_outer", "rightUpperLeg_inner",
        "rightUpperLeg_outer", "leftForeArm_back", "leftForeArm_pinky",
        "leftForeArm_wrist", "leftForeArm_thumb", "rightForeArm_back",
        "rightForeArm_pinky", "rightForeArm_wrist", "rightForeArm_thumb",
        "leftUpperArm_up", "leftUpperArm_down", "leftUpperArm_back",
        "leftUpperArm_front", "rightUpperArm_back", "rightUpperArm_up",
        "rightUpperArm_front", "rightUpperArm_down", "leftLowerLeg_front",
        "rightLowerLeg_front", "rightLowerLeg_outer", "leftLowerLeg_outer",
        "leftLowerLeg_back", "rightLowerLeg_back", "leftLowerLeg_inner",
        "rightLowerLeg_inner", "rightUpperLeg_front", "rightUpperLeg_back",
        "leftUpperLeg_front", "leftUpperLeg_back", "rightFoot_instep",
        "leftFoot_instep", "rightFoot_sole", "leftFoot_sole", "buttocks_right",
        "buttocks_left", "leftHand_back", "leftHand_palm", "Thumb_left",
        "Index_left", "Middle_left", "Ring_left", "Pinky_left", "rightHand_back",
        "Thumb_right", "Index_right", "Middle_right", "Ring_right", "Pinky_right",
        "upperSpine_back", "upperSpine_right", "upperSpine_front", "upperSpine_left",
        "middleSpine_front", "middleSpine_right", "middleSpine_back", "middleSpine_left",
        "hip_front", "hip_left", "hip_back", "hip_right", "rightHand_palm"
    ]
    
    def __init__(self, keypoints_json: Optional[str] = None):
        """
        Initialize the keypoint manager.
        
        Args:
            keypoints_json: Path to the keypoint definition JSON file
        """
        self.keypoints: Dict[str, Dict] = {}
        self.vertex_indices: List[int] = []
        self.keypoint_positions: np.ndarray = None
        self.name_to_idx: Dict[str, int] = {}
        self.idx_to_name: Dict[int, str] = {}
        
        if keypoints_json is not None:
            self.load_keypoints(keypoints_json)
    
    def load_keypoints(self, json_path: str) -> None:
        """
        Load keypoint definitions from JSON file.
        
        Args:
            json_path: Path to the keypoint JSON file
        """
        with open(json_path, 'r') as f:
            self.keypoints = json.load(f)
        
        # Extract ordered list of vertex indices and positions
        self.vertex_indices = []
        positions = []
        
        for idx, name in enumerate(self.keypoints.keys()):
            self.name_to_idx[name] = idx
            self.idx_to_name[idx] = name
            self.vertex_indices.append(self.keypoints[name]['index'])
            positions.append(self.keypoints[name]['point'])
        
        self.keypoint_positions = np.array(positions, dtype=np.float32)
        
    def get_vertex_indices(self) -> np.ndarray:
        """Get array of vertex indices for the 74 keypoints."""
        return np.array(self.vertex_indices, dtype=np.int64)
    
    def get_keypoint_positions(self) -> np.ndarray:
        """Get canonical positions of the 74 keypoints."""
        return self.keypoint_positions
    
    def vertices_to_keypoints(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Extract keypoint positions from mesh vertices.
        
        Args:
            vertices: (B, V, 3) mesh vertices
            
        Returns:
            keypoints: (B, 74, 3) keypoint positions
        """
        indices = torch.tensor(self.vertex_indices, device=vertices.device)
        return vertices[:, indices, :]
    
    def contact_to_binary_labels(
        self, 
        contact_vertices: torch.Tensor,
        threshold: float = 0.05
    ) -> torch.Tensor:
        """
        Convert contact vertex indices to binary labels for 74 keypoints.
        
        Args:
            contact_vertices: (B, V) binary contact labels for all vertices
            threshold: Distance threshold for contact assignment
            
        Returns:
            labels: (B, 74) binary contact labels for keypoints
        """
        indices = torch.tensor(self.vertex_indices, device=contact_vertices.device)
        return contact_vertices[:, indices]
    
    def assign_contacts_by_distance(
        self,
        mesh_vertices: torch.Tensor,
        contact_points: torch.Tensor,
        threshold: float = 0.05
    ) -> torch.Tensor:
        """
        Assign binary contact labels based on distance to contact points.
        
        Args:
            mesh_vertices: (B, V, 3) mesh vertices
            contact_points: (B, N, 3) 3D contact points
            threshold: Distance threshold for contact
            
        Returns:
            labels: (B, 74) binary contact labels
        """
        B = mesh_vertices.shape[0]
        indices = torch.tensor(self.vertex_indices, device=mesh_vertices.device)
        
        # Extract keypoint positions
        keypoints = mesh_vertices[:, indices, :]  # (B, 74, 3)
        
        # Compute distances to all contact points
        # keypoints: (B, 74, 1, 3), contact_points: (B, 1, N, 3)
        distances = torch.cdist(keypoints, contact_points)  # (B, 74, N)
        
        # Min distance to any contact point
        min_distances = distances.min(dim=-1)[0]  # (B, 74)
        
        # Binary labels
        labels = (min_distances < threshold).float()
        
        return labels
    
    def get_body_part_groups(self) -> Dict[str, List[int]]:
        """
        Get grouping of keypoints by body part for analysis.
        
        Returns:
            Dictionary mapping body part names to keypoint indices
        """
        groups = {
            'left_hand': [],
            'right_hand': [],
            'left_arm': [],
            'right_arm': [],
            'left_leg': [],
            'right_leg': [],
            'torso': [],
            'neck': [],
            'feet': []
        }
        
        for name, idx in self.name_to_idx.items():
            name_lower = name.lower()
            if 'hand' in name_lower or 'thumb' in name_lower or \
               'index' in name_lower or 'middle' in name_lower or \
               'ring' in name_lower or 'pinky' in name_lower:
                if 'left' in name_lower:
                    groups['left_hand'].append(idx)
                else:
                    groups['right_hand'].append(idx)
            elif 'arm' in name_lower:
                if 'left' in name_lower:
                    groups['left_arm'].append(idx)
                else:
                    groups['right_arm'].append(idx)
            elif 'leg' in name_lower:
                if 'left' in name_lower:
                    groups['left_leg'].append(idx)
                else:
                    groups['right_leg'].append(idx)
            elif 'spine' in name_lower or 'hip' in name_lower or 'buttock' in name_lower:
                groups['torso'].append(idx)
            elif 'neck' in name_lower or 'shoulder' in name_lower:
                groups['neck'].append(idx)
            elif 'toe' in name_lower or 'foot' in name_lower:
                groups['feet'].append(idx)
        
        return groups
    
    def __len__(self) -> int:
        return len(self.keypoints)
    
    def __repr__(self) -> str:
        return f"KeypointManager(num_keypoints={len(self)})"


def create_keypoint_queries(
    num_keypoints: int = 74,
    d_model: int = 256,
    learnable: bool = True
) -> torch.nn.Parameter:
    """
    Create learnable query embeddings for keypoints.
    
    Args:
        num_keypoints: Number of keypoint queries
        d_model: Query embedding dimension
        learnable: Whether queries are learnable
        
    Returns:
        Query tensor (num_keypoints, d_model)
    """
    queries = torch.randn(num_keypoints, d_model) * 0.02
    
    if learnable:
        return torch.nn.Parameter(queries)
    else:
        return queries
