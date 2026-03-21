import torch

def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    Convert 3x3 rotation matrix to Axis-Angle vector (Rodrigues).
    Input: (B, 3, 3)
    Output: (B, 3)
    """
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)

def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    3x3 rotation matrix -> quaternion (w, x, y, z)
    """
    x_dim = 1
    batch_dim = rotation_matrix.shape[0]

    m00 = rotation_matrix[:, 0, 0]
    m01 = rotation_matrix[:, 0, 1]
    m02 = rotation_matrix[:, 0, 2]
    m10 = rotation_matrix[:, 1, 0]
    m11 = rotation_matrix[:, 1, 1]
    m12 = rotation_matrix[:, 1, 2]
    m20 = rotation_matrix[:, 2, 0]
    m21 = rotation_matrix[:, 2, 1]
    m22 = rotation_matrix[:, 2, 2]

    trace = m00 + m11 + m22

    def safe_sqrt(x):
        return torch.sqrt(torch.clamp(x, min=0))

    q_abs = safe_sqrt(torch.stack([
        1.0 + trace,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22
    ], dim=x_dim))

    # Choose the largest component for numerical stability
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[:, 0], m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[:, 1], m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[:, 2], m21 + m12], dim=-1),
        torch.stack([m10 - m01, m22 + m20, m21 + m12, q_abs[:, 3], ], dim=-1),
    ], dim=-2)

    flr = torch.argmax(q_abs, dim=-1)
    quat_candidates = quat_by_rijk / (2 * q_abs[..., None].max(dim=-1, keepdim=True)[0])

    # Select the most stable result based on argmax
    mask = torch.nn.functional.one_hot(flr, num_classes=4).to(rotation_matrix.dtype).unsqueeze(-1)
    quat = (quat_candidates * mask).sum(dim=-2)
    return quat

def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Quaternion (w, x, y, z) -> Axis-Angle
    """
    # Normalize quaternion
    q1 = quaternion[:, 1:]
    q0 = quaternion[:, 0]

    norm = torch.norm(q1, p=2, dim=1)

    # Prevent division by zero
    epsilon = 1e-8
    mask = norm > epsilon

    angle_axis = torch.zeros_like(q1)

    angle = 2 * torch.atan2(norm, q0)

    # Compute valid entries
    angle_axis[mask] = q1[mask] / norm[mask].unsqueeze(1) * angle[mask].unsqueeze(1)

    # Near-zero rotations: keep as zero (already initialized)

    return angle_axis
