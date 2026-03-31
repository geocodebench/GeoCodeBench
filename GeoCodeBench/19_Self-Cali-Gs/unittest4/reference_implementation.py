"""
Reference Implementation for apply_flow_up_down_left_right
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch
import torch.nn.functional as F


def homogenize(X: torch.Tensor):
    assert X.ndim == 2
    assert X.shape[1] in (2, 3)
    return torch.cat(
        (X, torch.ones((X.shape[0], 1), dtype=X.dtype, device=X.device)), dim=1
    )


def dehomogenize(X: torch.Tensor):
    assert X.ndim == 2
    assert X.shape[1] in (3, 4)
    return X[:, :-1] / X[:, -1:]


def apply_flow_up_down_left_right(viewpoint_cam, rays_dis_hom, img, types="forward", is_fisheye=False, iteration=None):
    """
    Apply flow transformation for different camera directions (left/right/up/down).
    
    Args:
        viewpoint_cam: Camera object with image_width, image_height, and get_K attributes
        rays_dis_hom: Homogeneous ray directions [N, 3]
        img: Input image tensor [C, H, W]
        types: Direction type - 'forward', 'left', 'right', 'up', or 'down'
        is_fisheye: Whether the camera is fisheye
        iteration: Current iteration (optional)
    
    Returns:
        distorted_img: Distorted image [C, H, W]
        img: Original image [C, H, W]
    """
    width = viewpoint_cam.image_width
    height = viewpoint_cam.image_height
    K = viewpoint_cam.get_K
    
    if types == 'left':
        x = rays_dis_hom[:, 0]  # First column (x)
        y = rays_dis_hom[:, 1]  # Second column (y)
        z = rays_dis_hom[:, 2]  # Third column (z)
        P_left = torch.stack((-z / x, -y / x), dim=1)  # Shape: [N, 2]
        rays_dis_hom = homogenize(P_left)
    elif types == 'right':
        x = rays_dis_hom[:, 0]  # First column (x)
        y = rays_dis_hom[:, 1]  # Second column (y)
        z = rays_dis_hom[:, 2]  # Third column (z)
        P_right = torch.stack((-z / x, y / x), dim=1)  # Shape: [N, 2]
        rays_dis_hom = homogenize(P_right)
    elif types == 'up':
        x = rays_dis_hom[:, 0]  # First column (x)
        y = rays_dis_hom[:, 1]  # Second column (y)
        z = rays_dis_hom[:, 2]  # Third column (z)
        P_up = torch.stack((-x / y, -z / y), dim=1)  # Shape: [N, 2]
        rays_dis_hom = homogenize(P_up)
    elif types == 'down':
        x = rays_dis_hom[:, 0]  # First column (x)
        y = rays_dis_hom[:, 1]  # Second column (y)
        z = rays_dis_hom[:, 2]  # Third column (z)
        P_down = torch.stack((x / y, -z / y), dim=1)  # Shape: [N, 2]
        rays_dis_hom = homogenize(P_down)

    rays_dis_inside = dehomogenize((K @ rays_dis_hom.T).T).reshape(height, width, 2)

    # apply flow field
    x_coords = rays_dis_inside[..., 0]  # Shape: [height, width]
    y_coords = rays_dis_inside[..., 1]  # Shape: [height, width]
    x_coords_norm = (x_coords / (img.shape[2] - 1)) * 2 - 1
    y_coords_norm = (y_coords / (img.shape[1] - 1)) * 2 - 1
    grid = torch.stack((x_coords_norm, y_coords_norm), dim=-1)  # Shape: [height, width, 2]
    grid = grid.unsqueeze(0)

    img_forward_batch = img.unsqueeze(0)  # Shape: [1, C, H, W]
    distorted_img = F.grid_sample(
        img_forward_batch,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True)
    distorted_img = distorted_img.squeeze(0)  # Shape: [C, H, W]

    return distorted_img, img

