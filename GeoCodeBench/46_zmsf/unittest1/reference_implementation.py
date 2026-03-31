"""
Reference Implementation for compute_scene_flow()
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch
import torch.nn.functional as F


def compute_scene_flow(
    pts3d_left, 
    flow_left_to_right, 
    pts3d_right):
    """
    Compute scene flow in the left camera space.
    
    Args:
    pts3d_left: Left camera's 3D points (BxHxWx3)
    flow_left_to_right: Optical flow from left to right image (BxHxWx2)
    pts3d_right: Right camera's 3D points in left camera space (BxHxWx3)
    
    Returns:
    scene_flow: Scene flow for each point in pts3d_left (BxHxWx3)
    """
    B, H, W, _ = pts3d_left.shape
    device = pts3d_left.device

    # Create a grid of pixel coordinates
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), 
                                        torch.arange(W, device=device),
                                        indexing='ij')
    pixel_coords = torch.stack([x_coords, y_coords], dim=-1).float()  # HxWx2
    pixel_coords = pixel_coords.unsqueeze(0).expand(B, -1, -1, -1)  # BxHxWx2

    # Add flow to get corresponding pixels in the right image
    right_pixels = pixel_coords + flow_left_to_right

    # Normalize coordinates to [-1, 1] for grid_sample
    right_pixels_normalized = 2.0 * right_pixels / torch.tensor([W-1, H-1], device=device) - 1.0

    # Sample 3D points from right image using the flow
    right_pts3d_sampled = F.grid_sample(pts3d_right.permute(0, 3, 1, 2), 
                                        right_pixels_normalized,
                                        mode='bilinear', 
                                        padding_mode='border', 
                                        align_corners=True)
    right_pts3d_sampled = right_pts3d_sampled.permute(0, 2, 3, 1)

    # Compute scene flow
    scene_flow = right_pts3d_sampled - pts3d_left

    return scene_flow
