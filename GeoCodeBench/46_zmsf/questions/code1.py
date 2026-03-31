# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import zmsf.utils.path_to_mast3r
import mast3r.utils.path_to_dust3r
from dust3r.utils.misc import invalid_to_zeros, invalid_to_nans
from dust3r.utils.geometry import normalize_pointcloud as normalize_pointcloud_dust3r


def visualize_camera_rays(pixels, pts3d, K, save_dir="test_vis.png", sample_rate=50):
    """
    Visualize 3D lines connecting camera center, pixel centers, and 3D points.
    
    Args:
    pixels: torch.Tensor of shape (B, H, W, 2)
    pts3d: torch.Tensor of shape (B, H, W, 3)
    K: torch.Tensor of shape (3, 3), camera intrinsic matrix
    sample_rate: int, sample every nth pixel to reduce clutter
    """
    # Ensure we're working with the first batch for simplicity
    pixels = pixels[0].cpu().numpy()
    pts3d = pts3d[0].cpu().numpy()
    K = K.cpu().numpy()
    
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Camera center (origin)
    camera_center = np.array([0, 0, 0])
    
    # Function to project pixel to 3D point on image plane
    def pixel_to_3d(pixel):
        x, y = pixel
        z = 1.0  # Set z to 1 (image plane)
        point_3d = np.linalg.inv(K) @ np.array([x, y, 1])
        return point_3d / point_3d[2]  # Normalize to ensure z=1
    
    # Sample pixels and 3D points
    H, W = pixels.shape[:2]
    for i in range(0, H, sample_rate):
        for j in range(0, W, sample_rate):
            pixel = pixels[i, j]
            point3d = pts3d[i, j]
            pixel[0] += W/2
            pixel[1] += H/2
            # Project pixel to 3D point on image plane
            pixel_3d = pixel_to_3d(pixel)
            
            # Line from camera center to pixel on image plane
            ax.plot([camera_center[0], pixel_3d[0]], 
                    [camera_center[1], pixel_3d[1]], 
                    [camera_center[2], pixel_3d[2]], 'r-', alpha=0.1)
            
            # Line from pixel on image plane to 3D point
            ax.plot([pixel_3d[0], point3d[0]], 
                    [pixel_3d[1], point3d[1]], 
                    [pixel_3d[2], point3d[2]], 'b-', alpha=0.1)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Rays Visualization')
    
    # Show the plot
    plt.savefig(save_dir)
    plt.close()



def compose_scene_flow(
    pts3d_left,
    flow_left_to_right,
    depth_change,
    intrinsics,
):
    """
    Compose Camera Space Scene Flow with Left camera space 3D Points, optical flow and depth change.
    
    Args:
    - pts3d_left: BxHxWx3
    - flow_left_to_right: BxHxWx2 in pixel space
    - depth_change: BxHxWx1
    - intrinsics: torch.Tensor of shape (B, 3, 3) containing camera intrinsics
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
    right_depth = (pts3d_left[..., -1:] + depth_change).squeeze(-1)

    pts3d_flowed = pixel_to_camera_coordinates(
        pixels=right_pixels, 
        depth=right_depth, 
        intrinsics=intrinsics)
    
    return pts3d_flowed - pts3d_left


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

    ****EMPTY****

    return scene_flow

