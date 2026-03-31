"""
Reference Implementation for depth_to_point_cloud
This serves as the ground truth for testing LLM-generated implementations.
"""

import numpy as np


def depth_to_point_cloud(depth_map, intrinsic_matrix, c2w, mask, rgb_map):
    """Convert depth map to 3D point cloud in world coordinates.
    
    Args:
        depth_map: Depth values for each pixel, shape (H, W)
        intrinsic_matrix: Camera intrinsic matrix (3x3)
        c2w: Camera-to-world transformation matrix (4x4)
        mask: Binary mask indicating valid pixels, shape (H, W)
        rgb_map: RGB color values for each pixel, shape (H, W, 3)
    
    Returns:
        points_world: 3D points in world coordinates, shape (N, 3)
        rgb: RGB colors for each point, shape (N, 3)
    """
    # Get the image dimensions
    H, W = depth_map.shape
    
    # Create a grid of (u, v) coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()
    
    # Flatten the depth map, mask, and rgb map
    depth = depth_map.flatten()
    mask = mask.flatten()
    rgb_map = rgb_map.reshape(-1, 3)
    
    # Apply the mask
    u = u[mask == 1]
    v = v[mask == 1]
    depth = depth[mask == 1]
    rgb = rgb_map[mask == 1]
    
    # Intrinsic matrix components
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    # Convert pixel coordinates and depth to camera coordinates
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    # Stack into a 3xN array of 3D points in camera coordinates
    points_camera = np.vstack((x, y, z, np.ones_like(z)))
    
    # Transform to world coordinates using the c2w matrix
    points_world = c2w @ points_camera
    
    # Drop the homogeneous coordinate
    points_world = points_world[:3, :].T
    
    # Return points with their corresponding RGB colors
    return points_world, rgb

