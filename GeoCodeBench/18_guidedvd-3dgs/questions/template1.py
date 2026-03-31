
"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
"""

import numpy as np


def depth_to_point_cloud(depth_map, intrinsic_matrix, c2w, mask, rgb_map):
    """Convert depth map to 3D point cloud in world coordinates.
    Args:
        depth_map: Depth values for each pixel, shape (H, W)
                   Contains distance from camera to scene points
        intrinsic_matrix: Camera intrinsic matrix (3x3), format:
                         [[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]]
                         where fx, fy are focal lengths, cx, cy are principal point
        c2w: Camera-to-world transformation matrix (4x4)
             Converts points from camera space to world space
        mask: Binary mask indicating valid pixels, shape (H, W)
              1 = valid pixel, 0 = invalid pixel
        rgb_map: RGB color values for each pixel, shape (H, W, 3)
                 Color information to associate with 3D points
    
    Returns:
        points_world: 3D points in world coordinates, shape (N, 3)
                     where N is the number of valid (masked) pixels
        rgb: RGB colors for each point, shape (N, 3)
             Colors corresponding to each 3D point
    
    """
    # rgb_map: [h, w, 3]
    # depth_map: [h, w]

    # Get the image dimensions
    H, W = depth_map.shape

    # Create a grid of (u, v) coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()

    # TODO: Fill in LLM-generated code here

    
    raise NotImplementedError("Please implement this function")

    # Return points with their corresponding RGB colors
    return points_world, rgb
