"""
Reference Implementation for get_image_coor_from_world_points2
This is the ground truth implementation that all LLM implementations will be compared against.
"""

import torch


def get_image_coor_from_world_points2(points, view, mode=None, scale=1):
    """Project 3D world points to 2D image coordinates.
    
    Args:
        points: World space points, shape (n, 3)
        view: Camera view object containing camera parameters
        mode: Optional mode for coordinate normalization ("scale" or None)
        scale: Scale factor for intrinsic matrix (default: 1)
    
    Returns:
        coor_2d: 2D image coordinates, shape (2, n)
        coor_mask: Boolean mask indicating points outside the view, shape (n,)
        z: Depth values, shape (n,)
    """
    # points n 3 
    _, h, w = view.original_image.shape
    K = view.K.float().clone()  # 3, 3 - clone to avoid modifying view.K
    if scale != 1:
        K[:2, :] = K[:2, :] * scale
    w2c = view.w2c.float()  # 4 4
    p_h = torch.cat([points, torch.ones_like(points[:, 1, None])], dim=-1).permute(1, 0)  # n 4 -> 4 n
    cam_points_h = torch.matmul(w2c, p_h)  # 4 n
    cam_points = cam_points_h[:3]  # 3 n
    coor = torch.matmul(K, cam_points)  # 3 n 
    z = coor[2, None]  # 1 n
    coor_2d = (coor[:2] / z)  # ((width - 1) / 2) - 1 # 
    if mode == "scale":
        coor_2d[0] = coor_2d[0] / ((w - 1) / 2) - 1
        coor_2d[1] = coor_2d[1] / ((h - 1) / 2) - 1
        coor_mask = torch.logical_or(torch.abs(coor_2d[0]) > 1, torch.abs(coor_2d[1]) > 1)  # Fixed: use logical_or
    else:
        coor_mask = torch.logical_or(torch.abs(coor_2d[0] / ((w - 1) / 2) - 1) > 1, torch.abs(coor_2d[1] / ((h - 1) / 2) - 1) > 1)  # Fixed: use logical_or
    return coor_2d, coor_mask, z.squeeze()

