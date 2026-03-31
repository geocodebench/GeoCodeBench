
"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
"""

import torch


def camera_to_world_coordinates(X_cam, camera_pose):
    """
    Convert camera coordinates to world coordinates.
    
    Args:
    - X_cam: torch.Tensor of shape (B, H, W, 3) or (B, N, 3) containing 3D points in camera coordinates
    - camera_pose: torch.Tensor of shape (B, 4, 4) representing the camera poses (world to camera transformation)
    
    Returns:
    - X_world: torch.Tensor of shape (B, H, W, 3) or (B, N, 3) containing 3D points in world coordinates
    """
    if len(X_cam.shape) == 4:
        B, H, W, _ = X_cam.shape
        reshape_needed = True
    elif len(X_cam.shape) == 3:
        B, N, _ = X_cam.shape
        reshape_needed = False
    else:
        raise ValueError("X_cam must have shape (B, H, W, 3) or (B, N, 3)")

    # Extract rotation and translation from camera pose
    R_world2cam = camera_pose[:, :3, :3]
    t_cam2world = camera_pose[:, :3, 3]

    # Compute inverse transformation
    R_cam2world = R_world2cam.transpose(1, 2).contiguous()  # Shape: (B, 3, 3)
    
    if reshape_needed:
        # Reshape X_cam to (B, H*W, 3)
        X_cam_reshaped = X_cam.view(B, H*W, 3)
    else:
        X_cam_reshaped = X_cam

    # Transform points
    X_world = torch.bmm(X_cam_reshaped, R_cam2world) + t_cam2world.unsqueeze(1)

    if reshape_needed:
        # Reshape back to (B, H, W, 3)
        X_world = X_world.view(B, H, W, 3)

    return X_world


def world_to_camera_coordinates(X_world, camera_pose):
    """
    Convert world coordinates to camera coordinates.
    
    Args:
    - X_world: torch.Tensor of shape (B, H, W, 3) or (B, N, 3) containing 3D points in world coordinates
    - camera_pose: torch.Tensor of shape (B, 4, 4) representing the camera poses (world to camera transformation)
    
    Returns:
    - X_cam: torch.Tensor of shape (B, H, W, 3) or (B, N, 3) containing 3D points in camera coordinates
    """
    if len(X_world.shape) == 4:
        B, H, W, _ = X_world.shape
        reshape_needed = True
    elif len(X_world.shape) == 3:
        B, N, _ = X_world.shape
        reshape_needed = False
    else:
        raise ValueError("X_world must have shape (B, H, W, 3) or (B, N, 3)")

    # Extract rotation and translation from camera pose
    R_world2cam = camera_pose[:, :3, :3]
    t_cam2world = camera_pose[:, :3, 3]

    if reshape_needed:
        # Reshape X_world to (B, H*W, 3)
        X_world_reshaped = X_world.view(B, H*W, 3)
    else:
        X_world_reshaped = X_world

    # Transform points
    X_cam = torch.bmm(X_world_reshaped - t_cam2world.unsqueeze(1), R_world2cam)

    if reshape_needed:
        # Reshape back to (B, H, W, 3)
        X_cam = X_cam.view(B, H, W, 3)

    return X_cam


def world_flow_to_optical_flow(flow_3d_fwd_world, points_3d, intrinsics_src, camera_pose_src, camera_pose_tgt, eps=1e-8):
    """
    Convert world space 3D flow to optical flow, given 3D points instead of depth.

    Args:
    flow_3d_fwd_world (torch.Tensor): World space 3D flow, shape (B, H, W, 3)
    points_3d (torch.Tensor): 3D points in source camera space, shape (B, H, W, 3)
    intrinsics_src (torch.Tensor): Source camera intrinsic parameters, shape (B, 3, 3)
    camera_pose_src (torch.Tensor): Source camera pose (world to camera), shape (B, 4, 4)
    camera_pose_tgt (torch.Tensor): Target camera pose (world to camera), shape (B, 4, 4)
    eps (float): Small value to avoid division by zero

    Returns:
    torch.Tensor: Optical flow, shape (B, H, W, 2)
    """
    B, H, W, _ = flow_3d_fwd_world.shape

    ****EMPTY****

    return optical_flow


def scene_flow_to_optical_flow(flow_3d_fwd, intrinsics, points_3d, eps=1e-8):
    """
    Convert camera space scene flow to optical flow.

    Args:
    flow_3d_fwd (torch.Tensor): Camera space scene flow, shape (B, H, W, 3)
    intrinsics (torch.Tensor): Camera intrinsic parameters, shape (B, 3, 3)
    points_3d (torch.Tensor): Unprojected point map in camera space, shape (B, H, W, 3)
    eps (float): Small value to avoid division by zero

    Returns:
    torch.Tensor: Optical flow, shape (B, H, W, 2)
    """
    B, H, W, _ = flow_3d_fwd.shape

    ****EMPTY****

    return optical_flow
