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



def project_point_cloud_to_image_batch(point_cloud, images, intrinsics):
    """
    Project 3D point cloud to 2D image plane for a batch of images with corresponding intrinsics.
    
    Args:
    point_cloud (torch.Tensor): Point cloud with shape BxHxWx3
    images (torch.Tensor): Original images with shape Bx3xHxW
    intrinsics (torch.Tensor): Camera intrinsics with shape Bx3x3
    
    Returns:
    torch.Tensor: Projected image with shape Bx3xHxW
    """
    B, H, W, _ = point_cloud.shape
    
    # Reshape point cloud to BxHWx3
    points = point_cloud.view(B, H*W, 3)
     
    # Project points using camera intrinsics
    proj = torch.bmm(points, intrinsics.transpose(1, 2))  # BxHWx3
    
    # Normalize by z coordinate
    proj_normalized = proj[..., :2] / (proj[..., 2:] + 1e-7)  # BxHWx2
    
    # The projection now gives coordinates centered around (0,0)
    # Scale to image coordinates
    
    # Reshape to BxHxWx2
    grid = proj_normalized.view(B, H, W, 2)

    # Normalize grid coordinates to [-1, 1] range for grid_sample
    grid[..., 0] = (grid[..., 0] / W) * 2 - 1
    grid[..., 1] = (grid[..., 1] / H) * 2 - 1

    # Sample from original images
    projected_images = F.grid_sample(images, grid, align_corners=True)
    
    return projected_images  # Bx3xHxW


def first_flow_to_optical_flow(
    flow_3d_fwd, flow_3d_bwd,
    points_src, points_tgt,
    intrinsics_src, intrinsics_tgt,
    camera_pose_src, camera_pose_tgt, eps=1e-8):
    """
    Convert first camera space 3D flow to optical flow, given 3D points instead of depth.

    Args:
    flow_3d_fwd/flow_3d_bwd (torch.Tensor): First camera space 3D flow, shape (B, H, W, 3)
    points_src/points_tgt (torch.Tensor): 3D points in first camera space, shape (B, H, W, 3)
    intrinsics_src/intrinsics_tgt (torch.Tensor): Source and Target camera intrinsic parameters, shape (B, 3, 3)
    camera_pose_src (torch.Tensor): Source camera pose (world to camera), shape (B, 4, 4)
    camera_pose_tgt (torch.Tensor): Target camera pose (world to camera), shape (B, 4, 4)
    eps (float): Small value to avoid division by zero

    Returns:
    torch.Tensor: a list of Optical flow, shape (B, H, W, 2)
    """
    B, H, W, _ = flow_3d_fwd.shape

    # Convert camera coordinates to left camera coordinates
    
    # Apply first space flow to get first space point clouds
    points_src_moved = points_src + flow_3d_fwd
    points_tgt_moved = points_tgt + flow_3d_bwd

    # Convert left camera coordinates back to right camera coordinates (in target camera frame)
    points_src_moved_right = world_to_camera_coordinates(
        camera_to_world_coordinates(points_src_moved, camera_pose_src),
        camera_pose_tgt)
    points_tgt_right = world_to_camera_coordinates(
        camera_to_world_coordinates(points_tgt, camera_pose_src),
        camera_pose_tgt)

    # Compute camera space flow
    flow_3d_fwd_cam = points_src_moved_right - points_src
    flow_3d_bwd_cam = points_tgt_moved - points_tgt_right 

    # Use the provided function to convert camera space flow to optical flow
    optical_flow_fwd = scene_flow_to_optical_flow(
        flow_3d_fwd_cam, 
        intrinsics_src, points_src)
    optical_flow_bwd = scene_flow_to_optical_flow(
        flow_3d_bwd_cam,
        intrinsics_tgt, points_tgt_right
    )

    return optical_flow_fwd, optical_flow_bwd


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

