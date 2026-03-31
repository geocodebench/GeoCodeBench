"""
Reference Implementation for get_rays_intersection
This serves as the ground truth for testing LLM-generated implementations.
"""

from __future__ import annotations

import torch
import numpy as np


def get_rays_intersection(ray_group_a, ray_group_b):
    """
    Compute intersection points between two groups of rays.
    
    Each ray is represented as a 6D vector: [start_x, start_y, start_z, dir_x, dir_y, dir_z]
    where (start_x, start_y, start_z) is the ray origin and (dir_x, dir_y, dir_z) is the direction.
    
    This function assumes ray_group_a and ray_group_b have the same number of rays (N = M),
    as guaranteed by the calling function region2zone_3d. It computes the closest
    points (P_A_i, P_B_i) for each ray pair (A_i, B_i) and returns all 2N points.
    
    Args:
        ray_group_a: Tensor of shape (N, 6) representing N rays from group A
        ray_group_b: Tensor of shape (N, 6) representing N rays from group B (N=M)
    
    Returns:
        Tensor of shape (2*N, 3) containing the N closest points from group A
        and the N closest points from group B, concatenated together.
    """
    # Ensure tensors are on the same device
    device = ray_group_a.device
    ray_group_a = ray_group_a.to(device)
    ray_group_b = ray_group_b.to(device)
    start_a = ray_group_a[:, :3]
    dir_a = ray_group_a[:, 3:]
    start_b = ray_group_b[:, :3]
    dir_b = ray_group_b[:, 3:]
    cross = torch.cross(dir_a, dir_b, dim=1)

    # Handle parallel rays (cross product is zero)
    cross_norm_sq = torch.sum(cross * cross, dim=1)
    eps = 1e-8
    
    # For parallel rays, set intersection to a large distance along the ray
    t_a = torch.sum(torch.cross(start_b - start_a, dir_b, dim=1) * cross, dim=1) / torch.clamp(cross_norm_sq, min=eps)
    t_b = torch.sum(torch.cross(start_a - start_b, dir_a, dim=1) * cross, dim=1) / torch.clamp(cross_norm_sq, min=eps)
    
    # For parallel rays, use a large distance instead of NaN
    parallel_mask = cross_norm_sq < eps
    t_a = torch.where(parallel_mask, torch.tensor(1000.0, device=device), t_a)
    t_b = torch.where(parallel_mask, torch.tensor(1000.0, device=device), t_b)

    intersection_points_a = start_a + t_a.view(-1, 1) * dir_a
    intersection_points_b = start_b + t_b.view(-1, 1) * dir_b * (-1)
    intersection_points = torch.cat((intersection_points_a, intersection_points_b), dim=0)
    return intersection_points
