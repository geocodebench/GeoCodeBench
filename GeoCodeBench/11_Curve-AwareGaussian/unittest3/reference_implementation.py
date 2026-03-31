"""
Reference Implementation for curve_split_curvature function
This serves as the ground truth for testing LLM-generated implementations.
"""

from __future__ import annotations

import torch
from einops import rearrange


def curve_split_computation(rotation_matrix, sample_t, n_gaussians, threshold_angle, threshold_radian_skip):
    """Compute curvature-based split masks for curves.
    
    This function analyzes the curvature of curves based on rotation matrices,
    identifies points where the curve has high curvature, and determines
    split positions.
    
    Args:
        rotation_matrix: Rotation matrices for all gaussians, shape (num_curves * n_gaussians, 3, 3)
        sample_t: Sample parameter values, shape (n_gaussians, 1, 1)
        n_gaussians: Number of gaussians per curve
        threshold_angle: Threshold angle in degrees for splitting
        threshold_radian_skip: Threshold angle in degrees for skip splitting
        
    Returns:
        mask_split: Boolean mask indicating which curves should be split, shape (num_curves,)
        end_t: Split positions for each curve, shape (num_curves, 1, 1)
    """
    threshold_radian = threshold_angle * (torch.pi / 180)
    threshold_radian_skip_val = threshold_radian_skip * (torch.pi / 180)
    curvature = rearrange(rotation_matrix[..., 0], '(b m) c-> b m c', m=n_gaussians)
    cos_theta = torch.einsum('bij,bij->bi', curvature[:, :-1, :], curvature[:, 1:, :])
    angles = torch.acos(cos_theta.clamp(-1, 1))
    cos_theta_skip = torch.einsum('bij,bij->bi', curvature[:, :-2, :], curvature[:, 2:, :])
    angles_skip = torch.acos(cos_theta_skip.clamp(-1, 1))
    mask_split = torch.max(angles, dim=-1).values > threshold_radian
    mask_skip = torch.max(angles_skip, dim=-1).values > threshold_radian_skip_val
    mask_split |= mask_skip
    angles_max, t = torch.max(angles, dim=-1)
    end_t = sample_t[t] + 0.5 / n_gaussians
    return mask_split, end_t

