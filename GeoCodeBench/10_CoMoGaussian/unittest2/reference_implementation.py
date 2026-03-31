"""
Reference Implementation for skew_symmetric, transform_SE3, and rodrigues_formula
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch
import numpy as np


def skew_symmetric(w: torch.Tensor) -> torch.Tensor:
    """
    Compute the skew-symmetric matrix from a 3D vector.
    
    Args:
        w: 3D vector, shape (*, 3)
    
    Returns:
        w_skew: Skew-symmetric matrix, shape (*, 3, 3)
    
    The skew-symmetric matrix is defined as:
        [ 0   -w3  w2 ]
        [ w3   0   -w1]
        [-w2  w1   0  ]
    """
    w1, w2, w3 = torch.chunk(w, 3, dim=-1)
    
    w_skew = torch.cat([torch.zeros_like(w1), -w3, w2,
                       w3, torch.zeros_like(w1), -w1,
                       -w2, w1, torch.zeros_like(w1)], dim=-1)
    w_skew = w_skew.reshape(-1, 3, 3)
    return w_skew


def transform_SE3(exp_w_skew: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Construct an SE(3) transformation matrix from rotation matrix and translation vector.
    
    Args:
        exp_w_skew: Rotation matrix (or exp of skew-symmetric), shape (*, 3, 3)
        p: Translation vector, shape (*, 3, 1)
    
    Returns:
        delta_Rt: SE(3) transformation matrix, shape (*, 4, 4)
        
    The SE(3) transformation has the form:
        [ R  t ]
        [ 0  1 ]
    where R is 3x3 rotation matrix and t is 3x1 translation vector.
    """
    # Add batch dimension if needed
    needs_batch = exp_w_skew.dim() == 2
    if needs_batch:
        exp_w_skew = exp_w_skew.unsqueeze(0)
    if p.dim() == 2:
        p = p.unsqueeze(0)
    
    delta_Rt = torch.cat([exp_w_skew, p], dim=-1)
    delta_Rt_fill = torch.tensor([0, 0, 0, 1])[None].repeat(delta_Rt.size(0), 1, 1).to(delta_Rt)
    delta_Rt = torch.cat([delta_Rt, delta_Rt_fill], dim=1)
    
    # Remove batch dimension if we added it
    if needs_batch:
        delta_Rt = delta_Rt.squeeze(0)
    
    return delta_Rt


def rodrigues_formula(w: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Compute the rotation matrix using Rodrigues' rotation formula.
    
    Args:
        w: Skew-symmetric matrix, shape (*, 3, 3)
        theta: Rotation angle, shape (*, 1)
    
    Returns:
        Rotation matrix, shape (*, 3, 3)
    
    Rodrigues' formula:
        R = I + sin(theta) * w + (1 - cos(theta)) * w^2
    where w is the normalized axis skew-symmetric matrix.
    """
    term1 = torch.eye(3).to(w)
    theta = theta.unsqueeze(-1)
    term2 = torch.sin(theta) * w
    term3 = (1 - torch.cos(theta)) * torch.matmul(w, w)
    return term1 + term2 + term3


