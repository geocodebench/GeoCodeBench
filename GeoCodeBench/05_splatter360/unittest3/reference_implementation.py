"""
Reference Implementation for Rotation Matrix Functions
This serves as the ground truth for testing LLM-generated implementations.
"""

import math
import torch


def get_rotation_x(angle, device='cpu'):
    """
    Generate rotation matrix for rotation around X-axis.
    
    Args:
        angle: Rotation angle in degrees
        device: Device to create the tensor on
        
    Returns:
        r_mat: 3x3 rotation matrix [3, 3]
    """
    angle = math.radians(angle)
    sin, cos = math.sin(angle), math.cos(angle)
    r_mat = torch.eye(3).to(device)
    r_mat[1, 1] = cos
    r_mat[1, 2] = -sin
    r_mat[2, 1] = sin
    r_mat[2, 2] = cos
    return r_mat


def get_rotation_y(angle, device='cpu'):
    """
    Generate rotation matrix for rotation around Y-axis.
    
    Args:
        angle: Rotation angle in degrees
        device: Device to create the tensor on
        
    Returns:
        r_mat: 3x3 rotation matrix [3, 3]
    """
    angle = math.radians(angle)
    sin, cos = math.sin(angle), math.cos(angle)
    r_mat = torch.eye(3).to(device)
    r_mat[0, 0] = cos
    r_mat[0, 2] = sin
    r_mat[2, 0] = -sin
    r_mat[2, 2] = cos
    return r_mat


def get_rotation_z(angle, device='cpu'):
    """
    Generate rotation matrix for rotation around Z-axis.
    
    Args:
        angle: Rotation angle in degrees
        device: Device to create the tensor on
        
    Returns:
        r_mat: 3x3 rotation matrix [3, 3]
    """
    angle = math.radians(angle)
    sin, cos = math.sin(angle), math.cos(angle)
    r_mat = torch.eye(3).to(device)
    r_mat[0, 0] = cos
    r_mat[0, 1] = -sin
    r_mat[1, 0] = sin
    r_mat[1, 1] = cos
    return r_mat

