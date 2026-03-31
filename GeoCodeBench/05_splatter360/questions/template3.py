
"""
Template for LLM Implementation
Copy this file and fill in the function bodies with LLM-generated code.
"""

import math
import torch


def get_rotation_x(angle, device='cpu'):
    """
    Generate rotation matrix for rotation around X-axis.
    
    Input:
        angle: float - Rotation angle in degrees
        device: str - Device to create the tensor on (default: 'cpu')
        
    Output:
        torch.Tensor - 3x3 rotation matrix, shape [3, 3]
    
    Example:
        >>> r = get_rotation_x(90, device='cpu')
        >>> r.shape
        torch.Size([3, 3])
    """
    # TODO: Implement the rotation matrix for X-axis rotation
    
    raise NotImplementedError("Please implement this function")


def get_rotation_y(angle, device='cpu'):
    """
    Generate rotation matrix for rotation around Y-axis.
    
    Input:
        angle: float - Rotation angle in degrees
        device: str - Device to create the tensor on (default: 'cpu')
        
    Output:
        torch.Tensor - 3x3 rotation matrix, shape [3, 3]
    
    Example:
        >>> r = get_rotation_y(45, device='cpu')
        >>> r.shape
        torch.Size([3, 3])
    """
    # TODO: Implement the rotation matrix for Y-axis rotation
    
    raise NotImplementedError("Please implement this function")


def get_rotation_z(angle, device='cpu'):
    """
    Generate rotation matrix for rotation around Z-axis.
    
    Input:
        angle: float - Rotation angle in degrees
        device: str - Device to create the tensor on (default: 'cpu')
        
    Output:
        torch.Tensor - 3x3 rotation matrix, shape [3, 3]
    
    Example:
        >>> r = get_rotation_z(180, device='cpu')
        >>> r.shape
        torch.Size([3, 3])
    """
    # TODO: Implement the rotation matrix for Z-axis rotation
    
    raise NotImplementedError("Please implement this function")
