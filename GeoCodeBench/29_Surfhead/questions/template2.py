
"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
"""

import torch


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)


def length(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return x / length(x, eps)


def compute_face_orientation(verts, faces, return_scale=True):
    """
    Compute face orientation from vertices and face indices.
    
    Args:
        verts: Vertices tensor of shape (..., num_verts, 3)
        faces: Face indices tensor of shape (..., num_faces, 3)
        return_scale: Whether to return scale information
        
    Returns:
        orientation: Face orientation tensor of shape (..., num_faces, 3, 3)
        scale: Scale tensor of shape (..., num_faces, 1) (if return_scale=True)
    """
    assert return_scale
    # TODO: Fill in LLM-generated code here
    ****EMPTY****
    #* original
    if return_scale:
        ****EMPTY****

        return orientation, scale
    else:
        return orientation
