"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
"""

import torch
import torch.nn.functional as F


def rotation_between_z(vec):
    """
    Compute rotation matrix that rotates z-axis to align with vec.
    
    Args:
        vec: [..., 3] - target vector(s), can be any shape ending with 3
    
    Returns:
        R: [..., 3, 3] - rotation matrix/matrices
    
    Example:
        >>> vec = torch.tensor([1.0, 0.0, 0.0])  # x-axis
        >>> R = rotation_between_z(vec)
        >>> R.shape
        torch.Size([3, 3])
        >>> # R @ [0, 0, 1] should align with [1, 0, 0]
    """
    # TODO: Fill in LLM-generated code here
    
    raise NotImplementedError("Please implement this function")
