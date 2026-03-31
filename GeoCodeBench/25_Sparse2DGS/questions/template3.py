
"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
"""

import torch


def build_rotation(r):
    """
    Build rotation matrices from quaternions.
    
    This function converts quaternions to 3x3 rotation matrices.
    
    Args:
        r: Quaternions with shape [N, 4]
           Each quaternion is represented as a 4D vector: [w, x, y, z]
           where w is the scalar part and (x, y, z) is the vector part
    
    Returns:
        R: Rotation matrices with shape [N, 3, 3]
           Each matrix is a 3x3 orthogonal matrix representing the same rotation as the input quaternion
    
    Example:
        >>> quaternions = torch.randn(10, 4)  # 10 random quaternions
        >>> rotation_matrices = build_rotation(quaternions)
        >>> rotation_matrices.shape
        torch.Size([10, 3, 3])
    """
    # TODO: Fill in LLM-generated code here
    
    raise NotImplementedError("Please implement this function")
