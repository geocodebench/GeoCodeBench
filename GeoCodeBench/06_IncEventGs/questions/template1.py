"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
"""

from __future__ import annotations

import pypose as pp
import torch
from jaxtyping import Float
from pypose import LieTensor
from torch import Tensor

_EPS = 1e-6


def linear_interpolation(
        ctrl_knots: Float[LieTensor, "*batch_size 2 7"],
        u: Float[Tensor, "interpolations"] | Float[Tensor, "*batch_size interpolations"],
        enable_eps: bool = False,
) -> Float[LieTensor, "*batch_size interpolations 7"]:
    """Linear interpolation between batches of two SE(3) poses.
    
    This function performs linear interpolation on SE(3) (Special Euclidean group) poses.
    SE(3) represents 3D rigid transformations (rotation + translation).
    
    Args:
        ctrl_knots: The control knots, shape (*batch_size, 2, 7)
                   Each SE(3) pose is represented as a 7D vector: [x, y, z, qw, qx, qy, qz]
                   where (x, y, z) is translation and (qw, qx, qy, qz) is rotation quaternion
        u: Normalized positions between two SE(3) poses. Range: [0, 1].
           Shape: (interpolations,) or (*batch_size, interpolations)
        enable_eps: Whether to clip the normalized position with a small epsilon 
                   to avoid possible numerical issues.
    
    Returns:
        The interpolated poses, shape (*batch_size, interpolations, 7)
    
    Example:
        >>> ctrl_knots = pp.randn_SE3(2)  # Two random SE3 poses
        >>> u = torch.linspace(0, 1, 10)  # 10 interpolation points
        >>> result = linear_interpolation(ctrl_knots, u)
        >>> result.shape
        torch.Size([10, 7])
    """
    # TODO: Fill in LLM-generated code here
    
    raise NotImplementedError("Please implement this function")
