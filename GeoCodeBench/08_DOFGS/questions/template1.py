
"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def distance_to_gaussian_surface(mean, svec, rotmat, query):
    """
    Compute the distance from query points to a 3D Gaussian surface.
    
    Args:
        mean: Center of the Gaussian, shape (..., 3)
        svec: Scale vector (sx, sy, sz) defining the ellipsoid axes, shape (..., 3)
        rotmat: Rotation matrix, shape (..., 3, 3)
        query: Query points, shape (..., 3)
    
    Returns:
        Distance to the Gaussian surface, shape (...,)
    """
    # TODO: Fill in LLM-generated code here
    
    raise NotImplementedError("Please implement this function")
