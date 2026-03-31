"""
Reference Implementation for distance_to_gaussian_surface
This serves as the ground truth for testing LLM-generated implementations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def distance_to_gaussian_surface(mean, svec, rotmat, query):
    """
    Compute the distance from query points to a 3D Gaussian surface.
    
    This function calculates the distance from query points to the surface of a 3D Gaussian
    defined by its mean, scale vector, and rotation matrix. The Gaussian is represented
    as an ellipsoid in 3D space.
    
    Args:
        mean: Center of the Gaussian, shape (..., 3)
        svec: Scale vector (sx, sy, sz) defining the ellipsoid axes, shape (..., 3)
        rotmat: Rotation matrix, shape (..., 3, 3)
        query: Query points, shape (..., 3)
    
    Returns:
        Distance to the Gaussian surface, shape (...,)
    """
    # Handle different input dimension combinations
    if mean.dim() == 1 and query.dim() == 1:
        # Single Gaussian, single query point
        mean = mean.unsqueeze(0)
        svec = svec.unsqueeze(0)
        rotmat = rotmat.unsqueeze(0)
        query = query.unsqueeze(0)
        single_point = True
    elif mean.dim() == 1 and query.dim() == 2:
        # Single Gaussian, multiple query points
        mean = mean.unsqueeze(0).expand(query.shape[0], -1)
        svec = svec.unsqueeze(0).expand(query.shape[0], -1)
        rotmat = rotmat.unsqueeze(0).expand(query.shape[0], -1, -1)
        single_point = False
    elif mean.dim() == 2 and query.dim() == 3:
        # Batch Gaussians, multiple query points per Gaussian
        # Reshape to (batch_size * num_queries, 3) for einsum
        batch_size, num_queries = query.shape[:2]
        mean_expanded = mean.unsqueeze(1).expand(-1, num_queries, -1).contiguous().view(-1, 3)
        svec_expanded = svec.unsqueeze(1).expand(-1, num_queries, -1).contiguous().view(-1, 3)
        rotmat_expanded = rotmat.unsqueeze(1).expand(-1, num_queries, -1, -1).contiguous().view(-1, 3, 3)
        query_flat = query.view(-1, 3)
        mean = mean_expanded
        svec = svec_expanded
        rotmat = rotmat_expanded
        query = query_flat
        single_point = False
        batch_3d = True
    else:
        # Batch case
        single_point = False
        batch_3d = False
    
    xyz = query - mean
    xyz = torch.einsum("bij,bj->bi", rotmat.transpose(-1, -2), xyz)
    xyz = F.normalize(xyz, dim=-1)
    z = xyz[..., 2]
    y = xyz[..., 1]
    x = xyz[..., 0]
    r_xy = torch.sqrt(x**2 + y**2 + 1e-10)
    cos_theta = z
    sin_theta = r_xy
    cos_phi = x / r_xy
    sin_phi = y / r_xy

    d2 = svec[..., 0] ** 2 * cos_phi**2 + svec[..., 1] ** 2 * sin_phi**2
    r2 = svec[..., 2] ** 2 * cos_theta**2 + d2**2 * sin_theta**2

    result = torch.sqrt(r2 + 1e-10)
    
    # Handle different output shapes
    if single_point:
        result = result.squeeze(0)
    elif 'batch_3d' in locals() and batch_3d:
        # Reshape back to (batch_size, num_queries)
        result = result.view(batch_size, num_queries)
    
    return result
