"""
Reference Implementation for compute_E
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch


def compute_E(verts, faces):
    """
    Compute the E matrix for triangular faces.
    
    Args:
        verts: Vertex positions, shape (..., N_verts, 3)
        faces: Face indices, shape (..., N_faces, 3)
    
    Returns:
        E: The computed E matrix, shape (..., N_faces, 3, 3)
    """
    # assert return_scale
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]

    
    r0 = v1 - v0
    r1 = v2 - v0 
    # r2 = safe_normalize(torch.cross(r0, v2 - v0, dim=-1))
    crs_tri = torch.cross(r0, r1, dim=-1)
    # Add epsilon to avoid division by zero for degenerate triangles
    crs_norm = torch.norm(crs_tri, dim=-1, keepdim=True, p=2)
    r2 = crs_tri / torch.sqrt(crs_norm + 1e-12)
    E = torch.cat([r0[..., None], r1[..., None], r2[..., None]], dim=-1)
    # E_inverse = torch.linalg.inv(E)
    return E

