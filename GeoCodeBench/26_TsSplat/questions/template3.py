
"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
"""

import torch
import torch.nn.functional as F


def dot(x, y):
    """Helper function: dot product."""
    return torch.sum(x * y, -1, keepdim=True)


def _compute_vertex_normal(self):
    """
    Compute vertex normals from triangle mesh.
    
    Args:
        self: Object with attributes:
            - v_pos: vertex positions [num_vertices, 3]
            - t_pos_idx: triangle face indices [num_faces, 3]
    
    Returns:
        v_nrm: vertex normals [num_vertices, 3]
    """
    i0 = self.t_pos_idx[:, 0]
    i1 = self.t_pos_idx[:, 1]
    i2 = self.t_pos_idx[:, 2]

    v0 = self.v_pos[i0, :]
    v1 = self.v_pos[i1, :]
    v2 = self.v_pos[i2, :]

    # TODO: Fill in LLM-generated code here
    raise NotImplementedError("Please implement this function")

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_nrm))

    return v_nrm


def _compute_vertex_tangent(self):
    """
    Compute vertex tangents from triangle mesh with texture coordinates.
    
    Args:
        self: Object with attributes:
            - v_pos: vertex positions [num_vertices, 3]
            - t_pos_idx: triangle face indices [num_faces, 3]
            - v_tex: texture coordinates [num_tex_vertices, 2]
            - t_tex_idx: texture coordinate indices [num_faces, 3]
            - v_nrm: vertex normals [num_vertices, 3]
    
    Returns:
        tangents: vertex tangents [num_vertices, 3]
    """
    vn_idx = [None] * 3
    pos = [None] * 3
    tex = [None] * 3
    for i in range(0, 3):
        pos[i] = self.v_pos[self.t_pos_idx[:, i]]
        tex[i] = self.v_tex[self.t_tex_idx[:, i]]
        # t_nrm_idx is always the same as t_pos_idx
        vn_idx[i] = self.t_pos_idx[:, i]

    tangents = torch.zeros_like(self.v_nrm)
    tansum = torch.zeros_like(self.v_nrm)

    # Compute tangent space for each triangle
    # TODO: Fill in LLM-generated code here
    raise NotImplementedError("Please implement this function")

    # Avoid division by zero for degenerated texture coordinates
    tang = nom / torch.where(
        denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6)
    )

    # Update all 3 vertices
    for i in range(0, 3):
        idx = vn_idx[i][:, None].repeat(1, 3)
        # tangents[n_i] = tangents[n_i] + tang
        tangents.scatter_add_(0, idx, tang)
        tansum.scatter_add_(0, idx, torch.ones_like(tang))  # tansum[n_i] = tansum[n_i] + 1
    tangents = tangents / tansum

    # Normalize and make sure tangent is perpendicular to normal
    tangents = F.normalize(tangents, dim=1)
    tangents = F.normalize(tangents - dot(tangents, self.v_nrm) * self.v_nrm)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(tangents))

    return tangents
