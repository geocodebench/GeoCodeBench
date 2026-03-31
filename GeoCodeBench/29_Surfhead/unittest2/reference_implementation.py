"""
Reference Implementation for compute_face_orientation
This serves as the ground truth for testing LLM-generated implementations.
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
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()
    # breakpoint()
    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]

      # will have artifacts without negation
    # if return_type == 'similarity':
    a0 = safe_normalize(v1 - v0)
    # a1 = safe_normalize(v2 - v0)
    # a2 = safe_normalize(torch.cross(a0, a1, dim=-1))
    a1 = safe_normalize(torch.cross(a0, v2 - v0, dim=-1))
    a2 = safe_normalize(torch.cross(a1, a0, dim=-1)) #! no negation by right hand drill-law
    orientation = torch.cat([a0[..., None], a2[..., None], a1[..., None]], dim=-1)
    #! 0 2 1 == base height perpendi
    
    #* original
    # a0 = safe_normalize(v1 - v0)
    # a1 = safe_normalize(torch.cross(a0, v2 - v0, dim=-1))
    # a2 = -safe_normalize(torch.cross(a1, a0, dim=-1))  # will have artifacts without negation

    # orientation = torch.cat([a0[..., None], a1[..., None], a2[..., None]], dim=-1)
    #* original
    if return_scale:
        # breakpoint()
        s0 = length(v1 - v0) #! a0 axis
        s1 = dot(a2, (v2 - v0)).abs() #! a2 axis
        # if scale_dim == 1:
            # scale = (s0 * s1) / 2
        scale = (s0 + s1) / 2
        # elif scale_dim == 2:
        #     s_dot5 = torch.one_like(s0).to(s0)
        #     scale = torch.cat([s0,s_dot5,s1],-1)
        #     # scale = torch.cat([s0,s1],-1)

        return orientation, scale
    else:
        return orientation

