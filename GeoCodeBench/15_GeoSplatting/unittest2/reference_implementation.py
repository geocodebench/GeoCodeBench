"""
Reference Implementation of make() function
This is the correct implementation used for testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from jaxtyping import Float32
from torch import Tensor

# Simplified Splats class for testing (avoiding rfstudio dependency)
@dataclass
class Splats:
    """Simplified Splats class for unittest purposes."""
    means: Tensor
    scales: Tensor
    quats: Tensor
    colors: Tensor
    opacities: Tensor
    shs: Tensor
    
    def __getitem__(self, key):
        """Allow indexing like original Splats."""
        return Splats(
            means=self.means[key],
            scales=self.scales[key],
            quats=self.quats[key],
            colors=self.colors[key],
            opacities=self.opacities[key],
            shs=self.shs[key],
        )
    
    def replace_(self, **kwargs):
        """In-place replacement for testing."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
    
    @staticmethod
    def cat(splats_list: list, dim=0):
        """Concatenate a list of Splats objects."""
        return Splats(
            means=torch.cat([s.means for s in splats_list], dim=dim),
            scales=torch.cat([s.scales for s in splats_list], dim=dim),
            quats=torch.cat([s.quats for s in splats_list], dim=dim),
            colors=torch.cat([s.colors for s in splats_list], dim=dim),
            opacities=torch.cat([s.opacities for s in splats_list], dim=dim),
            shs=torch.cat([s.shs for s in splats_list], dim=dim),
        )


# Simplified TriangleMesh class for testing
@dataclass
class TriangleMesh:
    """Simplified TriangleMesh class for unittest purposes."""
    vertices: Tensor
    indices: Tensor
    normals: Tensor = None
    
    @property
    def num_faces(self):
        return len(self.indices)


def safe_normalize(x: Tensor, dim: int = -1, eps: float = 1e-10) -> Tensor:
    """Safely normalize a tensor along a dimension."""
    return x / x.norm(dim=dim, keepdim=True).clamp(min=eps)


def rot2quat(rotation_matrices: Float32[Tensor, "N 3 3"]) -> Float32[Tensor, "N 4"]:
    """
    Convert rotation matrices to quaternions.
    This is a simplified implementation for testing purposes.
    
    Args:
        rotation_matrices: Rotation matrices, shape [N, 3, 3]
        
    Returns:
        Quaternions in [w, x, y, z] format, shape [N, 4]
    """
    R = rotation_matrices
    
    # Trace of the matrix
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    
    # Determine which case to use based on trace
    quats = torch.zeros(R.shape[0], 4, dtype=R.dtype, device=R.device)
    
    # Case 1: trace > 0
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * qw
        quats[mask1, 0] = 0.25 * s
        quats[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
        quats[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
        quats[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s
    
    # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask2 = (~mask1) & (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2  # s = 4 * qx
        quats[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
        quats[mask2, 1] = 0.25 * s
        quats[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        quats[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s
    
    # Case 3: R[1,1] > R[2,2]
    mask3 = (~mask1) & (~mask2) & (R[..., 1, 1] > R[..., 2, 2])
    if mask3.any():
        s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2  # s = 4 * qy
        quats[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
        quats[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
        quats[mask3, 2] = 0.25 * s
        quats[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s
    
    # Case 4: else
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2  # s = 4 * qz
        quats[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
        quats[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
        quats[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
        quats[mask4, 3] = 0.25 * s
    
    # Normalize quaternion
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    
    return quats


@dataclass
class MGAdapter:

    scale_ratio1: float = 0.5
    scale_ratio2: float = 1.3
    g_scale_ratio: float = 1.6
    l_scale_ratio1: float = 1 / 3
    l_scale_ratio2: float = 3

    bias1: float = -1 / 24
    bias2: float = 0.0

    def bary2gs(
        self,
        p0: Float32[Tensor, "N 3"],
        p1: Float32[Tensor, "N 3"],
        area: Float32[Tensor, "N 1"],
        normals: Float32[Tensor, "N 3"],
        *,
        max_scale_ratio: float,
    ) -> Splats:
        means = (p0 + p1) / 2 # [N, 3]
        max_rots = p1 - means # [N, 3]
        max_scales: Tensor = (p1 - means).norm(dim=-1, keepdim=True).clamp(min=1e-10) # [N, 1]
        min_scales = area / 4 / max_scales # [N, 1]
        max_rots = max_rots / max_scales # [N, 3]
        scales = torch.cat((
            (self.g_scale_ratio * max_scale_ratio * max_scales).log(),
            (self.g_scale_ratio / max_scale_ratio * min_scales).log(),
            torch.empty_like(max_scales).fill_(-10),
        ), dim=-1)
        min_rots = normals.cross(max_rots, dim=-1) # [N, 3]
        quats = rot2quat(
            torch.stack((
                max_rots,
                min_rots,
                normals,
            ), dim=-1) # [N, 3, 3]
        ) # [N, 4]
        return Splats(
            means=means,
            scales=scales,
            quats=quats,
            colors=normals,
            opacities=torch.empty_like(means[:, :1]).fill_(0.99).logit(),
            shs=means.new_empty(means.shape[0], 0, 3),
        )

    def make(
        self,
        mesh: TriangleMesh,
        *,
        normal_interpolation: bool = True,
    ) -> Tuple[Splats, Float32[Tensor, "N 3"]]:

        splats = []

        p0 = mesh.vertices[mesh.indices[..., 0], :] # [F, 3]
        p1 = mesh.vertices[mesh.indices[..., 1], :] # [F, 3]
        p2 = mesh.vertices[mesh.indices[..., 2], :] # [F, 3]
        if normal_interpolation:
            vn0 = mesh.normals[mesh.indices[..., 0], :] # [F, 3]
            vn1 = mesh.normals[mesh.indices[..., 1], :] # [F, 3]
            vn2 = mesh.normals[mesh.indices[..., 2], :] # [F, 3]
        normals = (p1 - p0).cross(p2 - p0) # [F, 3]
        area: Tensor = normals.norm(dim=-1, keepdim=True).clamp(min=1e-10) / 2 # [F, 1]
        normals = safe_normalize(normals)
        offsets = normals.detach() * area.detach().sqrt()
        for u_coeff, a_coeff, s_ratio in zip(
            [1 / 9 + self.bias1, 2 / 9 + self.bias2],
            [1 / 4 * self.l_scale_ratio1, 1 / 12 * self.l_scale_ratio2],
            [self.scale_ratio1, self.scale_ratio2]
        ):
            u0 = p0 * (1 - 2 * u_coeff) + (p1 + p2) * u_coeff # [F, 3]
            u1 = p1 * (1 - 2 * u_coeff) + (p2 + p0) * u_coeff # [F, 3]
            u2 = p2 * (1 - 2 * u_coeff) + (p0 + p1) * u_coeff # [F, 3]
            if normal_interpolation:
                n0 = vn0 * (1 - 2 * u_coeff) + (vn1 + vn2) * u_coeff # [F, 3]
                n1 = vn1 * (1 - 2 * u_coeff) + (vn2 + vn0) * u_coeff # [F, 3]
                n2 = vn2 * (1 - 2 * u_coeff) + (vn0 + vn1) * u_coeff # [F, 3]
            a = area * a_coeff
            splats += [
                self.bary2gs(u0, u1, a, normals, max_scale_ratio=s_ratio),
                self.bary2gs(u1, u2, a, normals, max_scale_ratio=s_ratio),
                self.bary2gs(u2, u0, a, normals, max_scale_ratio=s_ratio),
            ]
            if normal_interpolation:
                splats[-3].replace_(colors=safe_normalize((n0 + n1) / 2))
                splats[-2].replace_(colors=safe_normalize((n1 + n2) / 2))
                splats[-1].replace_(colors=safe_normalize((n2 + n0) / 2))

        return (
            Splats.cat(splats, dim=0),
            torch.cat([offsets] * len(splats), dim=0),
        )

