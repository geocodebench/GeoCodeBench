
"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from jaxtyping import Float32
from rfstudio.graphics import Splats, TriangleMesh
from rfstudio.graphics.math import rot2quat, safe_normalize
from torch import Tensor


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
            ****EMPTY****
            if normal_interpolation:
                ****EMPTY****
            a = area * a_coeff
            splats += [
                ****EMPTY****
            ]
            if normal_interpolation:
                ****EMPTY****
        return (
            Splats.cat(splats, dim=0),
            torch.cat([offsets] * len(splats), dim=0),
        )
