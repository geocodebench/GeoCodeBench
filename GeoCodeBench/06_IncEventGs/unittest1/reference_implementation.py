"""
Reference Implementation for linear_interpolation
This serves as the ground truth for testing LLM-generated implementations.
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

    Args:
        ctrl_knots: The control knots.
        u: Normalized positions between two SE(3) poses. Range: [0, 1].
        enable_eps: Whether to clip the normalized position with a small epsilon to avoid possible numerical issues.

    Returns:
        The interpolated poses.
    """
    start_pose, end_pose = ctrl_knots[..., 0, :], ctrl_knots[..., 1, :]
    batch_size = start_pose.shape[:-1]
    interpolations = u.shape[-1]

    t_start, q_start = start_pose.translation(), start_pose.rotation()
    t_end, q_end = end_pose.translation(), end_pose.rotation()

    # If u only has one dim, broadcast it to all batches. This means same interpolations for all batches.
    # Otherwise, u should have the same batch size as the control knots (*batch_size, interpolations).
    if u.dim() == 1:
        u = u.tile((*batch_size, 1))  # (*batch_size, interpolations)
    if enable_eps:
        u = torch.clip(u, _EPS, 1.0 - _EPS)

    t = pp.bvv(1 - u, t_start) + pp.bvv(u, t_end)

    q_tau_0 = q_start.Inv() @ q_end
    r_tau_0 = q_tau_0.Log()
    q_t_0 = pp.Exp(pp.so3(pp.bvv(u, r_tau_0)))
    q = q_start.unsqueeze(-2).tile((interpolations, 1)) @ q_t_0

    ret = pp.SE3(torch.cat([t, q], dim=-1))
    return ret

