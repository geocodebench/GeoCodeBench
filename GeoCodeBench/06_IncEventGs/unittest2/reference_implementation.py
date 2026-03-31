"""
Reference Implementation for cubic_bspline_interpolation
This serves as the ground truth for testing LLM-generated implementations.
"""

from __future__ import annotations

import pypose as pp
import torch
from jaxtyping import Float
from pypose import LieTensor
from torch import Tensor

_EPS = 1e-6


def cubic_bspline_interpolation(
        ctrl_knots: Float[LieTensor, "*batch_size 4 7"],
        u: Float[Tensor, "interpolations"] | Float[Tensor, "*batch_size interpolations"],
        enable_eps: bool = False,
) -> Float[LieTensor, "*batch_size interpolations 7"]:
    """Cubic B-spline interpolation with batches of four SE(3) control knots.

    Args:
        ctrl_knots: The control knots.
        u: Normalized positions on the trajectory segments. Range: [0, 1].
        enable_eps: Whether to clip the normalized position with a small epsilon to avoid possible numerical issues.

    Returns:
        The interpolated poses.
    """
    batch_size = ctrl_knots.shape[:-2]
    interpolations = u.shape[-1]

    # If u only has one dim, broadcast it to all batches. This means same interpolations for all batches.
    # Otherwise, u should have the same batch size as the control knots (*batch_size, interpolations).
    if u.dim() == 1:
        u = u.tile((*batch_size, 1))  # (*batch_size, interpolations)
    if enable_eps:
        u = torch.clip(u, _EPS, 1.0 - _EPS)

    uu = u * u
    uuu = uu * u
    oos = 1.0 / 6.0  # one over six

    # t coefficients
    coeffs_t = torch.stack([
        oos - 0.5 * u + 0.5 * uu - oos * uuu,
        4.0 * oos - uu + 0.5 * uuu,
        oos + 0.5 * u + 0.5 * uu - 0.5 * uuu,
        oos * uuu
    ], dim=-2)

    # spline t
    t_t = torch.sum(pp.bvv(coeffs_t, ctrl_knots.translation()), dim=-3)

    # q coefficients
    coeffs_r = torch.stack([
        5.0 * oos + 0.5 * u - 0.5 * uu + oos * uuu,
        oos + 0.5 * u + 0.5 * uu - 2 * oos * uuu,
        oos * uuu
    ], dim=-2)

    # spline q
    q_adjacent = ctrl_knots[..., :-1, :].rotation().Inv() @ ctrl_knots[..., 1:, :].rotation()
    r_adjacent = q_adjacent.Log()
    q_ts = pp.Exp(pp.so3(pp.bvv(coeffs_r, r_adjacent)))
    q0 = ctrl_knots[..., 0, :].rotation()  # (*batch_size, 4)
    q_ts = torch.cat([
        q0.unsqueeze(-2).tile((interpolations, 1)).unsqueeze(-3),
        q_ts
    ], dim=-3)  # (*batch_size, num_ctrl_knots=4, interpolations, 4)
    q_t = pp.cumprod(q_ts, dim=-3, left=False)[..., -1, :, :]

    ret = pp.SE3(torch.cat([t_t, q_t], dim=-1))
    return ret

