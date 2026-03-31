"""
Reference Implementation for de_casteljau_split
This serves as the ground truth for testing LLM-generated implementations.
"""

from __future__ import annotations

import torch
from torch import Tensor


def de_casteljau_split(curves: Tensor, t: Tensor, is_bezier: Tensor) -> tuple[Tensor, Tensor]:
    """
    Split Bezier curves using De Casteljau's algorithm.
    
    For Bezier curves:
    - Uses De Casteljau's algorithm to split at parameter t
    - Returns left and right curve segments
    
    For straight lines:
    - Linearly interpolates at parameter t
    - Returns left and right curve segments
    
    Args:
        curves: Curve control points, shape [B, 4, 3]
                For Bezier: P0, P1, P2, P3 (4 control points)
                For line: P0, P1, P2, P3 (only P0 and P3 used)
        t: Split parameter, shape [B] or scalar, range [0, 1]
        is_bezier: Boolean tensor indicating which curves are Bezier, shape [B]
    
    Returns:
        left: Left curve segments, shape [B, 4, 3]
        right: Right curve segments, shape [B, 4, 3]
    """
    # Ensure t is the right shape
    if t.dim() == 0:
        t = t.unsqueeze(0).expand(curves.shape[0])
    elif t.dim() == 1 and t.shape[0] == 1:
        t = t.expand(curves.shape[0])
    
    # Handle Bezier curves
    Q0 = (1 - t[:, None]) * curves[:, 0, :] + t[:, None] * curves[:, 1, :]
    Q1 = (1 - t[:, None]) * curves[:, 1, :] + t[:, None] * curves[:, 2, :]
    Q2 = (1 - t[:, None]) * curves[:, 2, :] + t[:, None] * curves[:, 3, :]
    R0 = (1 - t[:, None]) * Q0 + t[:, None] * Q1
    R1 = (1 - t[:, None]) * Q1 + t[:, None] * Q2
    S = (1 - t[:, None]) * R0 + t[:, None] * R1
    
    left_bezier = torch.stack([curves[:, 0], Q0, R0, S], dim=1)
    right_bezier = torch.stack([S, R1, Q2, curves[:, 3]], dim=1)
    
    # Handle straight lines
    S_line = (1 - t[:, None]) * curves[:, 0, :] + t[:, None] * curves[:, 3, :]
    
    left_straight = torch.stack([
        curves[:, 0, :],
        (2 / 3) * curves[:, 0, :] + (1 / 3) * S_line,
        (1 / 3) * curves[:, 0, :] + (2 / 3) * S_line,
        S_line
    ], dim=1)
    
    right_straight = torch.stack([
        S_line,
        (2 / 3) * S_line + (1 / 3) * curves[:, 3, :],
        (1 / 3) * S_line + (2 / 3) * curves[:, 3, :],
        curves[:, 3, :]
    ], dim=1)
    
    # Select based on is_bezier
    left = torch.where(is_bezier[:, None, None], left_bezier, left_straight)
    right = torch.where(is_bezier[:, None, None], right_bezier, right_straight)
    
    return left, right
