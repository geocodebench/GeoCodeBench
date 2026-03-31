"""
Reference Implementation for finite_cone_formulation and points_in_finite_cone
This serves as the ground truth for testing LLM-generated implementations.
"""

from __future__ import annotations

import torch
from torch import Tensor


def finite_cone_formulation(top_point: Tensor, base_center: Tensor, radius: float) -> tuple[Tensor, float, float]:
    """Formulate a finite cone from top point, base center, and radius.
    
    Args:
        top_point: The apex of the cone, shape (3,)
        base_center: The center of the base circle, shape (3,)
        radius: The radius of the base circle
        
    Returns:
        direction: Unit vector from top to base center, shape (3,)
        height: Height of the cone
        half_angle_degrees: Cosine of half angle of the cone
    """
    height = torch.norm(top_point - base_center)
    slant_height = torch.sqrt(height**2 + radius**2)
    half_angle = torch.atan(radius / height)
    half_angle_degrees = torch.cos(half_angle)
    direction = base_center - top_point
    direction = direction / torch.norm(direction)
    
    return direction, height, half_angle_degrees


def points_in_finite_cone(points: Tensor, apex: Tensor, direction: Tensor, angle_cosine: float, height: float) -> Tensor:
    """Determine if points are within a finite cone defined by apex, direction, angle cosine, and height.
    
    Args:
        points: Points to test, shape (N, 3)
        apex: Apex of the cone, shape (3,)
        direction: Unit direction vector from apex, shape (3,)
        angle_cosine: Cosine of half angle of the cone
        height: Height of the cone
        
    Returns:
        mask: Boolean mask indicating which points are inside the cone, shape (N,)
    """
    direction_normalized = direction / torch.norm(direction)
    vectors = points - apex
    vector_lengths = torch.norm(vectors, dim=1)
    vector_norms = vectors / vector_lengths.unsqueeze(1)
    dot_products = torch.sum(vector_norms * direction_normalized, dim=1)
    mask_angle = dot_products >= angle_cosine
    mask_height = vector_lengths <= height
    return mask_angle & mask_height
