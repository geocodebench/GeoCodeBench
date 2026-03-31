"""
Reference Implementation for get_opacity_with_3D_filter
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch
from torch import nn


class MockGaussianModel:
    """Mock GaussianModel for testing get_opacity_with_3D_filter."""
    
    def __init__(self, num_points, device=None):
        self.num_points = num_points
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize parameters
        self._opacity = nn.Parameter(torch.randn(num_points, 1, device=self.device))
        self._scaling = nn.Parameter(torch.randn(num_points, 3, device=self.device))
        self.filter_3D = torch.randn(num_points, 1, device=self.device) * 0.1 + 0.05
        
        # Setup functions
        self.opacity_activation = torch.sigmoid
        self.scaling_activation = torch.exp
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)


def get_opacity_with_3D_filter(model):
    """Reference implementation of get_opacity_with_3D_filter.
    
    Computes opacity with 3D filter applied, which modifies opacity based on 
    the ratio of covariance determinants before and after applying 3D filter.
    
    Args:
        model: MockGaussianModel instance with opacity, scaling, and filter_3D attributes
        
    Returns:
        Filtered opacity tensor of shape (num_points, 1)
    """
    opacity = model.opacity_activation(model._opacity)
    # apply 3D filter
    scales = model.get_scaling
    
    scales_square = torch.square(scales)
    det1 = scales_square.prod(dim=1)
    
    scales_after_square = scales_square + torch.square(model.filter_3D) 
    det2 = scales_after_square.prod(dim=1) 
    coef = torch.sqrt(det1 / det2)
    return opacity * coef[..., None]
