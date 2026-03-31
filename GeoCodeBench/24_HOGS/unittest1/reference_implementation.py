"""
Reference Implementation for HOGS Gaussian Model Functions
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch
import numpy as np


class GaussianModel:
    """Simplified GaussianModel class for testing purposes."""
    
    def __init__(self):
        self._xyz = torch.empty(0)
        self._w = torch.empty(0)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_w(self):
        """Get the w parameter (distance to world origin)."""
        w = torch.exp(self._w)
        return w
    
    @property
    def get_w_inv(self):
        """Get the inverse of w parameter."""
        w = torch.exp(self._w)
        return 1 / w
    
    @property
    def get_means3D(self):
        """Get 3D means in Cartesian coordinates."""
        means3D = self.get_xyz * self.get_w_inv.unsqueeze(1)
        return means3D
    
    @property
    def get_points_hom(self):
        """Get points in homogeneous coordinates."""
        xyz = self.get_xyz
        points_hom = torch.stack([xyz[:, 0], xyz[:, 1], xyz[:, 2], self.get_w], dim=1)
        return points_hom
    
    def xyz_to_polar(self, xyz):
        """Convert XYZ coordinates to polar coordinates.
        
        Args:
            xyz: Tensor of shape (N, 3) containing XYZ coordinates
            
        Returns:
            polar_coord: Tensor of shape (N, 2) containing [theta, phi]
            inv_r: Tensor of shape (N,) containing 1/r
            r: Tensor of shape (N,) containing r
        """
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta_coord = torch.atan2(torch.sqrt(x ** 2 + y ** 2), z)
        phi_coord = torch.atan2(y, x)
        polar_coord = torch.stack([theta_coord, phi_coord], dim=1)
        return polar_coord, 1 / r, r
    
    def xyz_to_polar_np(self, xyz):
        """Convert XYZ coordinates to polar coordinates (NumPy version).
        
        Args:
            xyz: NumPy array of shape (N, 3) containing XYZ coordinates
            
        Returns:
            polar_coord: NumPy array of shape (N, 2) containing [theta, phi]
            inv_r: NumPy array of shape (N,) containing 1/r
            r: NumPy array of shape (N,) containing r
        """
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta_coord = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
        phi_coord = np.arctan2(y, x)
        polar_coord = np.stack([theta_coord, phi_coord], axis=1)
        return polar_coord, 1 / r, r

