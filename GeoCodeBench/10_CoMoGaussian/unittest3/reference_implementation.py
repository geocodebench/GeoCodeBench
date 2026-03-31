"""
Reference Implementation for get_rigid_transformation and get_cmr_transformation
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch
import torch.nn as nn
import math


class CoMoModuleRef:
    """Reference implementation of CoMoModule for testing.
    Only contains the methods we need to test.
    """
    
    def __init__(self, num_views=29, view_dim=32, num_warp=9):
        super().__init__()
        self.num_warp = num_warp
        
        # Initialize decoder modules (simplified for testing)
        self.decoder_rigid_w = nn.ModuleList([nn.Linear(view_dim // 2, 3) for _ in range(num_views)])
        self.decoder_rigid_v = nn.ModuleList([nn.Linear(view_dim // 2, 3) for _ in range(num_views)])
        self.decoder_rigid_theta = nn.ModuleList([nn.Linear(view_dim // 2, 1) for _ in range(num_views)])
        
        self.decoder_cmr_rot = nn.ModuleList([nn.Linear(view_dim // 2, 9) for _ in range(num_views)])
        self.decoder_cmr_trans = nn.ModuleList([nn.Linear(view_dim // 2, 3) for _ in range(num_views)])
        
        # Initialize with small weights
        gain = 0.00001 / (math.sqrt((view_dim // 2 + 3) / 6))
        for i in range(num_views):
            self._init(self.decoder_rigid_w[i], gain=gain)
            self._init(self.decoder_rigid_v[i], gain=gain)
            self._init(self.decoder_rigid_theta[i], gain=gain)
            self._init(self.decoder_cmr_rot[i], gain=gain)
            self._init(self.decoder_cmr_trans[i], gain=gain)
    
    def _init(self, layer, gain=0.00001):
        nn.init.xavier_uniform_(layer.weight, gain=gain)
        if layer.bias is not None:
            layer.bias.data.fill_(0)
    
    def get_rigid_transformation(self, latent_rigid, idx_view):
        """Reference implementation of get_rigid_transformation.
        
        Args:
            latent_rigid: torch.Tensor of shape (num_warp, view_dim)
            idx_view: int, index of the view
            
        Returns:
            T_rigid: torch.Tensor of shape (num_warp, 4, 4)
        """
        Z_RIGID_w, Z_RIGID_v = torch.chunk(latent_rigid, 2, dim=-1)
        
        w = self.decoder_rigid_w[idx_view](Z_RIGID_w)
        theta = self.decoder_rigid_theta[idx_view](Z_RIGID_w)[..., None]
        v = self.decoder_rigid_v[idx_view](Z_RIGID_v)
        
        w = self.exp_map(w)
        w_skew = self.skew_symmetric(w)
        R_exp = self.rodrigues_formula(w_skew, theta)
        G = self.G_formula(w_skew, theta)
        p = torch.matmul(G, v[..., None])
        T_rigid = self.transform_SE3(R_exp, p)
        
        return T_rigid
    
    def get_cmr_transformation(self, latent_cmr, idx_view):
        """Reference implementation of get_cmr_transformation.
        
        Args:
            latent_cmr: torch.Tensor of shape (num_warp, view_dim)
            idx_view: int, index of the view
            
        Returns:
            T_cmr: torch.Tensor of shape (num_warp, 4, 4)
            R_cmr: torch.Tensor of shape (num_warp, 3, 3)
        """
        Z_CMR_rot, Z_CMR_trans = torch.chunk(latent_cmr, 2, dim=-1)
        
        # Get the actual num_warp from the input tensor shape
        actual_num_warp = latent_cmr.shape[0]
        
        R_cmr = self.decoder_cmr_rot[idx_view](Z_CMR_rot).reshape(-1, 3, 3) \
            + torch.eye(3)[None].repeat(actual_num_warp, 1, 1).to(Z_CMR_rot)
        t_cmr = self.decoder_cmr_trans[idx_view](Z_CMR_trans)[..., None]
        T_cmr = self.transform_SE3(R_cmr, t_cmr)
        
        return T_cmr, R_cmr
    
    def transform_SE3(self, exp_w_skew, p):
        """Transform to SE3 format."""
        delta_Rt = torch.cat([exp_w_skew, p], dim=-1)
        delta_Rt_fill = torch.tensor([0, 0, 0, 1])[None].repeat(delta_Rt.size(0), 1, 1).to(delta_Rt)
        delta_Rt = torch.cat([delta_Rt, delta_Rt_fill], dim=1)
        return delta_Rt
    
    def rodrigues_formula(self, w, theta):
        """Rodrigues formula for rotation matrix."""
        term1 = torch.eye(3).to(w)
        term2 = torch.sin(theta) * w
        term3 = (1 - torch.cos(theta)) * torch.matmul(w, w)
        return term1 + term2 + term3
    
    def G_formula(self, w, theta):
        """G formula for SE(3) exponential map."""
        term1 = torch.eye(3)[None].to(w) * theta
        term2 = (1 - torch.cos(theta)) * w
        term3 = (theta - torch.sin(theta)) * torch.matmul(w, w)
        return term1 + term2 + term3
    
    def exp_map(self, w):
        """Normalize w to unit vector."""
        norm = torch.norm(w, dim=-1)[..., None] + 1e-10
        w = w / norm
        return w
    
    def skew_symmetric(self, w):
        """Convert vector to skew-symmetric matrix."""
        w1, w2, w3 = torch.chunk(w, 3, dim=-1)
        
        w_skew = torch.cat([torch.zeros_like(w1), -w3, w2,
                           w3, torch.zeros_like(w1), -w1,
                           -w2, w1, torch.zeros_like(w1)], dim=-1)
        w_skew = w_skew.reshape(-1, 3, 3)
        return w_skew

