"""
Reference Implementation (Correct)
This is the correct implementation from spec_utils.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from quaternion_utils_mock import init_predefined_omega


class RenderingEquationEncoding(torch.nn.Module):
    def __init__(self, num_theta, num_phi, device='cpu', type='asg'):
        super(RenderingEquationEncoding, self).__init__()

        self.num_theta = num_theta
        self.num_phi = num_phi
        if type == 'asg':
            sample_type = 'full'
        elif type == 'lasg':
            sample_type = 'frontal'
        else:
            sample_type = 'full'
        omega, omega_la, omega_mu = init_predefined_omega(num_theta, num_phi, type=sample_type)
        self.omega = omega.view(1, num_theta, num_phi, 3).to(device)  # incoming direction; lobe direction
        self.omega_la = omega_la.view(1, num_theta, num_phi, 3).to(device)  # incoming tangent direction
        self.omega_mu = omega_mu.view(1, num_theta, num_phi, 3).to(device)  # incoming bitangent direction

    def forward(self, omega_o, a, la, mu, sg_type):
        """
        Forward pass of RenderingEquationEncoding.
        
        Args:
            omega_o: Input direction (reflection direction from view direction), shape (N, 3)
            a: Amplitude parameters, shape (N, num_theta, num_phi, 2)
            la: Lambda (sharpness) parameters, shape (N, num_theta, num_phi, 1)
            mu: Mu parameters, shape (N, num_theta, num_phi, 1)
            sg_type: Type of spherical Gaussian ('asg', 'lasg', 'sg', or 'sg_angle')
        
        Returns:
            out: Rendered output, shape (N, num_theta, num_phi, 2)
        """
        if sg_type == 'asg' or sg_type == 'lasg':
            Smooth = F.relu((omega_o[:, None, None] * self.omega).sum(dim=-1, keepdim=True))  # N, num_theta, num_phi, 1
            la = F.softplus(la - 1)  # lambda
            mu = F.softplus(mu - 1)  # mu
            exp_input = -la * (self.omega_la * omega_o[:, None, None]).sum(dim=-1, keepdim=True).pow(2) - mu * (
                    self.omega_mu * omega_o[:, None, None]).sum(dim=-1, keepdim=True).pow(2)
            out = a * Smooth * torch.exp(exp_input)

        if sg_type == 'sg':
            la = F.softplus(la - 1)  # lambda
            mu = F.softplus(mu - 1)  # mu
            
            cos_value = (self.omega * omega_o[:, None, None]).sum(dim=-1, keepdim=True)
            clamped_cos = torch.clamp(cos_value, -1 + 1e-11, 1 - 1e-11)
            exp_input = la * (clamped_cos - 1)
            out = (mu * torch.exp(exp_input)).repeat(1, 1, 1, 2)
        
        if sg_type == 'sg_angle':
            la = torch.exp(la)
            la = torch.clamp(la, min=1e-2)  
            C = 1 / (np.sqrt(2) * np.pi **(2/3) * la)
            cos_value = (self.omega * omega_o[:, None, None]).sum(dim=-1, keepdim=True)
            clamped_cos = torch.clamp(cos_value, -1 + 1e-11, 1 - 1e-11)
            exp_input = -0.5 * (torch.arccos(clamped_cos) / la).pow(2)
            out = (C * torch.exp(exp_input)).repeat(1, 1, 1, 2)
        
        return out

