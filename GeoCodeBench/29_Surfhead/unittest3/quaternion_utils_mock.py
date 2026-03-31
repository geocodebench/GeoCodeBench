"""
Mock implementation of init_predefined_omega for testing purposes.
This generates the required omega, omega_la, omega_mu tensors.
"""

import torch
import numpy as np


def init_predefined_omega(num_theta, num_phi, type='full'):
    """
    Initialize predefined omega directions for spherical Gaussian lighting.
    
    Args:
        num_theta: Number of theta (elevation) samples
        num_phi: Number of phi (azimuth) samples
        type: Sampling type - 'full' or 'frontal'
    
    Returns:
        omega: Lobe directions (num_theta * num_phi, 3)
        omega_la: Tangent directions (num_theta * num_phi, 3)
        omega_mu: Bitangent directions (num_theta * num_phi, 3)
    """
    if type == 'frontal':
        # Frontal hemisphere sampling (for lasg)
        theta_range = np.linspace(0, np.pi / 2, num_theta)
    else:
        # Full sphere sampling (for asg)
        theta_range = np.linspace(0, np.pi, num_theta)
    
    phi_range = np.linspace(0, 2 * np.pi, num_phi, endpoint=False)
    
    omega_list = []
    omega_la_list = []
    omega_mu_list = []
    
    for theta in theta_range:
        for phi in phi_range:
            # Spherical to Cartesian coordinates
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            omega = np.array([x, y, z])
            
            # Tangent direction (partial derivative w.r.t. theta)
            dx_dtheta = np.cos(theta) * np.cos(phi)
            dy_dtheta = np.cos(theta) * np.sin(phi)
            dz_dtheta = -np.sin(theta)
            omega_la = np.array([dx_dtheta, dy_dtheta, dz_dtheta])
            omega_la = omega_la / (np.linalg.norm(omega_la) + 1e-8)
            
            # Bitangent direction (partial derivative w.r.t. phi)
            dx_dphi = -np.sin(theta) * np.sin(phi)
            dy_dphi = np.sin(theta) * np.cos(phi)
            dz_dphi = 0
            omega_mu = np.array([dx_dphi, dy_dphi, dz_dphi])
            omega_mu = omega_mu / (np.linalg.norm(omega_mu) + 1e-8)
            
            omega_list.append(omega)
            omega_la_list.append(omega_la)
            omega_mu_list.append(omega_mu)
    
    omega = torch.tensor(np.array(omega_list), dtype=torch.float32)
    omega_la = torch.tensor(np.array(omega_la_list), dtype=torch.float32)
    omega_mu = torch.tensor(np.array(omega_mu_list), dtype=torch.float32)
    
    return omega, omega_la, omega_mu

