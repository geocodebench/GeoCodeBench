"""
Reference Implementation for forward_event
This serves as the ground truth for testing LLM-generated implementations.

Based on the original code from scene_rep.py lines 393-414.
"""

import torch
from typing import Dict

log_eps = 1e-3
log = lambda x: torch.log(x + log_eps)


def forward_event(rays_o: torch.Tensor, rays_d: torch.Tensor, rgb_rendered: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Process event-based rendering output.
    
    This function simulates event camera behavior by computing logarithmic differences
    between two sets of rendered grayscale values. The input rgb_rendered is assumed
    to be the output from rendering with rays that represent two time steps.
    
    The key insight from the original code:
    - The batch of rays (Bs) represents rays from two consecutive time steps
    - The first half of rays corresponds to time t_k
    - The second half of rays corresponds to time t_k+Δt
    - Event accumulation = log(I_{k+Δt}) - log(I_k)
    
    Args:
        rays_o: Ray origins, shape (Bs, 3) - Not used in core logic but kept for interface compatibility
        rays_d: Ray directions, shape (Bs, 3) - Used to determine batch size
        rgb_rendered: Rendered RGB/grayscale values, shape (Bs,) or (Bs, 1) or (Bs, 3)
                     The batch size Bs contains rays from two time steps (should be even)
    
    Returns:
        A dictionary containing:
            - 'rgb': The original rendered RGB values (all Bs rays)
            - 'event_acc': Simulated event accumulation, shape (num_rays_half,) or (num_rays_half, C)
                          computed as log(grey_e2) - log(grey_e1)
    
    Implementation matches the original code:
        num_rays_half = int(rays_d.shape[0] / 2 + 0.5)
        grey_e1 = rend_dict["rgb"][:num_rays_half]  
        grey_e2 = rend_dict["rgb"][-num_rays_half:]
        simu_event = log(grey_e2) - log(grey_e1)
    """
    # Calculate half batch size (rounding up for odd batch sizes)
    # This matches: num_rays_half = int(rays_d.shape[0] / 2 + 0.5)
    num_rays_half = int(rays_d.shape[0] / 2 + 0.5)
    
    # Split rendered values into two halves (two time steps)
    # This matches: grey_e1 = rend_dict["rgb"][:num_rays_half]
    grey_e1 = rgb_rendered[:num_rays_half]  
    # This matches: grey_e2 = rend_dict["rgb"][-num_rays_half:]
    grey_e2 = rgb_rendered[-num_rays_half:]
    
    # Compute simulated event as logarithmic difference
    # This matches: simu_event = log(grey_e2) - log(grey_e1)
    simu_event = log(grey_e2) - log(grey_e1)
    
    # Return both rgb and event_acc
    # This matches: return {'rgb': rend_dict["rgb"], 'event_acc': simu_event}
    return {'rgb': rgb_rendered, 'event_acc': simu_event}

