
"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
"""

import torch
import numpy as np
try:
    import nvdiffrast.torch as dr
    HAS_NVDIFFRAST = True
except ImportError:
    HAS_NVDIFFRAST = False
    # Create a simple mock for dr.texture if nvdiffrast is not available
    class MockDR:
        @staticmethod
        def texture(texture, uv, filter_mode="linear", boundary_mode="clamp"):
            """Simple bilinear interpolation mock."""
            # texture: (1, H, W, C)
            # uv: (1, N, 1, 2)
            # returns: (1, N, 1, C)
            _, H, W, C = texture.shape
            _, N, _, _ = uv.shape
            
            # Clamp UV coordinates
            u = uv[..., 0].clamp(0, 1)
            v = uv[..., 1].clamp(0, 1)
            
            # Convert to pixel coordinates
            x = u * (W - 1)
            y = v * (H - 1)
            
            # Get integer parts
            x0 = x.floor().long().clamp(0, W - 1)
            y0 = y.floor().long().clamp(0, H - 1)
            x1 = (x0 + 1).clamp(0, W - 1)
            y1 = (y0 + 1).clamp(0, H - 1)
            
            # Get fractional parts
            fx = (x - x0.float()).unsqueeze(-1)
            fy = (y - y0.float()).unsqueeze(-1)
            
            # Bilinear interpolation
            c00 = texture[0, y0.squeeze(-1), x0.squeeze(-1)]
            c01 = texture[0, y0.squeeze(-1), x1.squeeze(-1)]
            c10 = texture[0, y1.squeeze(-1), x0.squeeze(-1)]
            c11 = texture[0, y1.squeeze(-1), x1.squeeze(-1)]
            
            c0 = c00 * (1 - fx) + c01 * fx
            c1 = c10 * (1 - fx) + c11 * fx
            result = c0 * (1 - fy) + c1 * fy
            
            return result.unsqueeze(0).unsqueeze(2)
    
    dr = MockDR()

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from helper_functions import safe_normalize, sample_camera_rays, sample_camera_rays_unnormalize, reflection


FG_LUT = None


def set_FG_LUT(lut):
    """Set the global FG_LUT for testing."""
    global FG_LUT
    FG_LUT = lut


def get_specular_color_surfel(envmap, albedo, HWK, R, T, normal_map, render_alpha, 
                               scaling_modifier=1.0, refl_strength=None, roughness=None, 
                               pc=None, surf_depth=None, indirect_light=None):
    """
    Get specular color for surfel rendering.
    
    Args:
        envmap: Environment map callable
        albedo: Albedo map (H, W, 3)
        HWK: Tuple of (H, W, K) where K is intrinsic matrix
        R: Rotation matrix (camera to world)
        T: Translation vector
        normal_map: Normal map (H, W, 3)
        render_alpha: Render alpha (H, W, 1)
        scaling_modifier: Scaling modifier (default: 1.0)
        refl_strength: Reflection strength (H, W, 1)
        roughness: Roughness map (H, W, 1)
        pc: Point cloud object with optional ray_tracer
        surf_depth: Surface depth (1, H, W)
        indirect_light: Indirect lighting (H, W, 3)
    
    Returns:
        specular: Specular color (3, H, W)
        extra_dict: Extra information dictionary (or None)
    """
    global FG_LUT
    H, W, K = HWK
    rays_cam, rays_o = sample_camera_rays(HWK, R, T)
    w_o = -rays_cam
    rays_refl, NdotV = reflection(w_o, normal_map)
    rays_refl = safe_normalize(rays_refl)

    # TODO: Fill in the BSDF query and direct light computation
    # ****EMPTY****
    raise NotImplementedError("Please implement BSDF query and direct light computation")
    
    # visibility
    visibility = torch.ones_like(render_alpha)
    if pc.ray_tracer is not None and indirect_light is not None:
        # TODO: Fill in the ray tracing visibility computation
        # ****EMPTY****
        raise NotImplementedError("Please implement ray tracing visibility")
    else:
        # TODO: Fill in the simple visibility (no ray tracing)
        # ****EMPTY****
        raise NotImplementedError("Please implement simple visibility")
    
    # TODO: Fill in the specular color computation
    # Compute specular color
    # ****EMPTY****
    raise NotImplementedError("Please implement specular color computation")
    

    if indirect_light is not None:
        extra_dict = {
            "visibility": visibility.permute(2,0,1),
            "indirect_light": indirect_light.permute(2,0,1),
            "direct_light": direct_light.permute(2,0,1),
            "indirect_color": indirect_color.permute(2,0,1)
        } 
    else:
        extra_dict = None
        
    return specular.permute(2,0,1), extra_dict
