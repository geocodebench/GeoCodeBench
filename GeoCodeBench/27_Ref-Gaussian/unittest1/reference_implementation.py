"""
Reference Implementation for get_specular_color_surfel
This serves as the ground truth for testing LLM-generated implementations.
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
            original_shape = uv.shape
            
            # Flatten to (N, 2)
            uv_flat = uv.reshape(-1, 2)
            N = uv_flat.shape[0]
            
            # Clamp UV coordinates
            u = uv_flat[:, 0].clamp(0, 1)
            v = uv_flat[:, 1].clamp(0, 1)
            
            # Convert to pixel coordinates
            x = u * (W - 1)
            y = v * (H - 1)
            
            # Get integer parts
            x0 = x.floor().long().clamp(0, W - 1)
            y0 = y.floor().long().clamp(0, H - 1)
            x1 = (x0 + 1).clamp(0, W - 1)
            y1 = (y0 + 1).clamp(0, H - 1)
            
            # Get fractional parts
            fx = (x - x0.float()).unsqueeze(-1)  # (N, 1)
            fy = (y - y0.float()).unsqueeze(-1)  # (N, 1)
            
            # Bilinear interpolation - sample at 4 corners
            c00 = texture[0, y0, x0]  # (N, C)
            c01 = texture[0, y0, x1]  # (N, C)
            c10 = texture[0, y1, x0]  # (N, C)
            c11 = texture[0, y1, x1]  # (N, C)
            
            # Interpolate
            c0 = c00 * (1 - fx) + c01 * fx
            c1 = c10 * (1 - fx) + c11 * fx
            result = c0 * (1 - fy) + c1 * fy  # (N, C)
            
            # Reshape back to original batch structure
            return result.reshape(original_shape[0], original_shape[1], original_shape[2], C)
    
    dr = MockDR()

from helper_functions import safe_normalize, sample_camera_rays, sample_camera_rays_unnormalize, reflection


FG_LUT = None


def set_FG_LUT(lut):
    """Set the global FG_LUT for testing."""
    global FG_LUT
    FG_LUT = lut


def get_specular_color_surfel(envmap, albedo, HWK, R, T, normal_map, render_alpha, 
                               scaling_modifier=1.0, refl_strength=None, roughness=None, 
                               pc=None, surf_depth=None, indirect_light=None):
    """Reference implementation from refl_utils.py lines 104-154."""
    global FG_LUT
    H, W, K = HWK
    rays_cam, rays_o = sample_camera_rays(HWK, R, T)
    w_o = -rays_cam
    rays_refl, NdotV = reflection(w_o, normal_map)
    rays_refl = safe_normalize(rays_refl)

    # Query BSDF
    fg_uv = torch.cat([NdotV, roughness], -1).clamp(0, 1) 
    fg = dr.texture(FG_LUT, fg_uv.reshape(1, -1, 1, 2).contiguous(), filter_mode="linear", boundary_mode="clamp").reshape(1, H, W, 2) 
    # Compute direct light
    direct_light = envmap(rays_refl, roughness=roughness)
    specular_weight = ((0.04 * (1 - refl_strength) + albedo * refl_strength) * fg[0][..., 0:1] + fg[0][..., 1:2]) 
    
    # visibility
    visibility = torch.ones_like(render_alpha)
    if pc.ray_tracer is not None and indirect_light is not None:
        mask = (render_alpha>0)[..., 0]
        if mask.any():
            rays_cam, rays_o = sample_camera_rays_unnormalize(HWK, R, T)
            w_o = safe_normalize(-rays_cam)
            rays_refl, _ = reflection(w_o, normal_map)
            rays_refl = safe_normalize(rays_refl)
            intersections = rays_o + surf_depth.permute(1, 2, 0) * rays_cam
            _, _, depth = pc.ray_tracer.trace(intersections[mask], rays_refl[mask])
            # Create a new visibility tensor to avoid in-place operation issues
            vis_values = (depth >= 10).float().unsqueeze(-1)
            visibility = visibility.clone()
            visibility[mask] = vis_values
    
        # indirect light
        specular_light = direct_light * visibility + (1 - visibility) * indirect_light
        indirect_color = (1 - visibility) * indirect_light * render_alpha * specular_weight
    else:
        specular_light = direct_light
    
    # Compute specular color
    specular_raw = specular_light * render_alpha
    specular = specular_raw * specular_weight
    

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

