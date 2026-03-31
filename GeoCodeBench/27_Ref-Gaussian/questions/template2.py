
"""
Template for LLM Implementation
Copy this file and fill in the EMPTY section with LLM-generated code.
"""

import torch
import numpy as np

# Global variables
FG_LUT = None

def init_fg_lut(device='cpu'):
    """Initialize FG_LUT with dummy data for testing."""
    global FG_LUT
    if FG_LUT is None:
        # Create a dummy FG_LUT for testing (normally loaded from file)
        FG_LUT = torch.rand(1, 256, 256, 2, device=device)


def safe_normalize(x, eps=1e-6):
    """Safely normalize a vector."""
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def reflection(w_o, normal):
    """Compute reflection vector and NdotV."""
    NdotV = torch.sum(w_o * normal, dim=-1, keepdim=True)
    w_k = 2 * normal * NdotV - w_o
    return w_k, NdotV


def sample_camera_rays(HWK, R, T):
    """Sample camera rays."""
    H, W, K = HWK
    R = R.T  # NOTE: the R rot matrix is transposed save in 3DGS
    
    K_np = K.cpu().numpy().astype(np.float32) if isinstance(K, torch.Tensor) else K.astype(np.float32)
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K_np).T)
    pixel_camera = torch.tensor(pixel_camera, device=R.device)
    
    rays_o = (-R.T @ T.unsqueeze(-1)).flatten()
    pixel_world = (pixel_camera - T[None, None]).reshape(-1, 3) @ R
    rays_d = pixel_world - rays_o[None]
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
    rays_d = rays_d.reshape(H, W, 3)
    return rays_d, rays_o


def texture_sample(texture, uv):
    """
    Simple texture sampling using bilinear interpolation.
    texture: [1, H, W, C]
    uv: [1, N, 1, 2] where uv values are in [0, 1]
    returns: [1, N, 1, C]
    """
    B, N, _, _ = uv.shape
    _, H, W, C = texture.shape
    
    # Scale UV to texture coordinates
    u = uv[..., 0] * (W - 1)  # [1, N, 1]
    v = uv[..., 1] * (H - 1)  # [1, N, 1]
    
    # Get integer coordinates
    u0 = torch.floor(u).long().clamp(0, W - 1)
    u1 = torch.ceil(u).long().clamp(0, W - 1)
    v0 = torch.floor(v).long().clamp(0, H - 1)
    v1 = torch.ceil(v).long().clamp(0, H - 1)
    
    # Get fractional parts
    u_frac = u - u0.float()
    v_frac = v - v0.float()
    
    # Sample texture at four corners
    result = torch.zeros(B, N, 1, C, device=texture.device, dtype=texture.dtype)
    
    for b in range(B):
        for n in range(N):
            for k in range(1):  # third dimension is always 1
                v0_idx = v0[b, n, k]
                v1_idx = v1[b, n, k]
                u0_idx = u0[b, n, k]
                u1_idx = u1[b, n, k]
                
                uf = u_frac[b, n, k]
                vf = v_frac[b, n, k]
                
                # Bilinear interpolation
                c00 = texture[b, v0_idx, u0_idx, :]
                c01 = texture[b, v0_idx, u1_idx, :]
                c10 = texture[b, v1_idx, u0_idx, :]
                c11 = texture[b, v1_idx, u1_idx, :]
                
                c0 = c00 * (1 - uf) + c01 * uf
                c1 = c10 * (1 - uf) + c11 * uf
                c = c0 * (1 - vf) + c1 * vf
                
                result[b, n, k, :] = c
    
    return result


def get_full_color_volume_indirect(envmap, xyz, albedo, HWK, R, T, normal_map, render_alpha, 
                                   scaling_modifier=1.0, refl_strength=None, roughness=None, 
                                   pc=None, indirect_light=None):
    """
    Compute full color volume with indirect lighting.
    
    Args:
        envmap: Environment map function that takes (directions, mode=None, roughness=None) and returns [N, 3] colors
        xyz: 3D positions [N, 3]
        albedo: Albedo values [N, 3]
        HWK: Tuple of (Height, Width, K_matrix)
        R: Rotation matrix [3, 3]
        T: Translation vector [3]
        normal_map: Normal vectors [N, 3]
        render_alpha: Alpha values [N, 1]
        scaling_modifier: Scaling factor (default: 1.0)
        refl_strength: Reflection strength [N, 1]
        roughness: Roughness values [N, 1]
        pc: Point cloud object with optional ray_tracer attribute
        indirect_light: Indirect lighting [N, 3]
    
    Returns:
        diffuse: Diffuse component [N, 3]
        specular: Specular component [N, 3]
        extra_dict: Dictionary with keys 'visibility' and 'direct_light'
    """
    global FG_LUT
    _, rays_o = sample_camera_rays(HWK, R, T)
    N, _ = normal_map.shape
    rays_o = rays_o.expand(N, -1)
    w_o = safe_normalize(rays_o - xyz)
    rays_refl, NdotV = reflection(w_o, normal_map)
    rays_refl = safe_normalize(rays_refl)

    # visibility
    visibility = torch.ones_like(render_alpha)
    if pc.ray_tracer is not None:
        mask = (render_alpha>0).squeeze()
        intersections = xyz
        _, _, depth = pc.ray_tracer.trace(intersections[mask], rays_refl[mask])
        visibility[mask] = (depth >= 10).unsqueeze(1).float()

    # Query BSDF
    ####EMPTY####
    extra_dict = {
        "visibility": visibility,
        "direct_light": direct_light,
    }

    return diffuse, specular, extra_dict
