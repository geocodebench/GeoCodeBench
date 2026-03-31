"""
Reference implementation for get_full_color_volume_indirect function.
This is the correct implementation used as ground truth for testing.
"""

import torch
import numpy as np
import time

# Global variables
FG_LUT = None

def init_fg_lut(device='cpu'):
    """Initialize FG_LUT with dummy data for testing."""
    global FG_LUT
    if FG_LUT is None:
        # Create a dummy FG_LUT for testing (normally loaded from file)
        # Use fixed seed to ensure consistency across modules
        old_state = torch.get_rng_state()
        torch.manual_seed(42)
        FG_LUT = torch.rand(1, 256, 256, 2, device=device)
        torch.set_rng_state(old_state)  # Restore original state


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


class MockEnvmap:
    """Mock environment map for testing."""
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def __call__(self, directions, mode=None, roughness=None):
        """Sample environment map."""
        # Simple mock: return normalized directions as colors
        if mode == "diffuse":
            return torch.abs(directions) * 0.5
        else:
            # For specular, incorporate roughness if provided
            result = torch.abs(directions) * 0.8
            if roughness is not None:
                result = result * (1.0 - roughness * 0.3)
            return result


class MockRayTracer:
    """Mock ray tracer for testing."""
    
    def trace(self, origins, directions):
        """Mock ray tracing."""
        # Return mock intersection, normal, and depth
        batch_size = origins.shape[0]
        device = origins.device
        
        # Use deterministic depth based on ray position to ensure consistency
        # Use a hash-like function based on origins to get pseudo-random but deterministic depths
        depth_seed = (origins[:, 0] * 1000 + origins[:, 1] * 100 + origins[:, 2] * 10).abs()
        depth_seed = (depth_seed * 12345.6789) % 20  # Deterministic pseudo-random in [0, 20]
        depths = depth_seed
        
        intersections = origins + directions * depths.unsqueeze(-1)
        normals = torch.nn.functional.normalize(directions, dim=-1)
        
        return intersections, normals, depths


class MockPC:
    """Mock point cloud object."""
    
    def __init__(self, use_ray_tracer=True, device='cpu'):
        self.ray_tracer = MockRayTracer() if use_ray_tracer else None


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
    # texture shape: [1, H, W, C], we need to index as [B, v, u, :]
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
    Reference implementation of get_full_color_volume_indirect function.
    
    Args:
        envmap: Environment map function
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
        pc: Point cloud object with optional ray_tracer
        indirect_light: Indirect lighting [N, 3]
    
    Returns:
        diffuse: Diffuse component [N, 3]
        specular: Specular component [N, 3]
        extra_dict: Dictionary with visibility and direct_light
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
        mask = (render_alpha > 0).squeeze()
        intersections = xyz
        _, _, depth = pc.ray_tracer.trace(intersections[mask], rays_refl[mask])
        visibility[mask] = (depth >= 10).unsqueeze(1).float()

    # Query BSDF
    fg_uv = torch.cat([NdotV, roughness], -1).clamp(0, 1) 
    fg_uv = fg_uv.unsqueeze(0).unsqueeze(2)  # [1, N, 1, 2]
    fg = texture_sample(FG_LUT, fg_uv).squeeze(2).squeeze(0)  # [N, 2]
    # Compute diffuse
    diffuse = envmap(normal_map, mode="diffuse") * (1 - refl_strength) * albedo
    # Compute specular
    direct_light = envmap(rays_refl, roughness=roughness) 
    specular_weight = ((0.04 * (1 - refl_strength) + albedo * refl_strength) * fg[..., 0:1] + fg[..., 1:2]) 
    specular_light = direct_light * visibility + (1 - visibility) * indirect_light
    specular = specular_light * specular_weight

    extra_dict = {
        "visibility": visibility,
        "direct_light": direct_light,
    }

    return diffuse, specular, extra_dict

