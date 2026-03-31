"""
Helper functions for get_specular_color_surfel testing.
These are simplified versions or mocks of the original functions.
"""

import torch
import numpy as np


def safe_normalize(x, eps=1e-6):
    """Normalize a tensor along the last dimension."""
    norm = torch.norm(x, dim=-1, keepdim=True)
    return x / (norm + eps)


def sample_camera_rays(HWK, R, T):
    """Sample camera rays for testing."""
    H, W, K = HWK
    R = R.T  # NOTE!!! the R rot matrix is transposed save in 3DGS
    
    K_mat = K.astype(np.float32)
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                    np.arange(H, dtype=np.float32),
                    indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K_mat).T)
    pixel_camera = torch.tensor(pixel_camera, device=R.device)

    rays_o = (-R.T @ T.unsqueeze(-1)).flatten()
    pixel_world = (pixel_camera - T[None, None]).reshape(-1, 3) @ R
    rays_d = pixel_world - rays_o[None]
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
    rays_d = rays_d.reshape(H, W, 3)
    return rays_d, rays_o


def sample_camera_rays_unnormalize(HWK, R, T):
    """Sample camera rays without normalization."""
    H, W, K = HWK
    R = R.T  # NOTE!!! the R rot matrix is transposed save in 3DGS
    
    K_mat = K.astype(np.float32)
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                    np.arange(H, dtype=np.float32),
                    indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K_mat).T)
    pixel_camera = torch.tensor(pixel_camera, device=R.device)

    rays_o = (-R.T @ T.unsqueeze(-1)).flatten()
    pixel_world = (pixel_camera - T[None, None]).reshape(-1, 3) @ R
    rays_d = pixel_world - rays_o[None]
    rays_d = rays_d.reshape(H, W, 3)
    return rays_d, rays_o


def reflection(w_o, normal):
    """Compute reflection direction."""
    NdotV = torch.sum(w_o*normal, dim=-1, keepdim=True)
    w_k = 2*normal*NdotV - w_o
    return w_k, NdotV


class MockEnvmap:
    """Mock environment map for testing.
    
    Note: This is a callable object, not a tensor.
    Usage: envmap(rays, roughness=...) to query the environment map.
    Do NOT use: envmap[...] or envmap.shape - these will fail!
    """
    
    def __init__(self, H=16, W=32, device='cpu'):
        self.H = H
        self.W = W
        self.device = device
        # Create a simple environment map data (for reference only)
        self.data = torch.rand(H, W, 3, device=device)
    
    def __call__(self, rays, roughness=None, mode=None):
        """Query the environment map.
        
        Args:
            rays: Ray directions, shape (..., 3)
            roughness: Optional roughness values
            mode: Optional query mode
            
        Returns:
            Environment lighting, shape (..., 3)
        """
        # Simple mock: return constant color with some variation based on ray direction
        batch_shape = rays.shape[:-1]
        result = torch.ones(*batch_shape, 3, device=self.device) * 0.5
        # Add some variation based on ray direction
        result = result + rays.abs() * 0.3
        return result.clamp(0, 1)
    
    def __getattr__(self, name):
        """Provide friendly error messages for common mistakes."""
        if name in ['shape', 'size', 'dim']:
            raise AttributeError(
                f"MockEnvmap is a callable object, not a tensor.\n"
                f"Use: envmap(rays, roughness=...) to query the environment map.\n"
                f"Do NOT use: envmap.{name} or envmap[...]"
            )
        raise AttributeError(f"'MockEnvmap' object has no attribute '{name}'")


class MockRayTracer:
    """Mock ray tracer for testing visibility."""
    
    def __init__(self, device='cpu'):
        self.device = device
        # Use fixed seed for deterministic results
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(42)
    
    def trace(self, origins, directions):
        """Mock ray tracing that returns dummy intersections."""
        N = origins.shape[0]
        # Return dummy values: intersections, normals, depths
        # For deterministic testing: use a pattern based on position
        # depths > 10 means no intersection (background)
        depths = torch.ones(N, device=self.device) * 15.0  # Default: no intersection
        return None, None, depths
    
    def trace_visibility(self, origins, directions, surf_depth):
        """Mock ray tracing for visibility (used by some LLM implementations)."""
        # Return a visibility mask (True = visible/no occlusion)
        # For testing, return all visible
        if hasattr(origins, 'shape'):
            H, W, _ = origins.shape if len(origins.shape) == 3 else (origins.shape[0], 1, origins.shape[1])
            return torch.ones(H, W, device=self.device, dtype=torch.bool)
        else:
            return torch.ones_like(surf_depth[0], dtype=torch.bool)


class MockPC:
    """Mock point cloud with ray tracer."""
    
    def __init__(self, has_ray_tracer=False, device='cpu'):
        self.device = device
        if has_ray_tracer:
            self.ray_tracer = MockRayTracer(device=device)
        else:
            self.ray_tracer = None


def create_mock_FG_LUT(device='cpu'):
    """Create a mock BSDF lookup table."""
    # Shape: (1, 256, 256, 2)
    return torch.rand(1, 256, 256, 2, device=device) * 0.5 + 0.5

