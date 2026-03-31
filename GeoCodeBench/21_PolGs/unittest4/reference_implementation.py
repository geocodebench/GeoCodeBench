"""
Reference Implementation for camera ray functions
This serves as the ground truth for testing LLM-generated implementations.
"""

import numpy as np
import torch

pixel_camera = None

def sample_camera_rays(HWK, R, T):
    """Sample camera rays for each pixel.
    
    Args:
        HWK: Tuple of (H, W, K) where H is height, W is width, K is intrinsic matrix
        R: Rotation matrix (world to camera), transposed in 3DGS format
        T: Translation vector (world to camera)
    
    Returns:
        rays_d: Ray directions in world space, shape (H, W, 3)
    """
    H, W, K = HWK
    R = R.T  # NOTE!!! the R rot matrix is transposed save in 3DGS
    global pixel_camera
    if pixel_camera is None or pixel_camera.shape[0] != H:
        K = K.astype(np.float32)
        i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32),
                        indexing='xy')
        xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
        pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
        pixel_camera = torch.tensor(pixel_camera)  # Changed from .cuda() to cpu
    rays_o = (-R.T @ T.unsqueeze(-1)).flatten()
    pixel_world = (pixel_camera - T[None, None]).reshape(-1, 3) @ R
    rays_d = pixel_world - rays_o[None]
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
    rays_d = rays_d.reshape(H, W, 3)
    return rays_d


def reflection(rayd, normal):
    """Compute reflection vector.
    
    Args:
        rayd: Ray direction, shape (..., 3)
        normal: Surface normal, shape (..., 3)
    
    Returns:
        refl: Reflected ray direction, shape (..., 3)
    """
    refl = rayd - 2 * normal * torch.sum(rayd * normal, dim=-1, keepdim=True)
    return refl


def sample_cubemap_color(rays_d, env_map):
    """Sample color from environment map (cubemap).
    
    Args:
        rays_d: Ray directions, shape (H, W, 3)
        env_map: Environment map function/network
    
    Returns:
        outcolor: Sampled colors, shape (3, H, W)
    """
    H, W = rays_d.shape[:2]
    outcolor = torch.sigmoid(env_map(rays_d.reshape(-1, 3)))
    outcolor = outcolor.reshape(H, W, 3).permute(2, 0, 1)
    return outcolor


def get_refl_color(envmap, HWK, R, T, normal_map):
    """Get reflection color from environment map.
    
    Args:
        envmap: Environment map function/network
        HWK: Tuple of (H, W, K) where H is height, W is width, K is intrinsic matrix
        R: Rotation matrix (world to camera)
        T: Translation vector (world to camera)
        normal_map: Normal map in world space, shape (H, W, 3)
    
    Returns:
        Reflected colors, shape (3, H, W)
    """
    rays_d = sample_camera_rays(HWK, R, T)
    rays_d = reflection(rays_d, normal_map)
    return sample_cubemap_color(rays_d, envmap)

