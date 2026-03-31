"""
Reference Implementation for compute_hom
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def patch_offsets(h_patch_size, device):
    """Generate patch offsets for sampling."""
    offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
    return torch.stack(torch.meshgrid(offsets, offsets)[::-1], dim=-1).view(1, -1, 2)


def patch_warp(H, uv):
    """Warp patches using homography."""
    B, P = uv.shape[:2]
    H = H.view(B, 3, 3)
    ones = torch.ones((B, P, 1), device=uv.device)
    homo_uv = torch.cat((uv, ones), dim=-1)

    grid_tmp = torch.einsum("bik,bpk->bpi", H, homo_uv)
    grid_tmp = grid_tmp.reshape(B, P, 3)
    grid = grid_tmp[..., :2] / (grid_tmp[..., 2:] + 1e-10)
    return grid


def lncc(ref, nea):
    """Compute local normalized cross-correlation."""
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 1)
    return ncc, mask


def compute_hom(depth, normal, points, view_ref, view_src, patch_size=3, patch_offset=None):
    """
    Compute homography-based photometric consistency.
    
    Args:
        depth: Depth map [1, H, W]
        normal: Normal map [3, H, W]
        points: 3D points [N, 3]
        view_ref: Reference view object
        view_src: Source view object
        patch_size: Half size of the patch (default: 3)
        patch_offset: Optional patch offsets [B, P, HW, 2]
    
    Returns:
        ncc: Normalized cross-correlation [N]
        mask: Valid mask [N]
        ori_pixels_patch: Original pixel patch coordinates [N, total_patch_size, 2]
    """
    ## sample mask
    H, W = depth.squeeze().shape
    ix, iy = torch.meshgrid(
        torch.arange(W), torch.arange(H), indexing='xy')
    pixels = torch.stack([ix, iy], dim=-1).float().to(depth.device)
    
    ## sample ref frame patch
    pixels = pixels.reshape(-1, 2)
    offsets = patch_offsets(patch_size, pixels.device)
    total_patch_size = (patch_size * 2 + 1) ** 2
    ori_pixels_patch = pixels.reshape(-1, 1, 2) + offsets.float()  # n total_patch_size 2
    ref_to_neareast_r = view_src.world_view_transform[:3, :3].transpose(-1, -2) @ view_ref.world_view_transform[:3, :3]
    # Rs * Rr^t
    ref_to_neareast_t = -ref_to_neareast_r @ view_ref.world_view_transform[3, :3] + view_src.world_view_transform[3, :3]
    # -Rs * Rr^t * Tr + Ts = -Rs(Rr^t * Tr - Rs^t * Ts)
    if patch_offset is not None:
        # patch_offset b p hw 2
        patch_offset = patch_offset.squeeze().permute(1, 0, 2)
        ori_pixels_patch = ori_pixels_patch + patch_offset
    ref_gt_image_gray = view_ref.original_image.to(depth.device)
    gt_image_gray = (0.299 * ref_gt_image_gray[0, :, :] + 0.587 * ref_gt_image_gray[1, :, :] + 0.114 * ref_gt_image_gray[2, :, :])[None]
    H, W = gt_image_gray.squeeze().shape
    pixels_patch = ori_pixels_patch.clone()
    pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
    pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
    ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
    ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)  # n p
        
    ## compute Homography
    ref_local_n = F.normalize(normal, p=2, dim=-1).permute(1, 2, 0)
    # ref_local_n = torch.eye(3)[None, :, :].repeat(H*W, 1, 1)[:,2,:]#n 3 [0 0 1]
    ref_local_n = ref_local_n.reshape(-1, 3)  # n 3
    ref_cam_points = torch.matmul(view_ref.w2c[None], torch.cat([points, torch.ones_like(points[:, 0, None])], dim=-1)[:, :, None])[:, :3, 0]
    # 1 4 4. n 4 1 -> n 3
    ref_local_d = -1 * (ref_cam_points * ref_local_n).sum(dim=-1)  # n 
    
    H_ref_to_neareast = ref_to_neareast_r[None] - \
        torch.matmul(ref_to_neareast_t[None, :, None].expand(ref_local_d.shape[0], 3, 1), 
                    ref_local_n[:, :, None].expand(ref_local_d.shape[0], 3, 1).permute(0, 2, 1)) / ref_local_d[..., None, None]
    # Rs * Rr^t - Rs(Rs^t * Ts - Rr^t * Tr) * n / d.   n 3 3 
    H_ref_to_neareast = torch.matmul(view_src.K.float()[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
    H_ref_to_neareast = H_ref_to_neareast @ torch.inverse(view_ref.K.float())
    
    ## compute neareast frame patch
    grid = patch_warp(H_ref_to_neareast.reshape(-1, 3, 3), ori_pixels_patch)
    grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
    grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
    view_src_ori_gt_image = view_src.original_image.to(depth.device)
    nearest_image_gray = (0.299 * view_src_ori_gt_image[0, :, :] + 0.587 * view_src_ori_gt_image[1, :, :] + 0.114 * view_src_ori_gt_image[2, :, :])[None]
    sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
    sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
    
    ## compute loss
    ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
    mask = ncc_mask.reshape(-1)
    ncc = ncc.reshape(-1)
    # ncc = ncc[mask].squeeze()
    return ncc, mask, ori_pixels_patch  # n 49 2


# Mock View class for testing
class MockView:
    """Mock view class for testing compute_hom function.
    
    Supports multiple naming conventions for compatibility with different LLM implementations:
    - Standard: K, world_view_transform, w2c, original_image
    - Alternative: intrinsics, extrinsics, R, t, img, image, W2C, C2W
    """
    
    def __init__(self, image_height, image_width, device='cpu'):
        self.image_height = image_height
        self.image_width = image_width
        self.device = device
        
        # Generate random camera parameters
        self.K = torch.tensor([
            [500.0, 0.0, image_width / 2],
            [0.0, 500.0, image_height / 2],
            [0.0, 0.0, 1.0]
        ], device=device)
        
        # Generate random world_view_transform (4x4)
        # Special format: R in [:3,:3], t in [3,:3], last row is [0,0,0,1]
        angle = torch.rand(3, device=device) * 0.3  # Smaller angles for stability
        R_matrix = self._euler_to_rotation_matrix(angle)
        t_vector = torch.randn(3, device=device) * 0.5  # Smaller translation
        
        self.world_view_transform = torch.eye(4, device=device)
        self.world_view_transform[:3, :3] = R_matrix
        self.world_view_transform[3, :3] = t_vector  # Note: special indexing [3, :3] for translation
        
        # w2c is the world to camera transform (standard 4x4 format)
        # Standard format: R in [:3,:3], t in [:3,3]
        self.w2c = torch.eye(4, device=device)
        self.w2c[:3, :3] = R_matrix
        self.w2c[:3, 3] = t_vector
        
        # Generate random image
        self.original_image = torch.rand(3, image_height, image_width, device=device)
        
        # ============ Add attribute aliases for compatibility ============
        
        # For GPT-style naming
        self.intrinsics = self.K
        self.extrinsics = self.world_view_transform
        
        # For Doubao-style naming
        self.R = R_matrix
        self.t = t_vector
        self.img = self.original_image
        
        # For Gemini-style naming
        self.image = self.original_image
        self.W2C = self.w2c
        self.C2W = torch.inverse(self.w2c)
        
        # Store for project method
        self._R = R_matrix
        self._t = t_vector
    
    def _euler_to_rotation_matrix(self, angles):
        """Convert euler angles to rotation matrix."""
        roll, pitch, yaw = angles
        
        R_x = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(roll), -torch.sin(roll)],
            [0, torch.sin(roll), torch.cos(roll)]
        ], device=angles.device)
        
        R_y = torch.tensor([
            [torch.cos(pitch), 0, torch.sin(pitch)],
            [0, 1, 0],
            [-torch.sin(pitch), 0, torch.cos(pitch)]
        ], device=angles.device)
        
        R_z = torch.tensor([
            [torch.cos(yaw), -torch.sin(yaw), 0],
            [torch.sin(yaw), torch.cos(yaw), 0],
            [0, 0, 1]
        ], device=angles.device)
        
        R = R_z @ R_y @ R_x
        return R
    
    def project(self, points):
        """Project 3D points to 2D image coordinates.
        
        Args:
            points: 3D points [N, 3]
        
        Returns:
            pixels: 2D coordinates [N, 2] in pixel space
        """
        # Transform points to camera space
        points_homo = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=-1)  # [N, 4]
        points_cam = (self.w2c @ points_homo.T).T  # [N, 4]
        points_cam_3d = points_cam[:, :3]  # [N, 3]
        
        # Project to image plane
        points_proj = (self.K @ points_cam_3d.T).T  # [N, 3]
        pixels = points_proj[:, :2] / (points_proj[:, 2:3] + 1e-8)  # [N, 2]
        
        return pixels

