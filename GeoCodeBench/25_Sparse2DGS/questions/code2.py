import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math

def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

def get_image_coor_from_world_points2(points, view, mode=None, scale=1):
    #points n 3 
    ****EMPTY****
    if mode == "scale":
        ****EMPTY****
    else:
        ****EMPTY****
    return coor_2d, coor_mask, z.squeeze()


def patch_warp(H, uv):
    B, P = uv.shape[:2]
    H = H.view(B, 3, 3)
    ones = torch.ones((B,P,1), device=uv.device)
    homo_uv = torch.cat((uv, ones), dim=-1)

    grid_tmp = torch.einsum("bik,bpk->bpi", H, homo_uv)
    grid_tmp = grid_tmp.reshape(B, P, 3)
    grid = grid_tmp[..., :2] / (grid_tmp[..., 2:] + 1e-10)
    return grid

def lncc(ref, nea):
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
    
def patch_offsets(h_patch_size, device):
    offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
    return torch.stack(torch.meshgrid(offsets, offsets)[::-1], dim=-1).view(1, -1, 2)

def compute_hom_v2(depth, normal, points, view_ref, view_src, patch_size=3, patch_offset=None):
    ## sample mask
    H, W = depth.squeeze().shape
    ix, iy = torch.meshgrid(
        torch.arange(W), torch.arange(H), indexing='xy')
    pixels = torch.stack([ix, iy], dim=-1).float().to(depth.device)
    
    ## sample ref frame patch
    pixels = pixels.reshape(-1,2)
    offsets = patch_offsets(patch_size, pixels.device)
    total_patch_size = (patch_size * 2 + 1) ** 2
    ori_pixels_patch = pixels.reshape(-1, 1, 2) + offsets.float()#n total_patch_size 2
    ref_to_neareast_r = view_src.world_view_transform[:3,:3].transpose(-1,-2) @ view_ref.world_view_transform[:3,:3]
    #Rs * Rr^t
    ref_to_neareast_t = -ref_to_neareast_r @ view_ref.world_view_transform[3,:3] + view_src.world_view_transform[3,:3]
    #-Rs * Rr^t * Tr + Ts = -Rs(Rr^t * Tr - Rs^t * Ts)
    if patch_offset is not None:
        #patch_offset b p hw 2
        patch_offset = patch_offset.squeeze().permute(1, 0, 2)
        ori_pixels_patch = ori_pixels_patch + patch_offset
    ref_gt_image_gray = view_ref.original_image.cuda()
    gt_image_gray = (0.299 * ref_gt_image_gray[0,:,:] + 0.587 * ref_gt_image_gray[1,:,:] + 0.114 * ref_gt_image_gray[2,:,:])[None]
    H, W = gt_image_gray.squeeze().shape
    pixels_patch = ori_pixels_patch.clone()
    pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
    pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
    ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
    ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)#n p
    coor_2d, coor_mask, _ = get_image_coor_from_world_points2(points, view_ref, mode="scale")#2 n
    coor_2d = coor_2d.permute(1, 0)[None, None]#1 1 n 2

    sample_ref_gray_val = F.grid_sample(ref_gray_val.reshape(H, W, -1).permute(2, 0, 1)[None], #1 p h w
                                        coor_2d, 
                                        mode='nearest', 
                                        align_corners=True) #1 p 1 n
    sample_ref_gray_val = sample_ref_gray_val.squeeze().permute(1, 0)#n p

    ori_pixels_patch_sample = F.grid_sample(ori_pixels_patch.reshape(H, W, -1).permute(2, 0, 1)[None], #1 p h w
                                        coor_2d, 
                                        mode='nearest', 
                                        align_corners=True) #1 p 1 n
    ori_pixels_patch_sample = ori_pixels_patch_sample.squeeze().permute(1, 0).reshape(-1, total_patch_size, 2)
    ## compute Homography
    ref_local_n = F.normalize(normal, p=2, dim=-1).permute(1,2,0)
    #ref_local_n = torch.eye(3)[None, :, :].repeat(H*W, 1, 1)[:,2,:]#n 3 [0 0 1]
    ref_local_n = ref_local_n.reshape(-1,3)#n 3
    ref_cam_points = torch.matmul(view_ref.w2c[None], torch.cat([points, torch.ones_like(points[:, 0, None])] ,dim=-1)[:, :, None])[:, :3, 0]
    #1 4 4. n 4 1 -> n 3
    ref_local_d = -1 * (ref_cam_points * ref_local_n).sum(dim=-1)#n 
    
    H_ref_to_neareast = ref_to_neareast_r[None] - \
        torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                    ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
    #Rs * Rr^t - Rs(Rs^t * Ts - Rr^t * Tr) * n / d.   n 3 3 
    H_ref_to_neareast = torch.matmul(view_src.K.float()[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
    H_ref_to_neareast = H_ref_to_neareast @ torch.inverse(view_ref.K.float())
    
    ## compute neareast frame patch
    grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch_sample)
    grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
    grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
    view_src_ori_gt_image = view_src.original_image.cuda()
    nearest_image_gray = (0.299 * view_src_ori_gt_image[0,:,:] + 0.587 * view_src_ori_gt_image[1,:,:] + 0.114 * view_src_ori_gt_image[2,:,:])[None]
    sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
    sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
    
    ## compute loss
    ncc, ncc_mask = lncc(sample_ref_gray_val, sampled_gray_val)
    mask = ncc_mask.reshape(-1)
    ncc = ncc.reshape(-1)
    #ncc = ncc[mask].squeeze()
    return ncc, mask#, ori_pixels_patch #n 49 2

def compute_hom(depth, normal, points, view_ref, view_src, patch_size=3, patch_offset=None):
    #with torch.no_grad():
    ## sample mask
    H, W = depth.squeeze().shape
    ix, iy = torch.meshgrid(
        torch.arange(W), torch.arange(H), indexing='xy')
    pixels = torch.stack([ix, iy], dim=-1).float().to(depth.device)
    
    ## sample ref frame patch
    pixels = pixels.reshape(-1,2)
    offsets = patch_offsets(patch_size, pixels.device)
    total_patch_size = (patch_size * 2 + 1) ** 2
    ori_pixels_patch = pixels.reshape(-1, 1, 2) + offsets.float()#n total_patch_size 2
    ref_to_neareast_r = view_src.world_view_transform[:3,:3].transpose(-1,-2) @ view_ref.world_view_transform[:3,:3]
    #Rs * Rr^t
    ref_to_neareast_t = -ref_to_neareast_r @ view_ref.world_view_transform[3,:3] + view_src.world_view_transform[3,:3]
    #-Rs * Rr^t * Tr + Ts = -Rs(Rr^t * Tr - Rs^t * Ts)
    if patch_offset is not None:
        #patch_offset b p hw 2
        patch_offset = patch_offset.squeeze().permute(1, 0, 2)
        ori_pixels_patch = ori_pixels_patch + patch_offset
    ref_gt_image_gray = view_ref.original_image.cuda()
    gt_image_gray = (0.299 * ref_gt_image_gray[0,:,:] + 0.587 * ref_gt_image_gray[1,:,:] + 0.114 * ref_gt_image_gray[2,:,:])[None]
    H, W = gt_image_gray.squeeze().shape
    pixels_patch = ori_pixels_patch.clone()
    pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
    pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
    ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
    ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)#n p
        
    ## compute Homography
    ref_local_n = F.normalize(normal, p=2, dim=-1).permute(1,2,0)
    #ref_local_n = torch.eye(3)[None, :, :].repeat(H*W, 1, 1)[:,2,:]#n 3 [0 0 1]
    ref_local_n = ref_local_n.reshape(-1,3)#n 3
    ref_cam_points = torch.matmul(view_ref.w2c[None], torch.cat([points, torch.ones_like(points[:, 0, None])] ,dim=-1)[:, :, None])[:, :3, 0]
    #1 4 4. n 4 1 -> n 3
    ref_local_d = -1 * (ref_cam_points * ref_local_n).sum(dim=-1)#n 
    
    H_ref_to_neareast = ref_to_neareast_r[None] - \
        torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                    ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
    #Rs * Rr^t - Rs(Rs^t * Ts - Rr^t * Tr) * n / d.   n 3 3 
    H_ref_to_neareast = torch.matmul(view_src.K.float()[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
    H_ref_to_neareast = H_ref_to_neareast @ torch.inverse(view_ref.K.float())
    
    ## compute neareast frame patch
    grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
    grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
    grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
    view_src_ori_gt_image = view_src.original_image.cuda()
    nearest_image_gray = (0.299 * view_src_ori_gt_image[0,:,:] + 0.587 * view_src_ori_gt_image[1,:,:] + 0.114 * view_src_ori_gt_image[2,:,:])[None]
    sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
    sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
    
    ## compute loss
    ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
    mask = ncc_mask.reshape(-1)
    ncc = ncc.reshape(-1)
    #ncc = ncc[mask].squeeze()
    return ncc, mask, ori_pixels_patch #n 49 2