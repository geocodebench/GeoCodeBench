# Copyright (c) OpenMMLab. All rights reserved.
# Attention: This file is mainly modified based on the file with the same
# name in the original project. For more details, please refer to the
# origin project.
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import KDTree
#from pyflann import FLANN
from sklearn.neighbors import NearestNeighbors

rng = np.random.RandomState(234)


# helper functions for nerf ray rendering
def volume_sampling(sample_pts, features, aabb):
    B, C, D, W, H = features.shape
    assert B == 1
    aabb = torch.Tensor(aabb).to(sample_pts.device)
    N_rays, N_samples, coords = sample_pts.shape
    sample_pts = sample_pts.view(1, N_rays * N_samples, 1, 1,
                                 3).repeat(B, 1, 1, 1, 1)
    aabbSize = aabb[1] - aabb[0]
    invgridSize = 1.0 / aabbSize * 2
    norm_pts = (sample_pts - aabb[0]) * invgridSize - 1
    sample_features = F.grid_sample(
        features, norm_pts, align_corners=True, padding_mode='border')
    masks = ((norm_pts < 1) & (norm_pts > -1)).float().sum(dim=-1)
    masks = (masks.view(N_rays, N_samples) == 3)
    return sample_features.view(C, N_rays,
                                N_samples).permute(1, 2, 0).contiguous(), masks


def _compute_projection(img_meta):
    views = len(img_meta['lidar2img']['extrinsic'])
    intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic'][:4, :4])
    ratio = img_meta['ori_shape'][0] / img_meta['img_shape'][0]
    intrinsic[:2] /= ratio
    intrinsic = intrinsic.unsqueeze(0).view(1, 16).repeat(views, 1)
    img_size = torch.Tensor(img_meta['img_shape'][:2]).to(intrinsic.device)
    img_size = img_size.unsqueeze(0).repeat(views, 1)
    extrinsics = []
    for v in range(views):
        extrinsics.append(
            torch.Tensor(img_meta['lidar2img']['extrinsic'][v]).to(
                intrinsic.device))
    extrinsic = torch.stack(extrinsics).view(views, 16)
    train_cameras = torch.cat([img_size, intrinsic, extrinsic], dim=-1)
    return train_cameras.unsqueeze(0)


def compute_mask_points(feature, mask):
    weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)
    mean = torch.sum(feature * weight, dim=2, keepdim=True)
    var = torch.sum((feature - mean)**2, dim=2, keepdim=True)
    var = var / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)
    var = torch.exp(-var)
    return mean, var


def sample_pdf(bins, weights, N_samples, det=False):
    """Helper function used for sampling.

    Args:
        bins (tensor):Tensor of shape [N_rays, M+1], M is the number of bins
        weights (tensor):Tensor of shape [N_rays, M+1], M is the number of bins
        N_samples (int):Number of samples along each ray
        det (bool):If True, will perform deterministic sampling

    Returns:
        samples (tuple): [N_rays, N_samples]
    """

    M = weights.shape[1]
    weights += 1e-5
    # Get pdf
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf], dim=-1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., N_samples, device=bins.device)
        u = u.unsqueeze(0).repeat(bins.shape[0], 1)
    else:
        u = torch.rand(bins.shape[0], N_samples, device=bins.device)

    # Invert CDF
    above_inds = torch.zeros_like(u, dtype=torch.long)
    for i in range(M):
        above_inds += (u >= cdf[:, i:i + 1]).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds - 1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=2)

    cdf = cdf.unsqueeze(1).repeat(1, N_samples, 1)
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)

    bins = bins.unsqueeze(1).repeat(1, N_samples, 1)
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)

    denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[:, :, 0]) / denom

    samples = bins_g[:, :, 0] + t * (bins_g[:, :, 1] - bins_g[:, :, 0])

    return samples

def get_voxel_indices(pts, grid_shape):
    """
    Get voxel indices for the given points.
    Args:
        pts: [N_rays, N_samples, 3], points to get voxel indices for
        grid_shape: (D, W, H), shape of the feature grid
    Returns:
        indices: [N_rays, N_samples, 3], voxel indices for the points
    """
    D, W, H = grid_shape
    indices = torch.floor(pts).long()
    indices[..., 0] = torch.clamp(indices[..., 0], 0, D - 1)
    indices[..., 1] = torch.clamp(indices[..., 1], 0, W - 1)
    indices[..., 2] = torch.clamp(indices[..., 2], 0, H - 1)
    return indices

def get_voxel_features(features, indices):
    """
    Get voxel features for the given indices.
    Args:
        features: [C, D, W, H], feature volume
        indices: [N_rays, N_samples, 3], voxel indices
    Returns:
        voxel_features: [N_rays, N_samples, C], voxel features for the indices
    """
    C, D, W, H = features.shape
    N_rays, N_samples = indices.shape[:2]
    voxel_features = features[:, indices[..., 0], indices[..., 1], indices[..., 2]]
    voxel_features = voxel_features.permute(1, 2, 0)  # [C, N_rays, N_samples] -> [N_rays, N_samples, C]
    return voxel_features
 


def sample_along_camera_ray(ray_o,
                            ray_d,
                            depth_range,
                            N_samples,
                            inv_uniform=False,
                            det=False):
    """Sampling along the camera ray.

    Args:
        ray_o (tensor): Origin of the ray in scene coordinate system;
            tensor of shape [N_rays, 3]
        ray_d (tensor): Homogeneous ray direction vectors in
            scene coordinate system; tensor of shape [N_rays, 3]
        depth_range (tuple): [near_depth, far_depth]
        inv_uniform (bool): If True,uniformly sampling inverse depth.
        det (bool): If True, will perform deterministic sampling.
    Returns:
        pts (tensor): Tensor of shape [N_rays, N_samples, 3]
        z_vals (tensor): Tensor of shape [N_rays, N_samples]
    """
    # will sample inside [near_depth, far_depth]
    # assume the nearest possible depth is at least (min_ratio * depth)
    near_depth_value = depth_range[0]
    far_depth_value = depth_range[1]
    assert near_depth_value > 0 and far_depth_value > 0 \
        and far_depth_value > near_depth_value

    near_depth = near_depth_value * torch.ones_like(ray_d[..., 0])

    far_depth = far_depth_value * torch.ones_like(ray_d[..., 0])

    if inv_uniform: #False
        start = 1. / near_depth
        step = (1. / far_depth - start) / (N_samples - 1)
        inv_z_vals = torch.stack([start + i * step for i in range(N_samples)],
                                 dim=1)
        z_vals = 1. / inv_z_vals
    else:
        start = near_depth
        step = (far_depth - near_depth) / (N_samples - 1)
        z_vals = torch.stack([start + i * step for i in range(N_samples)],
                             dim=1)

    if not det: # this 
        # get intervals between samples
        mids = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, 0:1], mids], dim=-1)
        # uniform samples in those intervals
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand

    ray_d = ray_d.unsqueeze(1).repeat(1, N_samples, 1)
    ray_o = ray_o.unsqueeze(1).repeat(1, N_samples, 1)
    pts = z_vals.unsqueeze(2) * ray_d + ray_o  # [N_rays, N_samples, 3]
    return pts, z_vals

def trilinear_interpolation(features, pts):
    """
    Perform trilinear interpolation on the feature grid to obtain more accurate feature values at sample points.
    Args:
        features: [C, D, W, H], feature volume
        pts: [N_rays, N_samples, 3], sample points in normalized coordinates
    Returns:
        interpolated_features: [N_rays, N_samples, C], interpolated feature values at sample points
    """
    C, D, W, H = features.shape
    N_rays, N_samples, _ = pts.shape

    # Normalize points to the range [0, 1]
    pts = (pts + 1) / 2
    pts[..., 0] = pts[..., 0] * (D - 1)
    pts[..., 1] = pts[..., 1] * (W - 1)
    pts[..., 2] = pts[..., 2] * (H - 1)

    x = pts[..., 0].view(-1)
    y = pts[..., 1].view(-1)
    z = pts[..., 2].view(-1)
    
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1
    z0 = torch.floor(z).long()
    z1 = z0 + 1
    
    x0 = torch.clamp(x0, 0, D - 1)
    x1 = torch.clamp(x1, 0, D - 1)
    y0 = torch.clamp(y0, 0, W - 1)
    y1 = torch.clamp(y1, 0, W - 1)
    z0 = torch.clamp(z0, 0, H - 1)
    z1 = torch.clamp(z1, 0, H - 1)

    c000 = features[:, x0, y0, z0].permute(1, 0)
    c001 = features[:, x0, y0, z1].permute(1, 0)
    c010 = features[:, x0, y1, z0].permute(1, 0)
    c011 = features[:, x0, y1, z1].permute(1, 0)
    c100 = features[:, x1, y0, z0].permute(1, 0)
    c101 = features[:, x1, y0, z1].permute(1, 0)
    c110 = features[:, x1, y1, z0].permute(1, 0)
    c111 = features[:, x1, y1, z1].permute(1, 0)
    
    xd = (x - x0.float()).unsqueeze(1)
    yd = (y - y0.float()).unsqueeze(1)
    zd = (z - z0.float()).unsqueeze(1)
    
    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd
    
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd
    
    interpolated_features = c0 * (1 - zd) + c1 * zd
    interpolated_features = interpolated_features.view(N_rays, N_samples, C)
    
    return interpolated_features

def importance_sampling3(ray_o, ray_d, pts, z_vals, density, rgb_pts, features, N_importance, det=False, detection_prior=None):
    with torch.no_grad():
        density = density.squeeze(-1)  # [N_rays, N_samples]

        if detection_prior is not None:
            density = density * detection_prior
        
        # 获取初始采样点的体素索引
        voxel_indices = get_voxel_indices(pts, features.shape[1:])  # [N_rays, N_samples, 3]

        # 直接获取特征值
        feature_values = features[:, voxel_indices[..., 0], voxel_indices[..., 1], voxel_indices[..., 2]]
        feature_dists = torch.norm(feature_values, dim=0)

        # 计算颜色梯度
        color_gradients = torch.zeros_like(rgb_pts)
        color_gradients[:, 1:, :] += torch.abs(rgb_pts[:, 1:, :] - rgb_pts[:, :-1, :])
        color_gradients[:, :-1, :] += torch.abs(rgb_pts[:, 1:, :] - rgb_pts[:, :-1, :])

        color_dists = torch.norm(color_gradients, dim=2)

        # 归一化密度
        density_min, density_max = density.min(dim=-1, keepdim=True)[0], density.max(dim=-1, keepdim=True)[0]
        density_normalized = (density - density_min) / (density_max - density_min + 1e-5)

        # 计算综合权重
        alpha = 1  # 特征值的权重
        beta = 0.5  # 颜色梯度的权重
        gamma = 0.5  # 密度权重
        total_weight = alpha + beta + gamma
        alpha /= total_weight
        beta /= total_weight
        gamma /= total_weight

        combined_dists = (feature_dists ** alpha + 1e-5) * (color_dists ** beta + 1e-5) * (density_normalized ** gamma + 1e-5)
        
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, dists[..., -1:]], dim=-1)

        weights = combined_dists
        weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

        # 对权重进行排序
        sorted_indices = torch.argsort(weights, dim=-1, descending=True)
        sorted_weights = torch.gather(weights, -1, sorted_indices)
        sorted_z_vals = torch.gather(z_vals, -1, sorted_indices)

        # 保证选择的点在射线上均匀分布
        N_rays, N_samples = sorted_weights.shape
        step = N_samples // N_importance
        selected_indices = []

        # 在每段中选择一个点，确保均匀分布
        for i in range(N_importance):
            start_idx = i * step
            end_idx = start_idx + step
            if end_idx > N_samples:
                end_idx = N_samples
            # 转换为浮点类型计算平均值，然后转换回长整型
            segment_indices = sorted_indices[:, start_idx:end_idx].float().mean(dim=-1).long()
            selected_indices.append(segment_indices)

        selected_indices = torch.stack(selected_indices, dim=-1)
        selected_z_vals = torch.gather(z_vals, 1, selected_indices)

        # 确保选择的z_vals在射线范围内分布均匀
        z_vals_fine, _ = torch.sort(selected_z_vals, dim=-1)
        pts_fine = z_vals_fine.unsqueeze(2) * ray_d.unsqueeze(1) + ray_o.unsqueeze(1)

    return pts_fine, z_vals_fine


def importance_sampling_gradient(ray_o, ray_d, pts, z_vals, density, rgb_pts, features, N_importance, det=False, detection_prior=None):
    with torch.no_grad():
        density = density.squeeze(-1)  # [N_rays, N_samples]

        if detection_prior is not None:
            density = density * detection_prior
        
        # 获取初始采样点的体素索引
        voxel_indices = get_voxel_indices(pts, features.shape[1:])  # [N_rays, N_samples, 3]

        # 计算特征梯度
        C, D, W, H = features.shape
        feature_gradients = torch.zeros_like(features)
        feature_gradients[:, 1:, :, :] += torch.abs(features[:, 1:, :, :] - features[:, :-1, :, :])
        feature_gradients[:, :-1, :, :] += torch.abs(features[:, 1:, :, :] - features[:, :-1, :, :])
        feature_gradients[:, :, 1:, :] += torch.abs(features[:, :, 1:, :] - features[:, :, :-1, :])
        feature_gradients[:, :, :-1, :] += torch.abs(features[:, :, 1:, :] - features[:, :, :-1, :])
        feature_gradients[:, :, :, 1:] += torch.abs(features[:, :, :, 1:] - features[:, :, :, :-1])
        feature_gradients[:, :, :, :-1] += torch.abs(features[:, :, :, 1:] - features[:, :, :, :-1])

        feature_dists = torch.norm(feature_gradients[:, voxel_indices[..., 0], voxel_indices[..., 1], voxel_indices[..., 2]], dim=0)

        # 计算颜色梯度
        color_gradients = torch.zeros_like(rgb_pts)
        color_gradients[:, 1:, :] += torch.abs(rgb_pts[:, 1:, :] - rgb_pts[:, :-1, :])
        color_gradients[:, :-1, :] += torch.abs(rgb_pts[:, 1:, :] - rgb_pts[:, :-1, :])

        color_dists = torch.norm(color_gradients, dim=2)

        # 归一化密度
        density_min, density_max = density.min(dim=-1, keepdim=True)[0], density.max(dim=-1, keepdim=True)[0]
        density_normalized = (density - density_min) / (density_max - density_min + 1e-5)

        # 计算综合权重
        alpha = 1  # 特征梯度的权重
        beta = 1  # 颜色梯度的权重
        gamma = 1  # 密度权重
        total_weight = alpha + beta + gamma
        alpha /= total_weight
        beta /= total_weight
        gamma /= total_weight

        combined_dists = (feature_dists ** alpha + 1e-5) * (color_dists ** beta + 1e-5) * (density_normalized ** gamma + 1e-5)
        
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, dists[..., -1:]], dim=-1)

        weights = combined_dists
        weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

        # 对权重进行排序
        sorted_indices = torch.argsort(weights, dim=-1, descending=True)
        sorted_weights = torch.gather(weights, -1, sorted_indices)
        sorted_z_vals = torch.gather(z_vals, -1, sorted_indices)

        # 保证选择的点在射线上均匀分布
        N_rays, N_samples = sorted_weights.shape
        step = N_samples // N_importance
        selected_indices = []
        for i in range(0, N_samples, step):
            selected_indices.append(sorted_indices[:, i])

        selected_indices = torch.stack(selected_indices, dim=-1)[:, :N_importance]
        selected_z_vals = torch.gather(z_vals, 1, selected_indices)

        # 确保选择的z_vals在射线范围内分布均匀
        z_vals_fine, _ = torch.sort(selected_z_vals, dim=-1)
        pts_fine = z_vals_fine.unsqueeze(2) * ray_d.unsqueeze(1) + ray_o.unsqueeze(1)

    return pts_fine, z_vals_fine

def importance_sampling_2(ray_o, ray_d, pts, z_vals, density, rgb_pts, features, N_importance, det=False, detection_prior=None):
    with torch.no_grad():
        density = density.squeeze(-1)  # [N_rays, N_samples]

        if detection_prior is not None:
            density = density * detection_prior
        
        # 获取初始采样点的体素索引
        voxel_indices = get_voxel_indices(pts, features.shape[1:])  # [N_rays, N_samples, 3]

        # 计算特征梯度
        C, D, W, H = features.shape
        feature_gradients = torch.zeros_like(features)
        feature_gradients[:, 1:, :, :] += torch.abs(features[:, 1:, :, :] - features[:, :-1, :, :])
        feature_gradients[:, :-1, :, :] += torch.abs(features[:, 1:, :, :] - features[:, :-1, :, :])
        feature_gradients[:, :, 1:, :] += torch.abs(features[:, :, 1:, :] - features[:, :, :-1, :])
        feature_gradients[:, :, :-1, :] += torch.abs(features[:, :, 1:, :] - features[:, :, :-1, :])
        feature_gradients[:, :, :, 1:] += torch.abs(features[:, :, :, 1:] - features[:, :, :, :-1])
        feature_gradients[:, :, :, :-1] += torch.abs(features[:, :, :, 1:] - features[:, :, :, :-1])

        feature_dists = torch.norm(feature_gradients[:, voxel_indices[..., 0], voxel_indices[..., 1], voxel_indices[..., 2]], dim=0)

        # 计算颜色梯度
        color_gradients = torch.zeros_like(rgb_pts)
        color_gradients[:, 1:, :] += torch.abs(rgb_pts[:, 1:, :] - rgb_pts[:, :-1, :])
        color_gradients[:, :-1, :] += torch.abs(rgb_pts[:, 1:, :] - rgb_pts[:, :-1, :])

        color_dists = torch.norm(color_gradients, dim=2)

        # 归一化密度
        density_min, density_max = density.min(dim=-1, keepdim=True)[0], density.max(dim=-1, keepdim=True)[0]
        density_normalized = (density - density_min) / (density_max - density_min + 1e-5)

        # 计算综合权重
        alpha = 1  # 特征梯度的权重
        beta = 0.5  # 颜色梯度的权重
        gamma = 0.5  # 密度权重
        #combined_dists = alpha * feature_dists + beta * color_dists + gamma * density_normalized
        #combined_dists = feature_dists * color_dists * density_normalized

        #combined_dists = torch.exp((torch.log(feature_dists) + torch.log(color_dists) + torch.log(density_normalized)) / 3)
        #combined_dists = torch.exp((alpha * torch.log(feature_dists) + beta * torch.log(color_dists) + gamma * torch.log(density_normalized)) / (alpha + beta + gamma))
        total_weight = alpha + beta + gamma
        alpha /= total_weight
        beta /= total_weight
        gamma /= total_weight

        combined_dists = (feature_dists ** alpha + 1e-5) * (color_dists ** beta + 1e-5) * (density_normalized ** gamma + 1e-5)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, dists[..., -1:]], dim=-1)

        weights = combined_dists
        weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

        cdf = torch.cumsum(weights, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf], dim=-1)

        if det:
            u = torch.linspace(0., 1., N_importance, device=cdf.device)
            u = u.unsqueeze(0).repeat(cdf.shape[0], 1)
        else:
            u = torch.rand(cdf.shape[0], N_importance, device=cdf.device)

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], dim=-1)

        cdf = cdf.unsqueeze(1).expand([-1, N_importance, -1])
        bins = torch.cat([z_vals, z_vals[:, -1:]], dim=-1)
        bins = bins.unsqueeze(1).expand([-1, N_importance, -1])

        cdf_g = torch.gather(cdf, 2, inds_g)
        bins_g = torch.gather(bins, 2, inds_g)

        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        z_samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        z_vals = z_vals[:, ::2]
        z_samples = torch.clamp(z_samples, min=z_vals.min(), max=z_vals.max())
        #z_vals_fine, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
        z_vals_fine, _ = torch.sort(z_samples, dim=-1)
        pts_fine = z_vals_fine.unsqueeze(2) * ray_d.unsqueeze(1) + ray_o.unsqueeze(1)

    return pts_fine, z_vals_fine

def importance_sampling_0903(ray_o, ray_d, pts, points,z_vals, density, rgb_pts, features, N_importance, det=False, detection_prior=None):
    with torch.no_grad():
        density = density.squeeze(-1)  # [N_rays, N_samples]

        if detection_prior is not None:
            density = density * detection_prior
        
        points_flat = points.view(3, -1).transpose(0, 1).cpu().numpy()

        # 将 pts 也转换为 [M, 3] 形状，并移动到 CPU，转换为 numpy 数组
        pts_flat = pts.view(-1, 3).cpu().numpy()

        # 使用 KDTree 进行近邻搜索
        kdtree = KDTree(points_flat)

        # 定义搜索半径
        radius = 1  # 根据你的数据尺度选择合适的半径

        # 计算每个 pts 中点在 points 中的邻居数目（即密度）
        pts_density = np.array([len(kdtree.query_ball_point(pt, radius)) for pt in pts_flat])

        # 如果你需要将结果转换回 torch.Tensor
        pts_density_tensor = torch.from_numpy(pts_density).float()
        
        # 获取初始采样点的体素索引
        voxel_indices = get_voxel_indices(pts, features.shape[1:])  # [N_rays, N_samples, 3]

        # 直接获取特征值
        C, D, W, H = features.shape
        N_rays, N_samples, _ = voxel_indices.shape
        
        # 将 voxel_indices 转换为一维索引
        voxel_indices_flat = voxel_indices[..., 0] * W * H + voxel_indices[..., 1] * H + voxel_indices[..., 2]
        voxel_indices_flat = voxel_indices_flat.view(-1)
        
        # 提取特征值
        feature_values = features.view(C, -1)[:, voxel_indices_flat]
        feature_values = feature_values.view(C, N_rays, N_samples)
        feature_values = feature_values.permute(1, 2, 0)  # [N_rays, N_samples, C]
        interpolated_features = trilinear_interpolation(features, pts)
        feature_dists = torch.norm(interpolated_features, dim=-1)

        # 计算颜色梯度
        color_gradients = torch.zeros_like(rgb_pts)
        color_gradients[:, 1:, :] += torch.abs(rgb_pts[:, 1:, :] - rgb_pts[:, :-1, :])
        color_gradients[:, :-1, :] += torch.abs(rgb_pts[:, 1:, :] - rgb_pts[:, :-1, :])

        color_dists = torch.norm(color_gradients, dim=2)

        # 归一化密度
        density_min, density_max = density.min(dim=-1, keepdim=True)[0], density.max(dim=-1, keepdim=True)[0]
        density_normalized = (density - density_min) / (density_max - density_min + 1e-5)

        # 计算综合权重
        alpha = 0  # 特征值的权重
        beta = 0  # 颜色梯度的权重
        gamma = 1 # 密度权重
        
        combined_dists = alpha * feature_dists + beta * color_dists + gamma * density_normalized

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, dists[..., -1:]], dim=-1)

        weights = combined_dists
        weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

        cdf = torch.cumsum(weights, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf], dim=-1)

        if det:
            u = torch.linspace(0., 1., N_importance, device=cdf.device)
            u = u.unsqueeze(0).repeat(cdf.shape[0], 1)
        else:
            u = torch.rand(cdf.shape[0], N_importance, device=cdf.device)

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], dim=-1)

        cdf = cdf.unsqueeze(1).expand([-1, N_importance, -1])
        bins = torch.cat([z_vals, z_vals[:, -1:]], dim=-1)
        bins = bins.unsqueeze(1).expand([-1, N_importance, -1])

        cdf_g = torch.gather(cdf, 2, inds_g)
        bins_g = torch.gather(bins, 2, inds_g)

        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        z_samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        z_vals = z_vals[:, ::2]
        z_samples = torch.clamp(z_samples, min=z_vals.min(), max=z_vals.max())
        #z_vals_fine, _ = torch.sort(z_samples, dim=-1)
        z_vals_fine, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
        pts_fine = z_vals_fine.unsqueeze(2) * ray_d.unsqueeze(1) + ray_o.unsqueeze(1)

    return pts_fine, z_vals_fine
def importance_sampling(ray_o, ray_d, pts, points, z_vals, density, rgb_pts, features, N_importance, det=False, detection_prior=None):
    device = ray_o.device
    with torch.no_grad():
        density = density.squeeze(-1)

        if detection_prior is not None:
            density = density * detection_prior
        # # 将 points_flat 转换为 numpy 数组并移动到 CPU
        # points_flat = points.view(3, -1).transpose(0, 1).cpu().numpy().astype(np.float32)
        # # 使用 scikit-learn 的 NearestNeighbors 进行近邻搜索
        # nbrs = NearestNeighbors(radius=0.5, algorithm='auto').fit(points_flat)        
        # # 将 pts 转换为 numpy 数组并移动到 CPU
        # pts_flat = pts.view(-1, 3).cpu().numpy().astype(np.float32)
        # # 找到每个点的近邻
        # indices = nbrs.radius_neighbors(pts_flat, return_distance=False)
        # # 计算每个 pts 中点在 points 中的邻居数目（即密度）
        # pts_density = np.array([len(ind) for ind in indices])
        # # 将结果转换为 torch.Tensor 并移动到相同设备
        # pts_density_tensor = torch.from_numpy(pts_density).float().to(device)
        # # 对 pts_density_tensor 进行归一化处理
        # density_min, density_max = pts_density_tensor.min(), pts_density_tensor.max()
        # pts_density_normalized = (pts_density_tensor - density_min) / (density_max - density_min + 1e-5)


        # 获取初始采样点的体素索引
        voxel_indices = get_voxel_indices(pts, features.shape[1:])  # [N_rays, N_samples, 3]

        # 直接获取特征值
        C, D, W, H = features.shape
        N_rays, N_samples, _ = voxel_indices.shape
        
        # 将 voxel_indices 转换为一维索引
        voxel_indices_flat = voxel_indices[..., 0] * W * H + voxel_indices[..., 1] * H + voxel_indices[..., 2]
        voxel_indices_flat = voxel_indices_flat.view(-1)
        
        # 提取特征值
        feature_values = features.view(C, -1)[:, voxel_indices_flat]
        feature_values = feature_values.view(C, N_rays, N_samples)
        feature_values = feature_values.permute(1, 2, 0)  # [N_rays, N_samples, C]
        interpolated_features = trilinear_interpolation(features, pts)
        feature_dists = torch.norm(interpolated_features, dim=-1)

        # 计算颜色梯度
        color_gradients = torch.zeros_like(rgb_pts)
        color_gradients[:, 1:, :] += torch.abs(rgb_pts[:, 1:, :] - rgb_pts[:, :-1, :])
        color_gradients[:, :-1, :] += torch.abs(rgb_pts[:, 1:, :] - rgb_pts[:, :-1, :])

        color_dists = torch.norm(color_gradients, dim=2)

        # 归一化 density
        density_min, density_max = density.min(dim=-1, keepdim=True)[0], density.max(dim=-1, keepdim=True)[0]
        density_normalized = (density - density_min) / (density_max - density_min + 1e-5)

        # 计算综合权重
        alpha = 0  # 特征值的权重
        beta = 0  # 颜色梯度的权重
        gamma = 1  # 密度权重
        #delta = 1  # pts_density_tensor 的权重 (可以调整这个值来控制影响)

        combined_dists = (
            alpha * feature_dists 
            + beta * color_dists 
            + gamma * density_normalized 
           # + delta * pts_density_normalized.view(N_rays, N_samples)
        )

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, dists[..., -1:]], dim=-1)

        weights = combined_dists
        weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

        cdf = torch.cumsum(weights, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf], dim=-1)

        if det:
            u = torch.linspace(0., 1., N_importance, device=cdf.device)
            u = u.unsqueeze(0).repeat(cdf.shape[0], 1)
        else:
            u = torch.rand(cdf.shape[0], N_importance, device=cdf.device)

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], dim=-1)

        cdf = cdf.unsqueeze(1).expand([-1, N_importance, -1])
        bins = torch.cat([z_vals, z_vals[:, -1:]], dim=-1)
        bins = bins.unsqueeze(1).expand([-1, N_importance, -1])

        cdf_g = torch.gather(cdf, 2, inds_g)
        bins_g = torch.gather(bins, 2, inds_g)

        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        z_samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        z_vals = z_vals[:, ::2]
        z_samples = torch.clamp(z_samples, min=z_vals.min(), max=z_vals.max())
        #z_vals_fine, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
        z_vals_fine, _ = torch.sort(z_samples, dim=-1)
        pts_fine = z_vals_fine.unsqueeze(2) * ray_d.unsqueeze(1) + ray_o.unsqueeze(1)

    return pts_fine, z_vals_fine


# ray rendering of nerf
def raw2outputs(raw, z_vals, mask, white_bkgd=False):
    """Transform raw data to outputs:

    Args:
        raw(tensor):Raw network output.Tensor of shape [N_rays, N_samples, 4]
        z_vals(tensor):Depth of point samples along rays.
            Tensor of shape [N_rays, N_samples]
        ray_d(tensor):[N_rays, 3]

    Returns:
        ret(dict):
            -rgb(tensor):[N_rays, 3]
            -depth(tensor):[N_rays,]
            -weights(tensor):[N_rays,]
            -depth_std(tensor):[N_rays,]
    """
    rgb = raw[:, :, :3]  # [N_rays, N_samples, 3]
    sigma = raw[:, :, 3]  # [N_rays, N_samples]

    # note: we did not use the intervals here,
    # because in practice different scenes from COLMAP can have
    # very different scales, and using interval can affect
    # the model's generalization ability.
    # Therefore we don't use the intervals for both training and evaluation.
    sigma2alpha = lambda sigma, dists: 1. - torch.exp(-sigma)  # noqa

    ****EMPTY****

    if white_bkgd:
        ****EMPTY****

    if mask is not None:
        mask = mask.float().sum(dim=1) > 8

    ****EMPTY****

    ret = OrderedDict([('rgb', rgb_map), ('depth', depth_map),
                       ('weights', weights), ('mask', mask), ('alpha', alpha),
                       ('z_vals', z_vals), ('transparency', T)])

    return ret


def render_rays_func(
        voxel_size,
        points,
        features_3D,
        ray_o,
        ray_d,
        mean_volume,
        cov_volume,
        features_2D,
        img,
        aabb,
        near_far_range,
        N_samples,
        N_rand=4096,
        nerf_mlp=None,
        img_meta=None,
        projector=None,
        mode='volume',  # volume and image
        nerf_sample_view=3,
        inv_uniform=False,
        N_importance=0,
        det=False,
        is_train=True,
        white_bkgd=False,
        gt_rgb=None,
        gt_depth=None):

    ret = {
        'outputs_coarse': None,
        'outputs_fine': None,
        'gt_rgb': gt_rgb,
        'gt_depth': gt_depth
    }

    # pts: [N_rays, N_samples, 3]
    # z_vals: [N_rays, N_samples]
    pts, z_vals = sample_along_camera_ray(
        ray_o=ray_o,
        ray_d=ray_d,
        depth_range=near_far_range,
        N_samples=N_samples,
        inv_uniform=inv_uniform,
        det=det)
    N_rays, N_samples = pts.shape[:2]

    if mode == 'image':     
        img = img.permute(0, 2, 3, 1).unsqueeze(0)
        train_camera = _compute_projection(img_meta).to(img.device)
        rgb_feat, mask = projector.compute(
            pts, img, train_camera, features_2D, grid_sample=True) # rgb_feat torch.Size([2048, 128, 40, 35])
    
        pixel_mask = mask[..., 0].sum(dim=2) > 1
        mean, var = compute_mask_points(rgb_feat, mask)
        globalfeat = torch.cat([mean, var], dim=-1).squeeze(2)
        rgb_pts, density_pts = nerf_mlp(pts, ray_d, globalfeat) # rgb_pts torch.Size([2048, 128, 3])
 
        raw_coarse = torch.cat([rgb_pts, density_pts], dim=-1)
        ret['sigma'] = density_pts

    # outputs_coarse = raw2outputs(
    #     raw_coarse, z_vals, pixel_mask, white_bkgd=white_bkgd)
    # ret['outputs_coarse'] = outputs_coarse

    N_importance = 128
    if N_importance > 0:
        #pts_fine, z_vals_fine = importance_sampling(ray_o, ray_d, pts,points, z_vals, density_pts,rgb_pts,features_3D,voxel_size,N_importance, det=det)
        pts_fine, z_vals_fine = importance_sampling(ray_o, ray_d, pts,points, z_vals, density_pts,rgb_pts,features_3D, N_importance, det=det)

        #train_camera = _compute_projection(img_meta) 
        rgb_feat, mask = projector.compute(
            pts_fine, img, train_camera, features_2D, grid_sample=True)
        pixel_mask = mask[..., 0].sum(dim=2) > 1
        mean, var = compute_mask_points(rgb_feat, mask)
        globalfeat = torch.cat([mean, var], dim=-1).squeeze(2)
        rgb_pts, density_pts = nerf_mlp(pts_fine, ray_d, globalfeat)
        raw_coarse = torch.cat([rgb_pts, density_pts], dim=-1)
        ret['sigma'] = density_pts
        # print("pts",pts.shape) # torch.Size([2048, 64, 3])
        # print("z_vals",z_vals.shape)  #torch.Size([2048, 64])       
        # print("pts_fine",pts_fine.shape) #torch.Size([2048, 128, 3])
        # print("z_vals_fine",z_vals_fine.shape)  # torch.Size([2048, 128])       
        # 重新计算 fine 采样点的特征
        outputs_coarse = raw2outputs(
            raw_coarse, z_vals_fine, pixel_mask, white_bkgd=white_bkgd)
        ret['outputs_coarse'] = outputs_coarse

    return ret

def render_rays(
        voxel_size,
        points,
        features_3D,
        ray_batch,
        mean_volume,
        cov_volume,
        features_2D,
        img,
        aabb,
        near_far_range,
        N_samples,
        N_rand=4096,
        nerf_mlp=None,
        img_meta=None,
        projector=None,
        mode='volume',  # volume and image
        nerf_sample_view=3,
        inv_uniform=False,
        N_importance=0,
        det=False,
        is_train=True,
        white_bkgd=False,
        render_testing=False):
    """The function of the nerf rendering."""

    ray_o = ray_batch['ray_o']
    ray_d = ray_batch['ray_d']
    gt_rgb = ray_batch['gt_rgb']
    gt_depth = ray_batch['gt_depth']
    nerf_sizes = ray_batch['nerf_sizes']
    if is_train:
        ray_o = ray_o.view(-1, 3)
        ray_d = ray_d.view(-1, 3)
        gt_rgb = gt_rgb.view(-1, 3)
        if gt_depth.shape[1] != 0:
            gt_depth = gt_depth.view(-1, 1)
            non_zero_depth = (gt_depth > 0).squeeze(-1)
            ray_o = ray_o[non_zero_depth]
            ray_d = ray_d[non_zero_depth]
            gt_rgb = gt_rgb[non_zero_depth]
            gt_depth = gt_depth[non_zero_depth]
        else:
            gt_depth = None
        total_rays = ray_d.shape[0]
        select_inds = rng.choice(total_rays, size=(N_rand, ), replace=False)
        ray_o = ray_o[select_inds]
        ray_d = ray_d[select_inds]
        gt_rgb = gt_rgb[select_inds]
        if gt_depth is not None:
            gt_depth = gt_depth[select_inds]

        rets = render_rays_func(
            voxel_size,
            points,
            features_3D,
            ray_o,
            ray_d,
            mean_volume,
            cov_volume,
            features_2D,
            img,
            aabb,
            near_far_range,
            N_samples,
            N_rand,
            nerf_mlp,
            img_meta,
            projector,
            mode,  # volume and image
            nerf_sample_view,
            inv_uniform,
            N_importance,
            det,
            is_train,
            white_bkgd,
            gt_rgb,
            gt_depth)

    elif render_testing:
        #print("nerf_size:",len(nerf_sizes))
        nerf_size = nerf_sizes[0]
        view_num = ray_o.shape[1]
        H = nerf_size[0][0]
        W = nerf_size[0][1]

        ray_o = ray_o.view(-1, 3)
        ray_d = ray_d.view(-1, 3)
        gt_rgb = gt_rgb.view(-1, 3)
        # print("nerf_size:",nerf_size)
        # print("ray_o:",ray_o.shape)
        # print("gt_rgb:",gt_rgb.shape)
        if len(gt_depth) != 0:
            gt_depth = gt_depth.view(-1, 1)
        else:
            gt_depth = None
        assert view_num * H * W == ray_o.shape[0]
        num_rays = ray_o.shape[0]
        results = []
        rgbs = []
        for i in range(0, num_rays, N_rand):
            ray_o_chunck = ray_o[i:i + N_rand, :]
            ray_d_chunck = ray_d[i:i + N_rand, :]

            ret = render_rays_func(voxel_size,points,features_3D,ray_o_chunck, ray_d_chunck, mean_volume,
                                   cov_volume, features_2D, img, aabb,
                                   near_far_range, N_samples, N_rand, nerf_mlp,
                                   img_meta, projector, mode, nerf_sample_view,
                                   inv_uniform, N_importance, True, is_train,
                                   white_bkgd, gt_rgb, gt_depth)
            results.append(ret)

        rgbs = []
        depths = []

        if results[0]['outputs_coarse'] is not None:
            for i in range(len(results)):
                rgb = results[i]['outputs_coarse']['rgb']
                rgbs.append(rgb)
                depth = results[i]['outputs_coarse']['depth']
                depths.append(depth)

        rets = {
            'outputs_coarse': {
                'rgb': torch.cat(rgbs, dim=0).view(view_num, H, W, 3),
                'depth': torch.cat(depths, dim=0).view(view_num, H, W, 1),
            },
            'gt_rgb':
            gt_rgb.view(view_num, H, W, 3),
            'gt_depth':
            gt_depth.view(view_num, H, W, 1) if gt_depth is not None else None,
        }
    else:
        rets = None
    return rets