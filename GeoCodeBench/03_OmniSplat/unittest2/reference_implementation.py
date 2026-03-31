"""
Reference Implementation for cross_warp_with_pose_depth_candidates
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch
import torch.nn.functional as F


def coords_grid(b, h, w, homogeneous=False, device=None):
    """Generate coordinate grid."""
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]
    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def yin_to_3d(grid, h, w):
    """Convert yin projection grid to 3D coordinates."""
    grid_x, grid_y = grid[:, 0], grid[:, 1]

    lat = -grid_y * (torch.pi / 2) / (h - 1) + torch.pi / 4
    lon = grid_x * (3 * torch.pi / 2) / (w - 1) - 3 * torch.pi / 4

    x = torch.cos(lat) * torch.sin(lon)
    y = -torch.sin(lat)
    z = torch.cos(lat) * torch.cos(lon)
    world_grid = torch.stack([x, y, z], dim=1)

    return world_grid


def yang90_to_3d(grid, h, w):
    """Convert yang90 projection grid to 3D coordinates."""
    grid_x, grid_y = grid[:, 0], grid[:, 1]

    lat = -grid_y * (torch.pi / 2) / (h - 1) + torch.pi / 4
    lon = grid_x * (3 * torch.pi / 2) / (w - 1) - 3 * torch.pi / 4

    x = -torch.sin(lat)
    y = torch.cos(lat) * torch.sin(lon)
    z = -torch.cos(lat) * torch.cos(lon)
    world_grid = torch.stack([x, y, z], dim=1)

    return world_grid


def yin_from_3d(points, h, w):
    """Project 3D points to yin projection grid."""
    points_x, points_y, points_z = points[:, 0], points[:, 1], points[:, 2]
    points_xz = torch.sqrt(points_x**2 + points_z**2)

    lat = torch.atan2(-points_y, points_xz)
    lon = torch.atan2(points_x, points_z)

    u = lon * 2 * (w - 1) / 3 / torch.pi + (w - 1) / 2
    v = -lat * 2 * (h - 1) / torch.pi + (h - 1) / 2
    ones = torch.ones_like(u)
    grid = torch.stack([u, v, ones], dim=1)

    return grid


def yang90_from_3d(points, h, w):
    """Project 3D points to yang90 projection grid."""
    points_x, points_y, points_z = points[:, 0], points[:, 1], points[:, 2]
    points_yz = torch.sqrt(points_y**2 + points_z**2)

    lat = torch.atan2(-points_x, points_yz)
    lon = torch.atan2(points_y, -points_z)

    u = lon * 2 * (w - 1) / 3 / torch.pi + (w - 1) / 2
    v = -lat * 2 * (h - 1) / torch.pi + (h - 1) / 2
    ones = torch.ones_like(u)
    grid = torch.stack([u, v, ones], dim=1)

    return grid


def cross_warp_with_pose_depth_candidates(
    feature1,
    intrinsics,
    pose,
    depth,
    clamp_min_depth=1e-3,
    warp_padding_mode="zeros",
):
    """
    Cross-warp features using pose and depth candidates.
    
    Args:
        feature1: [B, C, H, W, 2] # yin, yang90 order
        intrinsics: [B, 3, 3]
        pose: [B, 4, 4]
        depth: [B, D, H, W]
        clamp_min_depth: Minimum depth for clamping
        warp_padding_mode: Padding mode for grid_sample
        
    Returns:
        warped_feature: [B, C, D, H, W, 2]
    """
    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4
    assert feature1.size(-1) == 2

    b, d, h, w = depth.size()
    c = feature1.size(1)
    
    feature_yin = feature1[..., 0]  # b c h w
    feature_yang = feature1[..., 1]
    
    feature_one = torch.ones_like(feature_yin)[:, 0:1]  # b 1 h w
    feature_yin = torch.cat([feature_yin, feature_one], dim=1)  # b (c+1) h w
    feature_yang = torch.cat([feature_yang, feature_one], dim=1)  # b (c+1) h w
    
    with torch.no_grad():
        # pixel coordinates
        grid = coords_grid(
            b, h, w, homogeneous=True, device=depth.device
        )  # [B, 3, H, W]
        # back project to 3D and transform viewpoint

        points_from_yin = yin_to_3d(grid.view(b, 3, -1), h, w)   # [B, 3, H*W]
        points_from_yin = torch.bmm(pose[:, :3, :3], points_from_yin).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W]
        points_from_yin = points_from_yin + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        
        points_from_yang = yang90_to_3d(grid.view(b, 3, -1), h, w)   # [B, 3, H*W]
        points_from_yang = torch.bmm(pose[:, :3, :3], points_from_yang).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W]
        points_from_yang = points_from_yang + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        
        points_to_yin_from_yin = yin_from_3d(points_from_yin, h, w)
        points_to_yang_from_yin = yang90_from_3d(points_from_yin, h, w)
        points_to_yin_from_yang = yin_from_3d(points_from_yang, h, w)
        points_to_yang_from_yang = yang90_from_3d(points_from_yang, h, w)

        pixel_coords_to_yin_from_yin = points_to_yin_from_yin[:, :2] / points_to_yin_from_yin[:, -1:].clamp(min=clamp_min_depth)  # [B, 2, D, H*W]
        pixel_coords_to_yang_from_yin = points_to_yang_from_yin[:, :2] / points_to_yang_from_yin[:, -1:].clamp(min=clamp_min_depth)  # [B, 2, D, H*W]
        pixel_coords_to_yin_from_yang = points_to_yin_from_yang[:, :2] / points_to_yin_from_yang[:, -1:].clamp(min=clamp_min_depth)  # [B, 2, D, H*W]
        pixel_coords_to_yang_from_yang = points_to_yang_from_yang[:, :2] / points_to_yang_from_yang[:, -1:].clamp(min=clamp_min_depth)  # [B, 2, D, H*W]

        # normalize to [-1, 1]
        x_grid_tifi = 2 * pixel_coords_to_yin_from_yin[:, 0] / (w - 1) - 1
        y_grid_tifi = 2 * pixel_coords_to_yin_from_yin[:, 1] / (h - 1) - 1
        grid_tifi = torch.stack([x_grid_tifi, y_grid_tifi], dim=-1)  # [B, D, H*W, 2]

        x_grid_tafi = 2 * pixel_coords_to_yang_from_yin[:, 0] / (w - 1) - 1
        y_grid_tafi = 2 * pixel_coords_to_yang_from_yin[:, 1] / (h - 1) - 1
        grid_tafi = torch.stack([x_grid_tafi, y_grid_tafi], dim=-1)  # [B, D, H*W, 2]

        x_grid_tifa = 2 * pixel_coords_to_yin_from_yang[:, 0] / (w - 1) - 1
        y_grid_tifa = 2 * pixel_coords_to_yin_from_yang[:, 1] / (h - 1) - 1
        grid_tifa = torch.stack([x_grid_tifa, y_grid_tifa], dim=-1)  # [B, D, H*W, 2]

        x_grid_tafa = 2 * pixel_coords_to_yang_from_yang[:, 0] / (w - 1) - 1
        y_grid_tafa = 2 * pixel_coords_to_yang_from_yang[:, 1] / (h - 1) - 1
        grid_tafa = torch.stack([x_grid_tafa, y_grid_tafa], dim=-1)  # [B, D, H*W, 2]

    # sample features
    warped_features_tifi = F.grid_sample(
        feature_yin,
        grid_tifi.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(
        b, c+1, d, h, w
    )  # [B, C+1, D, H, W]
    warped_feature_tifi, warped_weight_tifi = warped_features_tifi[:, :c], warped_features_tifi[:, c:]

    warped_features_tifa = F.grid_sample(
        feature_yin,
        grid_tifa.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(
        b, c+1, d, h, w
    )  # [B, C+1, D, H, W]
    warped_feature_tifa, warped_weight_tifa = warped_features_tifa[:, :c], warped_features_tifa[:, c:]

    warped_features_tafi = F.grid_sample(
        feature_yang,
        grid_tafi.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(
        b, c+1, d, h, w
    )  # [B, C+1, D, H, W]
    warped_feature_tafi, warped_weight_tafi = warped_features_tafi[:, :c], warped_features_tafi[:, c:]

    warped_features_tafa = F.grid_sample(
        feature_yang,
        grid_tafa.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(
        b, c+1, d, h, w
    )  # [B, C+1, D, H, W]
    warped_feature_tafa, warped_weight_tafa = warped_features_tafa[:, :c], warped_features_tafa[:, c:]

    weight_yin = warped_weight_tifi / (warped_weight_tifi + warped_weight_tafi + 1e-8)
    warped_feature_yin = weight_yin * warped_feature_tifi + (1-weight_yin) * warped_feature_tafi

    weight_yang = warped_weight_tafa / (warped_weight_tifa + warped_weight_tafa + 1e-8)
    warped_feature_yang = (1-weight_yang) * warped_feature_tifa + weight_yang * warped_feature_tafa
    
    warped_feature = torch.stack([warped_feature_yin, warped_feature_yang], dim=-1)  # b c d h w 2

    return warped_feature

