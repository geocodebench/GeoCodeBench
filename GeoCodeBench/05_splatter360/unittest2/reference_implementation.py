"""
Reference Implementation for correlation_softmax_depth
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


def warp_with_pose_depth_candidates(feature1, intrinsics, pose, depth,
                                    clamp_min_depth=1e-3,
                                    ):
    """
    Warp feature1 to the viewpoint of feature0 using depth candidates.
    
    Args:
        feature1: [B, C, H, W]
        intrinsics: [B, 3, 3]
        pose: [B, 4, 4]
        depth: [B, D, H, W]
    
    Returns:
        warped_feature: [B, C, D, H, W]
    """

    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()
    c = feature1.size(1)

    with torch.no_grad():
        # pixel coordinates
        grid = coords_grid(b, h, w, homogeneous=True, device=depth.device)  # [B, 3, H, W]
        # back project to 3D and transform viewpoint
        points = torch.inverse(intrinsics).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
            1, 1, d, 1) * depth.view(b, 1, d, h * w)  # [B, 3, D, H*W]
        points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        # reproject to 2D image plane
        points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(b, 3, d, h * w)  # [B, 3, D, H*W]
        pixel_coords = points[:, :2] / points[:, -1:].clamp(min=clamp_min_depth)  # [B, 2, D, H*W]

        # normalize to [-1, 1]
        x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
        y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

    # sample features
    warped_feature = F.grid_sample(feature1, grid.view(b, d * h, w, 2), mode='bilinear',
                                   padding_mode='zeros',
                                   align_corners=True).view(b, c, d, h, w)  # [B, C, D, H, W]

    return warped_feature


def correlation_softmax_depth(feature0, feature1,
                              intrinsics,
                              pose,
                              depth_candidates,
                              depth_from_argmax=False,
                              pred_bidir_depth=False,
                              ):
    """
    Compute depth estimation using correlation and softmax.
    
    This function computes depth by warping feature1 to feature0's viewpoint
    using multiple depth candidates, computing correlation scores, and 
    selecting the most likely depth.
    
    Args:
        feature0: [B, C, H, W] - Features from first view
        feature1: [B, C, H, W] - Features from second view
        intrinsics: [B, 3, 3] - Camera intrinsic matrix
        pose: [B, 4, 4] - Relative pose transformation matrix
        depth_candidates: [B, D, H, W] - Depth candidates (inverse depth)
        depth_from_argmax: bool - If True, use argmax; if False, use weighted sum
        pred_bidir_depth: bool - If True, predict bidirectional depth
    
    Returns:
        depth: [B, 1, H, W] - Estimated depth
        match_prob: [B, D, H, W] - Matching probability for each depth candidate
    """
    b, c, h, w = feature0.size()
    assert depth_candidates.dim() == 4  # [B, D, H, W]
    scale_factor = c ** 0.5

    if pred_bidir_depth:
        feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat((feature1, feature0), dim=0)
        intrinsics = intrinsics.repeat(2, 1, 1)
        pose = torch.cat((pose, torch.inverse(pose)), dim=0)
        depth_candidates = depth_candidates.repeat(2, 1, 1, 1)

    # depth candidates are actually inverse depth
    warped_feature1 = warp_with_pose_depth_candidates(feature1, intrinsics, pose,
                                                      1. / depth_candidates,
                                                      )  # [B, C, D, H, W]

    correlation = (feature0.unsqueeze(2) * warped_feature1).sum(1) / scale_factor  # [B, D, H, W]

    match_prob = F.softmax(correlation, dim=1)  # [B, D, H, W]

    # for cross-task transfer (flow -> depth), extract depth with argmax at test time
    if depth_from_argmax:
        index = torch.argmax(match_prob, dim=1, keepdim=True)
        depth = torch.gather(depth_candidates, dim=1, index=index)
    else:
        depth = (match_prob * depth_candidates).sum(dim=1, keepdim=True)  # [B, 1, H, W]

    return depth, match_prob

