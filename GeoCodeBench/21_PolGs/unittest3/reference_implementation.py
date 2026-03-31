"""
Reference Implementation for linear_match
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch


def linear_match(d0, d1, mask, patch_size):
    """Linear match between two depth maps.
    
    This function performs linear matching between two depth maps (d0 and d1) using
    a least-squares approach on patches. It computes the optimal linear transformation
    (scale and shift) for each patch to align d0 with d1.
    
    Args:
        d0: First depth map, shape [1, H, W]
        d1: Second depth map (reference), shape [1, H, W]
        mask: Valid pixel mask, shape [1, H, W]
        patch_size: Size of patches for matching (int)
    
    Returns:
        d0_: Aligned depth map, shape [1, H, W]
    
    The function:
    1. Divides the depth maps into non-overlapping patches
    2. For each patch, solves a least-squares problem: d1 ≈ x_0 * d0 + x_1
    3. Applies the transformation to align d0 to d1
    4. Handles remaining pixels at the borders
    
    Reference: Adapted from MonoSDF (https://github.com/autonomousvision/monosdf/)
    """
    # copy from MonoSDF: https://github.com/autonomousvision/monosdf/
    d0 = d0.detach()
    d1 = d1.detach()
    mask = mask.detach()

    patch_dim = (torch.tensor(d0.shape[1:3]) / patch_size).to(torch.int32)
    patch_num = patch_dim[0] * patch_dim[1]

    comb = torch.cat([d0, d1, mask], 0)
    comb_ = comb[:, :patch_dim[0] * patch_size, :patch_dim[1] * patch_size]
    comb_ = comb_.reshape([3, patch_dim[0], patch_size, patch_dim[1], patch_size])
    comb_ = comb_.permute([0, 1, 3, 2, 4])
    comb_ = comb_.reshape([3, patch_num, patch_size, patch_size])

    d0_ = comb_[0]
    d1_ = comb_[1]
    mask_ = comb_[2]
    a_00 = torch.sum(mask_ * d0_ * d0_, (1, 2))
    a_01 = torch.sum(mask_ * d0_, (1, 2))
    a_11 = torch.sum(mask_, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask_ * d0_ * d1_, (1, 2))
    b_1 = torch.sum(mask_ * d1_, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]


    d0_ = x_0[:, None, None] * d0_ + x_1[:, None, None]
    d0_ = d0_.reshape([1, patch_dim[0], patch_dim[1], patch_size, patch_size])
    d0_ = d0_.permute([0, 1, 3, 2, 4])
    d0_ = d0_.reshape([1, patch_dim[0] * patch_size, patch_dim[1] * patch_size])
    d0_b = d0[:, patch_dim[0] * patch_size:, :patch_dim[1] * patch_size]
    d0_ = torch.cat([d0_, d0_b], 1)
    d0_r = d0[:, :, patch_dim[1] * patch_size:]
    d0_ = torch.cat([d0_, d0_r], 2)
    return d0_

