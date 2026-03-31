"""
Reference Implementation for arun_batched()
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch


def project_SO3(A):
    """Project a batched matrix to SO(3)"""
    U, S, Vh = torch.linalg.svd(A)
    d = torch.linalg.det(U) * torch.linalg.det(Vh)
    temp = U.clone()
    temp[:, :, -1] *= d[:, None]
    R = temp @ Vh
    return R, U, S, Vh, d


def arun_batched(source_points, target_points):
    """Run Arun's algorithm to estimate R, t (unweighted)
    target = R*source + t

    Note: this is a batched version
    """
    og_dtype = source_points.dtype
    assert source_points.ndim == target_points.ndim == 3

    N = source_points.shape[2]
    # centroids
    source_points_ave = torch.sum(source_points, dim=2) / N
    target_points_ave = torch.sum(target_points, dim=2) / N

    # getting the rotation
    source_points_centered = source_points - source_points_ave.unsqueeze(-1)  # (B, 3, N)
    target_points_centered = target_points - target_points_ave.unsqueeze(-1)  # (B, 3, N)

    # get rotation
    mat = target_points_centered @ source_points_centered.transpose(-1, -2) / N  # (B, 3, 3)
    R, _, S, _, d = project_SO3(mat)
    # S.clone() is necessary for avoiding backprop error due to in-place operations
    D = S.clone()
    D[:, -1] *= d

    # getting the translation
    t = target_points_ave.unsqueeze(-1) - R @ source_points_ave.unsqueeze(-1)

    return R.to(og_dtype), t.to(og_dtype)
