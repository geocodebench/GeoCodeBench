"""
Reference Implementation for wahba() function
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch


def wahba(source_points, target_points, device_=None):
    """
    inputs:
    source_points: torch.tensor of shape (B, 3, N)
    target_points: torch.tensor of shape (B, 3, N)

    where
        B = batch size
        N = number of points in each point set

    output:
    R   : torch.tensor of shape (B, 3, 3)
    """
    with torch.cuda.amp.autocast(enabled=False):
        batch_size = source_points.shape[0]

        if device_ == None:
            device_ = source_points.device

        mat = target_points @ source_points.transpose(-1, -2)  # (B, 3, 3)
        U, S, Vh = torch.linalg.svd(mat)

        D = torch.eye(3).to(device=device_).to(dtype=source_points.dtype)  # (3, 3)
        D = D.unsqueeze(0)  # (1, 3, 3)
        D = D.repeat(batch_size, 1, 1)  # (B, 3, 3)

        D[:, 2, 2] = torch.linalg.det(U) * torch.linalg.det(Vh)

    return U @ D @ Vh  # (B, 3, 3)
