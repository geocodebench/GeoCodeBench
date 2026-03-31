
"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
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

        ****EMPTY****

    return U @ D @ Vh  # (B, 3, 3)
