
"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
"""

import torch


def linear_match(d0, d1, mask, patch_size):
    """Linear match between two depth maps.
    
    Args:
        d0: First depth map, shape [1, H, W]
        d1: Second depth map (reference), shape [1, H, W]
        mask: Valid pixel mask, shape [1, H, W]
        patch_size: Size of patches for matching (int)
    
    Returns:
        d0_: Aligned depth map, shape [1, H, W]
    """
    d0 = d0.detach()
    d1 = d1.detach()
    mask = mask.detach()

    ****EMPTY****
    return d0_
