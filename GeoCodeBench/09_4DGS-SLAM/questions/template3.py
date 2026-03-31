"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
"""

import torch


def image_gradient(image):
    """Compute image gradient using Scharr Filter.
    
    Args:
        image: Input image tensor of shape (C, H, W) where C is channels, H is height, W is width.
    
    Returns:
        tuple: (img_grad_v, img_grad_h) - vertical and horizontal gradients
               img_grad_v: vertical gradient of shape (C, H, W), img_grad_v[:,0,0] is positive.
               img_grad_h: horizontal gradient of shape (C, H, W)
    """
    # TODO: Fill in LLM-generated code here
    
    raise NotImplementedError("Please implement this function")
