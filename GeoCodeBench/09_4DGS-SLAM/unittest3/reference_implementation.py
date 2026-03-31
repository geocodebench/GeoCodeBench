"""
Reference Implementation for image_gradient
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch


def image_gradient(image):
    """Compute image gradient using Scharr Filter.
    
    Args:
        image: Input image tensor of shape (C, H, W) where C is channels, H is height, W is width.
    
    Returns:
        tuple: (img_grad_v, img_grad_h) - vertical and horizontal gradients
               img_grad_v: vertical gradient of shape (C, H, W)
               img_grad_h: horizontal gradient of shape (C, H, W)
    """
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device=image.device
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device=image.device
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]
