"""
Reference Implementation for sample_operation
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch
from torch.nn import functional as F


def sample_operation(triangles, cir_centers, feature_map, image):
    """Sample features and colors from triangles and circle centers.
    
    Args:
        triangles: [N, 3, 2] - Triangle vertices in normalized coordinates [-1, 1]
        cir_centers: [N, 1, 2] - Circle centers in normalized coordinates [-1, 1]
        feature_map: [1, C, H, W] - Feature map with C channels
        image: [1, 3, H, W] - RGB image
    
    Returns:
        map_feature: [N, 4, C] - Concatenated feature from triangles and centers
        color_feature: [N, 4, 3] - Concatenated color from triangles and centers
    """
    tri_feature = F.grid_sample(
        feature_map,
        triangles.unsqueeze(0),
        mode="bilinear",
        align_corners=False
    )  # [1, 64, N, 3]
    tri_color = F.grid_sample(
        image,
        triangles.unsqueeze(0),
        mode="bilinear",
        align_corners=False
    )  # [1, 3, N, 3]
    center_color = F.grid_sample(
        image,
        cir_centers.unsqueeze(0),
        mode="bilinear",
        align_corners=False
    )  # [1, 3, N, 1]
    center_feature = F.grid_sample(
        feature_map,
        cir_centers.unsqueeze(0),
        mode="bilinear",
        align_corners=False
    )  # [1, 64, N, 1]

    tri_feature = tri_feature.squeeze(0).permute(1, 2, 0)  # [N, 3, 64]
    tri_color = tri_color.squeeze(0).permute(1, 2, 0)  # [N, 3, 3]
    center_color = center_color.squeeze(0).permute(1, 2, 0)  # [N, 1, 3]
    center_feature = center_feature.squeeze(0).permute(1, 2, 0)  # [N, 1, 64]
    map_feature = torch.cat((tri_feature, center_feature), dim=1)  # [N, 4, 64]
    color_feature = torch.cat((tri_color, center_color), dim=1)  # [N, 4, 3]
    return map_feature, color_feature

