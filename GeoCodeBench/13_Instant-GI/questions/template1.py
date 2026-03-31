
"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
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
    # TODO: Fill in LLM-generated code here
    
    raise NotImplementedError("Please implement this function")
