
"""
LLM Implementation Template for equirectangular_unwrap_topK_opacity
Replace the code between the markers with your LLM-generated implementation.
"""

import numpy as np


def equirectangular_unwrap_topK_opacity(points, opacity, height=512, width=512, K=4):
    """
    Unwrap a point cloud onto an equirectangular (lat-lon) image.

    Args:
        points (np.array): Input point cloud, shape (N,3), where N is #points.
                           Each row is (x, y, z).
        opacity (np.array): Opacity (or intensity) values per point, shape (N,).
        height (int): Vertical resolution of the output image.
        width (int):  Horizontal resolution of the output image.
        K (int): Number of top-opacity points to keep per pixel.

    Returns:
        np.array: An integer image array of shape (height, width, K),
                  containing point indices for the top K opacity values in each pixel.
                  Pixels with fewer than K points have some default fill (0 or no points).
    """
    # ============================================================================
    # INSERT LLM-GENERATED CODE HERE
    # ============================================================================
    

    # ============================================================================
    # END OF LLM-GENERATED CODE
    # ============================================================================
    
    return image
