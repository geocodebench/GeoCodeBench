"""
LLM Implementation Template for _isect_tiles and _isect_offset_encode
Fill in the ****EMPTY**** sections with your implementation.
"""

import numpy as np


def spherical_unwrap_opacity(points, opacity, height=256, width=256):
    """
    Unwrap a point cloud onto a spherical surface.

    Args:
    points (np.array): Input point cloud array of shape (N, 3), where N is the number of points.
    opacity (np.array): Opacity values for each point of shape (N,).
    height (int): The vertical resolution of the output image.
    width (int): The horizontal resolution of the output image.

    Returns:
    np.array: Unwrapped image of the spherical projection with shape (height, width, 3).
              Each pixel contains the index of the point with highest opacity at that location.
    """
    # ============================================================================
    # INSERT YOUR CODE HERE
    # ============================================================================
    
    ****EMPTY****
    
    # ============================================================================
    # END OF YOUR CODE
    # ============================================================================
    
    return image
