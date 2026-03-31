"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
"""

import numpy as np


def compute_epipolar_distance(T_21, K, p_1, p_2):
    """
    Compute the epipolar distance between corresponding points in two images.
    
    Args:
        T_21: 4x4 transformation matrix from camera 1 to camera 2
        K: 3x3 intrinsic camera matrix
        p_1: 3xN homogeneous coordinates of points in image 1
        p_2: 3xN homogeneous coordinates of points in image 2
    
    Returns:
        geometric_e_distance: 1D array of epipolar distances for each point pair
    """
    # TODO: Fill in LLM-generated code here
    
    raise NotImplementedError("Please implement this function")


def skew(x):
    """Compute the skew-symmetric matrix of a 3D vector."""
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])
