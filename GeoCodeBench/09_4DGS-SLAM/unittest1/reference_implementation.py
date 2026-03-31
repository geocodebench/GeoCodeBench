"""
Reference Implementation for compute_epipolar_distance
This serves as the ground truth for testing LLM-generated implementations.
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
    R_21 = T_21[:3, :3]
    t_21 = T_21[:3, 3]

    E_mat = np.dot(skew(t_21), R_21)
    # compute bearing vector
    inv_K = np.linalg.inv(K)

    F_mat = np.dot(np.dot(inv_K.T, E_mat), inv_K)

    l_2 = np.dot(F_mat, p_1)
    algebric_e_distance = np.sum(p_2 * l_2, axis=0)
    n_term = np.sqrt(l_2[0, :]**2 + l_2[1, :]**2) + 1e-8
    geometric_e_distance = algebric_e_distance/n_term
    geometric_e_distance = np.abs(geometric_e_distance)

    return geometric_e_distance


def skew(x):
    """Compute the skew-symmetric matrix of a 3D vector."""
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])
