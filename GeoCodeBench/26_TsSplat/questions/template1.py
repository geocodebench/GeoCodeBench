
"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
"""

import numpy as np
import torch


def compute_G_matrix(verts_init, faces):
    """Compute gradient operator matrix for tetrahedral mesh.
    
    Args:
        verts_init: Initial vertex positions, shape (V, 3) where V is number of vertices.
                   Can be numpy array or torch tensor.
        faces: Tetrahedral face indices, shape (T, 4) where T is number of tetrahedra.
               Each row contains 4 vertex indices forming a tetrahedron.
    
    Returns:
        G: Gradient operator matrix, shape (T, 9, 12), as numpy array.
    """
    # compute gradient operator matrix
    # TODO: Fill in LLM-generated code here
    
    raise NotImplementedError("Please implement this function")

    return G.numpy()  # T x 9 x 12
