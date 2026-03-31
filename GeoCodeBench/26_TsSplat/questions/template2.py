
"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
"""

import numpy as np


def get_surface_vf(faces):
    """Extract surface vertices and faces from tetrahedral mesh.
    
    This function takes a tetrahedral mesh and extracts the surface (boundary) vertices and triangles.
    A triangle is on the surface if it appears only once in the mesh (not shared by two tetrahedra).
    
    Args:
        faces: Tetrahedral face indices, shape (N, 4) where N is number of tetrahedra.
               Each row contains 4 vertex indices representing a tetrahedron.
               For example: [[0, 1, 2, 3], [1, 2, 3, 4], ...]
    
    Returns:
        surface_vertices: Array of unique surface vertex indices, shape (M,)
                         M is the number of vertices that appear on the surface.
                         For example: [0, 1, 2, 3, 5, 7, ...]
        mapped_triangles: Surface triangular faces with remapped indices, shape (K, 3)
                         K is the number of surface triangles.
                         The vertex indices are remapped to the range [0, M-1]
                         For example: [[0, 1, 2], [1, 2, 3], ...]
    """
    # TODO: Fill in LLM-generated code here
    
    raise NotImplementedError("Please implement this function")
