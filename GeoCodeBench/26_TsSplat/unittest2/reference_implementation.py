"""
Reference Implementation for get_surface_vf
This serves as the ground truth for testing LLM-generated implementations.
"""

import numpy as np


def get_surface_vf(faces):
    """Extract surface vertices and faces from tetrahedral mesh.
    
    Args:
        faces: Tetrahedral face indices, shape (N, 4) where N is number of tetrahedra.
               Each row contains 4 vertex indices representing a tetrahedron.
    
    Returns:
        surface_vertices: Array of unique surface vertex indices, shape (M,)
        mapped_triangles: Surface triangular faces with remapped indices, shape (K, 3)
                         where K is number of surface triangles.
    """
    # get surface faces
    org_triangles = np.vstack(
        [
            faces[:, [1, 2, 3]],
            faces[:, [0, 3, 2]],
            faces[:, [0, 1, 3]],
            faces[:, [0, 2, 1]],
        ]
    )

    # Sort each triangle's vertices to avoid duplicates due to ordering
    triangles = np.sort(org_triangles, axis=1)

    unique_triangles, tri_idx, counts = np.unique(
        triangles, axis=0, return_index=True, return_counts=True
    )

    once_tri_id = counts == 1
    surface_triangles = unique_triangles[once_tri_id]

    surface_vertices = np.unique(surface_triangles)

    vertex_mapping = {vertex_id: i for i,
                      vertex_id in enumerate(surface_vertices)}

    mapped_triangles = np.vectorize(vertex_mapping.get)(
        org_triangles[tri_idx][once_tri_id]
    )

    return surface_vertices, mapped_triangles

