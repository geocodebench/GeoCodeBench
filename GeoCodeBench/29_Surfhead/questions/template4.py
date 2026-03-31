
"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
"""

import torch


def compute_face_adjacency(self):
    """Compute face adjacency from edges.
    
    This function computes which faces are adjacent to each other based on shared edges.
    Two faces are adjacent if they share an edge (two vertices).
    
    Input (via self):
        self.faces: Tensor of shape (num_faces, 3)
                   Each row contains 3 vertex indices defining a triangular face
    
    Output (stored in self):
        self.face_adjacency: Tensor of shape (num_faces, 4)
                            For each face, stores [self_index, adj1, adj2, adj3]
                            where adj1-3 are indices of adjacent faces
                            If a face has fewer than 3 neighbors, duplicates its own index
    
    """
    num_faces = self.faces.size(0)
    edge_to_faces = {}
    # faces = self.faces
    def add_edge(face_idx, v1, v2):
        edge = tuple(sorted([v1.item(), v2.item()]))
        if edge not in edge_to_faces:
            edge_to_faces[edge] = []
        edge_to_faces[edge].append(face_idx)
    # Add all edges to the dictionary
    # TODO: Fill in LLM-generated code here
    
    raise NotImplementedError("Please implement the missing part")
