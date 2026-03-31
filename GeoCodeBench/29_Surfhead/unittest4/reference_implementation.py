"""
Reference Implementation for compute_face_adjacency
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch


class MockFlameGaussianModel:
    """Mock class to test compute_face_adjacency method."""
    
    def __init__(self, faces):
        """Initialize with faces tensor.
        
        Args:
            faces: Tensor of shape (num_faces, 3) containing vertex indices for each face
        """
        self.faces = faces
        self.face_adjacency = None
    
    def compute_face_adjacency(self):
        """Compute face adjacency from edges.
        
        This method computes which faces are adjacent (share an edge) to each face.
        The result is stored in self.face_adjacency with shape (num_faces, 4).
        
        For each face, the adjacency tensor stores:
        - [0]: the face index itself
        - [1-3]: indices of up to 3 adjacent faces (or the face itself if fewer than 3 neighbors)
        """
        num_faces = self.faces.size(0)
        edge_to_faces = {}
        
        def add_edge(face_idx, v1, v2):
            edge = tuple(sorted([v1.item(), v2.item()]))
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(face_idx)
        
        # Add all edges to the dictionary
        for i, face in enumerate(self.faces):
            add_edge(i, face[0], face[1])
            add_edge(i, face[1], face[2])
            add_edge(i, face[2], face[0])
        
        # Initialize adjacency tensor
        face_adjacency = torch.full((num_faces, 3), -1, dtype=torch.long)
        
        # Populate adjacency tensor
        for edge, face_list in edge_to_faces.items():
            if len(face_list) > 1:
                for face in face_list:
                    adj_faces = set(face_list) - {face}
                    for adj_face in adj_faces:
                        if -1 in face_adjacency[face]:
                            idx = (face_adjacency[face] == -1).nonzero(as_tuple=True)[0][0]
                            face_adjacency[face, idx] = adj_face
        
        face_adjacency_identity = torch.arange(face_adjacency.shape[0])
        face_adjacency[face_adjacency == -1] = face_adjacency_identity.view(-1, 1).repeat(1, 3)[face_adjacency == -1]
        
        self.face_adjacency = torch.cat([face_adjacency_identity[..., None], face_adjacency], -1)


def compute_face_adjacency(self):
    """Wrapper function for testing."""
    model = MockFlameGaussianModel(self.faces)
    model.compute_face_adjacency()
    self.face_adjacency = model.face_adjacency

