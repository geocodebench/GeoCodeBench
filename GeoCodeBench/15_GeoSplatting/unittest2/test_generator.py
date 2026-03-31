"""
Test Data Generator for make() function.
Generates test cases with different mesh configurations.
"""

from __future__ import annotations

import torch

from reference_implementation import TriangleMesh, safe_normalize


def create_simple_mesh(num_faces: int = 4, device='cpu') -> TriangleMesh:
    """Create a simple triangle mesh for testing."""
    torch.manual_seed(42)

    # Create vertices - make sure we have enough unique vertices
    # For num_faces triangles, we need at least some vertices (can share)
    num_vertices = max(num_faces + 1, 4)
    vertices = torch.randn(num_vertices, 3, device=device) * 0.5

    # Create indices for triangles (each triangle uses 3 vertices, can share)
    indices = []
    for i in range(num_faces):
        # Use indices that wrap around, ensuring valid triangles
        v0 = i % num_vertices
        v1 = (i + 1) % num_vertices
        v2 = (i + 2) % num_vertices
        indices.append([v0, v1, v2])
    indices = torch.tensor(indices, device=device, dtype=torch.long)

    # Compute normals from triangles
    p0 = vertices[indices[:, 0]]
    p1 = vertices[indices[:, 1]]
    p2 = vertices[indices[:, 2]]
    face_normals = (p1 - p0).cross(p2 - p0, dim=-1)
    face_normals = safe_normalize(face_normals)

    # Create vertex normals (average of face normals)
    vertex_normals = torch.zeros_like(vertices)
    for i, face_idx in enumerate(indices):
        vertex_normals[face_idx[0]] += face_normals[i]
        vertex_normals[face_idx[1]] += face_normals[i]
        vertex_normals[face_idx[2]] += face_normals[i]
    vertex_normals = safe_normalize(vertex_normals)

    # Create TriangleMesh using our simplified class
    mesh = TriangleMesh(
        vertices=vertices,
        indices=indices,
        normals=vertex_normals,
    )

    return mesh


class TestDataGenerator:
    """Generate test data for make() function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)
        self.device = 'cpu'  # No CUDA as per requirement

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case - small mesh, with normal interpolation
        mesh = create_simple_mesh(num_faces=4, device=self.device)
        num_faces = mesh.num_faces if hasattr(mesh, 'num_faces') else len(mesh.indices)
        test_cases.append({
            'mesh': mesh,
            'normal_interpolation': True,
            'description': f'Basic: {num_faces} faces, normal_interpolation=True',
        })

        if num_tests > 1:
            # Test 2: Without normal interpolation
            mesh = create_simple_mesh(num_faces=4, device=self.device)
            num_faces = mesh.num_faces if hasattr(mesh, 'num_faces') else len(mesh.indices)
            test_cases.append({
                'mesh': mesh,
                'normal_interpolation': False,
                'description': f'No normal interpolation: {num_faces} faces, normal_interpolation=False',
            })

        if num_tests > 2:
            # Test 3: Medium mesh
            mesh = create_simple_mesh(num_faces=8, device=self.device)
            num_faces = mesh.num_faces if hasattr(mesh, 'num_faces') else len(mesh.indices)
            test_cases.append({
                'mesh': mesh,
                'normal_interpolation': True,
                'description': f'Medium mesh: {num_faces} faces, normal_interpolation=True',
            })

        if num_tests > 3:
            # Test 4: Large mesh
            mesh = create_simple_mesh(num_faces=16, device=self.device)
            num_faces = mesh.num_faces if hasattr(mesh, 'num_faces') else len(mesh.indices)
            test_cases.append({
                'mesh': mesh,
                'normal_interpolation': True,
                'description': f'Large mesh: {num_faces} faces, normal_interpolation=True',
            })

        if num_tests > 4:
            # Test 5: Single triangle
            mesh = create_simple_mesh(num_faces=1, device=self.device)
            num_faces = mesh.num_faces if hasattr(mesh, 'num_faces') else len(mesh.indices)
            test_cases.append({
                'mesh': mesh,
                'normal_interpolation': True,
                'description': f'Single triangle: {num_faces} faces, normal_interpolation=True',
            })

        # Generate additional tests if needed
        for i in range(max(0, num_tests - len(test_cases))):
            num_faces_val = 2 + (i % 5) * 2
            mesh = create_simple_mesh(num_faces=num_faces_val, device=self.device)
            num_faces = mesh.num_faces if hasattr(mesh, 'num_faces') else len(mesh.indices)
            test_cases.append({
                'mesh': mesh,
                'normal_interpolation': i % 2 == 0,
                'description': f'Additional test {i+1}: {num_faces} faces, normal_interpolation={i % 2 == 0}',
            })

        return test_cases[:num_tests]
