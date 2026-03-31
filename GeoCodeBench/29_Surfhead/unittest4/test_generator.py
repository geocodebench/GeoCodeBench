"""
Test Data Generator for compute_face_adjacency function.
"""

import torch


class TestDataGenerator:
    """Generate test data for compute_face_adjacency function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different mesh configurations."""
        test_cases = []

        # Test 1: Simple tetrahedron (4 faces)
        faces = torch.tensor([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3]
        ], dtype=torch.long)
        test_cases.append({
            'faces': faces,
            'description': 'Simple tetrahedron (4 faces, fully connected)',
        })

        if num_tests > 1:
            # Test 2: Two separate triangles (no adjacency)
            faces = torch.tensor([
                [0, 1, 2],
                [3, 4, 5]
            ], dtype=torch.long)
            test_cases.append({
                'faces': faces,
                'description': 'Two separate triangles (no shared edges)',
            })

        if num_tests > 2:
            # Test 3: Square (2 triangles sharing an edge)
            faces = torch.tensor([
                [0, 1, 2],
                [0, 2, 3]
            ], dtype=torch.long)
            test_cases.append({
                'faces': faces,
                'description': 'Square divided into 2 triangles (shared edge)',
            })

        if num_tests > 3:
            # Test 4: Pyramid (5 faces)
            faces = torch.tensor([
                [0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4],
            ], dtype=torch.long)
            test_cases.append({
                'faces': faces,
                'description': 'Pyramid (5 faces with complex adjacency)',
            })

        if num_tests > 4:
            # Test 5: Cube (12 faces)
            faces = torch.tensor([
                [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5],
                [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            ], dtype=torch.long)
            test_cases.append({
                'faces': faces,
                'description': 'Cube (12 triangular faces)',
            })

        if num_tests > 5:
            # Test 6: Strip of triangles
            faces = torch.tensor([
                [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5],
            ], dtype=torch.long)
            test_cases.append({
                'faces': faces,
                'description': 'Strip of 4 triangles (linear connectivity)',
            })

        if num_tests > 6:
            # Test 7: Fan of triangles
            faces = torch.tensor([
                [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1],
            ], dtype=torch.long)
            test_cases.append({
                'faces': faces,
                'description': 'Fan of 5 triangles (circular connectivity)',
            })

        if num_tests > 7:
            # Test 8: Octahedron (8 faces)
            faces = torch.tensor([
                [0, 1, 4], [0, 4, 3], [0, 3, 2], [0, 2, 1],
                [5, 1, 4], [5, 4, 3], [5, 3, 2], [5, 2, 1],
            ], dtype=torch.long)
            test_cases.append({
                'faces': faces,
                'description': 'Octahedron (8 faces)',
            })

        if num_tests > 8:
            # Test 9: Single triangle (edge case)
            faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
            test_cases.append({
                'faces': faces,
                'description': 'Single triangle (no adjacent faces)',
            })

        if num_tests > 9:
            # Test 10: Random mesh with 20 faces
            torch.manual_seed(42)
            num_faces = 20
            max_vertex = 15
            faces = torch.randint(0, max_vertex, (num_faces, 3), dtype=torch.long)
            test_cases.append({
                'faces': faces,
                'description': 'Random mesh (20 faces, 15 vertices)',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            torch.manual_seed(42 + i)
            num_faces = 10 + i * 5
            max_vertex = 8 + i * 3
            faces = torch.randint(0, max_vertex, (num_faces, 3), dtype=torch.long)
            test_cases.append({
                'faces': faces,
                'description': f'Additional test {i+1}: {num_faces} faces, {max_vertex} vertices',
            })

        return test_cases[:num_tests]
