"""
Test Data Generator for compute_E function.
"""

import torch


class TestDataGenerator:
    """Generate test data for compute_E function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Simple triangle
        verts = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        test_cases.append({
            'verts': verts,
            'faces': faces,
            'description': 'Simple: single right triangle',
        })

        if num_tests > 1:
            # Test 2: Multiple triangles
            verts = torch.tensor([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 1.0]
            ], dtype=torch.float32)
            faces = torch.tensor([
                [0, 1, 2],
                [0, 1, 3],
                [1, 2, 3]
            ], dtype=torch.long)
            test_cases.append({
                'verts': verts,
                'faces': faces,
                'description': 'Multiple: 3 triangles from tetrahedron',
            })

        if num_tests > 2:
            # Test 3: Random vertices and faces
            num_verts = 10
            num_faces = 5
            verts = torch.randn(num_verts, 3, dtype=torch.float32)
            faces = torch.randint(0, num_verts, (num_faces, 3), dtype=torch.long)
            test_cases.append({
                'verts': verts,
                'faces': faces,
                'description': f'Random: {num_verts} vertices, {num_faces} faces',
            })

        if num_tests > 3:
            # Test 4: Batch of triangles
            batch_size = 4
            num_verts = 6
            num_faces = 3
            verts = torch.randn(batch_size, num_verts, 3, dtype=torch.float32)
            faces = torch.randint(0, num_verts, (batch_size, num_faces, 3), dtype=torch.long)
            test_cases.append({
                'verts': verts,
                'faces': faces,
                'description': f'Batch: batch_size={batch_size}, {num_faces} faces per batch',
            })

        if num_tests > 4:
            # Test 5: Larger random mesh
            num_verts = 20
            num_faces = 15
            verts = torch.randn(num_verts, 3, dtype=torch.float32) * 5.0
            faces = torch.randint(0, num_verts, (num_faces, 3), dtype=torch.long)
            test_cases.append({
                'verts': verts,
                'faces': faces,
                'description': f'Large: {num_verts} vertices, {num_faces} faces',
            })

        if num_tests > 5:
            # Test 6: Regular grid vertices
            x = torch.linspace(0, 1, 5)
            y = torch.linspace(0, 1, 5)
            z = torch.zeros(1)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            verts = torch.stack([X.flatten(), Y.flatten(), z.expand_as(X.flatten())], dim=-1)
            num_faces = 10
            faces = torch.randint(0, len(verts), (num_faces, 3), dtype=torch.long)
            test_cases.append({
                'verts': verts,
                'faces': faces,
                'description': f'Grid: regular grid with {num_faces} faces',
            })

        if num_tests > 6:
            # Test 7: Scaled vertices
            num_verts = 8
            num_faces = 6
            verts = torch.randn(num_verts, 3, dtype=torch.float32) * 100.0
            faces = torch.randint(0, num_verts, (num_faces, 3), dtype=torch.long)
            test_cases.append({
                'verts': verts,
                'faces': faces,
                'description': f'Scaled: large scale vertices ({num_faces} faces)',
            })

        if num_tests > 7:
            # Test 8: Small scale vertices
            num_verts = 12
            num_faces = 8
            verts = torch.randn(num_verts, 3, dtype=torch.float32) * 0.01
            faces = torch.randint(0, num_verts, (num_faces, 3), dtype=torch.long)
            test_cases.append({
                'verts': verts,
                'faces': faces,
                'description': f'Small scale: tiny vertices ({num_faces} faces)',
            })

        if num_tests > 8:
            # Test 9: Large batch
            batch_size = 8
            num_verts = 10
            num_faces = 5
            verts = torch.randn(batch_size, num_verts, 3, dtype=torch.float32)
            faces = torch.randint(0, num_verts, (batch_size, num_faces, 3), dtype=torch.long)
            test_cases.append({
                'verts': verts,
                'faces': faces,
                'description': f'Large batch: batch_size={batch_size}',
            })

        if num_tests > 9:
            # Test 10: Many faces
            num_verts = 30
            num_faces = 40
            verts = torch.randn(num_verts, 3, dtype=torch.float32)
            faces = torch.randint(0, num_verts, (num_faces, 3), dtype=torch.long)
            test_cases.append({
                'verts': verts,
                'faces': faces,
                'description': f'Many faces: {num_faces} triangular faces',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            num_verts = 10 + i * 2
            num_faces = 5 + i
            if i % 2 == 0:
                verts = torch.randn(num_verts, 3, dtype=torch.float32)
                faces = torch.randint(0, num_verts, (num_faces, 3), dtype=torch.long)
            else:
                batch_size = 2 + i % 3
                verts = torch.randn(batch_size, num_verts, 3, dtype=torch.float32)
                faces = torch.randint(0, num_verts, (batch_size, num_faces, 3), dtype=torch.long)

            test_cases.append({
                'verts': verts,
                'faces': faces,
                'description': f'Additional test {i+1}: {num_verts} verts, {num_faces} faces',
            })

        return test_cases[:num_tests]
