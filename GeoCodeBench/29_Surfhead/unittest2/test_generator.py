"""
Test Data Generator for compute_face_orientation function.
"""

import torch


class TestDataGenerator:
    """Generate test data for compute_face_orientation function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic triangle mesh
        num_verts = 4
        num_faces = 2
        verts = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ], dtype=torch.float32)
        faces = torch.tensor([
            [0, 1, 2],
            [1, 3, 2]
        ], dtype=torch.long)
        test_cases.append({
            'verts': verts,
            'faces': faces,
            'return_scale': True,
            'description': f'Basic: simple 2-face mesh, {num_verts} vertices, {num_faces} faces',
        })

        if num_tests > 1:
            # Test 2: Random mesh
            num_verts = 10
            num_faces = 8
            verts = torch.randn(num_verts, 3) * 2.0
            faces = torch.randint(0, num_verts, (num_faces, 3))
            test_cases.append({
                'verts': verts,
                'faces': faces,
                'return_scale': True,
                'description': f'Random mesh: {num_verts} vertices, {num_faces} faces',
            })

        if num_tests > 2:
            # Test 3: Larger mesh
            num_verts = 50
            num_faces = 30
            verts = torch.randn(num_verts, 3) * 5.0
            faces = torch.randint(0, num_verts, (num_faces, 3))
            test_cases.append({
                'verts': verts,
                'faces': faces,
                'return_scale': True,
                'description': f'Large mesh: {num_verts} vertices, {num_faces} faces',
            })

        if num_tests > 3:
            # Test 4: Batched mesh
            batch_size = 3
            num_verts = 12
            num_faces = 10
            verts = torch.randn(batch_size, num_verts, 3) * 3.0
            faces = torch.randint(0, num_verts, (batch_size, num_faces, 3))
            test_cases.append({
                'verts': verts,
                'faces': faces,
                'return_scale': True,
                'description': f'Batched: batch_size={batch_size}, {num_verts} vertices, {num_faces} faces',
            })

        if num_tests > 4:
            # Test 5: Small triangles
            num_verts = 6
            num_faces = 4
            verts = torch.randn(num_verts, 3) * 0.1
            faces = torch.randint(0, num_verts, (num_faces, 3))
            test_cases.append({
                'verts': verts,
                'faces': faces,
                'return_scale': True,
                'description': f'Small triangles: {num_verts} vertices, {num_faces} faces, small scale',
            })

        if num_tests > 5:
            # Test 6: Large batch
            batch_size = 8
            num_verts = 20
            num_faces = 15
            verts = torch.randn(batch_size, num_verts, 3) * 4.0
            faces = torch.randint(0, num_verts, (batch_size, num_faces, 3))
            test_cases.append({
                'verts': verts,
                'faces': faces,
                'return_scale': True,
                'description': f'Large batch: batch_size={batch_size}, {num_verts} vertices, {num_faces} faces',
            })

        if num_tests > 6:
            # Test 7: 2D batch
            batch_size = (4, 2)
            num_verts = 15
            num_faces = 12
            verts = torch.randn(*batch_size, num_verts, 3) * 2.5
            faces = torch.randint(0, num_verts, (*batch_size, num_faces, 3))
            test_cases.append({
                'verts': verts,
                'faces': faces,
                'return_scale': True,
                'description': f'2D batch: batch_size={batch_size}, {num_verts} vertices, {num_faces} faces',
            })

        if num_tests > 7:
            # Test 8: Many faces
            num_verts = 30
            num_faces = 50
            verts = torch.randn(num_verts, 3) * 3.0
            faces = torch.randint(0, num_verts, (num_faces, 3))
            test_cases.append({
                'verts': verts,
                'faces': faces,
                'return_scale': True,
                'description': f'Many faces: {num_verts} vertices, {num_faces} faces',
            })

        if num_tests > 8:
            # Test 9: Regular grid mesh
            num_verts = 25
            num_faces = 20
            verts = torch.randn(num_verts, 3) * 1.5
            verts[:, 2] = 0  # Make it planar
            faces = torch.randint(0, num_verts, (num_faces, 3))
            test_cases.append({
                'verts': verts,
                'faces': faces,
                'return_scale': True,
                'description': f'Planar mesh: {num_verts} vertices, {num_faces} faces',
            })

        if num_tests > 9:
            # Test 10: 3D batch with many faces
            batch_size = (2, 3, 2)
            num_verts = 18
            num_faces = 15
            verts = torch.randn(*batch_size, num_verts, 3) * 3.5
            faces = torch.randint(0, num_verts, (*batch_size, num_faces, 3))
            test_cases.append({
                'verts': verts,
                'faces': faces,
                'return_scale': True,
                'description': f'3D batch: batch_size={batch_size}, {num_verts} vertices, {num_faces} faces',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            batch_dims = []
            if i % 3 == 0:
                batch_dims = [3 + (i % 5)]
            elif i % 3 == 1:
                batch_dims = [2 + (i % 4), 2]

            num_verts = 10 + (i * 3)
            num_faces = 8 + (i * 2)

            if batch_dims:
                verts = torch.randn(*batch_dims, num_verts, 3) * (2.0 + i * 0.5)
                faces = torch.randint(0, num_verts, (*batch_dims, num_faces, 3))
            else:
                verts = torch.randn(num_verts, 3) * (2.0 + i * 0.5)
                faces = torch.randint(0, num_verts, (num_faces, 3))

            test_cases.append({
                'verts': verts,
                'faces': faces,
                'return_scale': True,
                'description': f'Additional test {i+1}: {num_verts} vertices, {num_faces} faces',
            })

        return test_cases[:num_tests]
