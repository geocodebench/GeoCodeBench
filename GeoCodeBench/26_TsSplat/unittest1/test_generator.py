"""
Test Data Generator for compute_G_matrix function.
Generates test cases with different tetrahedral mesh configurations.
"""

import numpy as np
import torch


class TestDataGenerator:
    """Generate test data for compute_G_matrix function."""

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _generate_valid_tetrahedra(self, num_verts, num_tets, scale=1.0):
        """Generate valid (non-degenerate) tetrahedra."""
        verts = np.random.randn(num_verts, 3).astype(np.float64) * scale
        faces = []

        # Generate valid tetrahedra by ensuring vertices are not coplanar
        for _ in range(num_tets):
            # Randomly select 4 vertices
            tet_verts = np.random.choice(num_verts, 4, replace=False)
            faces.append(tet_verts)

        return verts, np.array(faces, dtype=np.int64)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with single tetrahedron
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2, 3]], dtype=np.int64)
        test_cases.append({
            'verts_init': verts,
            'faces': faces,
            'description': 'Basic: single regular tetrahedron',
        })

        if num_tests > 1:
            # Test 2: Multiple tetrahedra
            verts, faces = self._generate_valid_tetrahedra(10, 3, scale=2.0)
            test_cases.append({
                'verts_init': verts,
                'faces': faces,
                'description': 'Multiple tetrahedra: 3 tets, 10 vertices',
            })

        if num_tests > 2:
            # Test 3: Random mesh with more tetrahedra
            verts, faces = self._generate_valid_tetrahedra(20, 5, scale=2.0)
            test_cases.append({
                'verts_init': verts,
                'faces': faces,
                'description': f'Random mesh: {faces.shape[0]} tets, {verts.shape[0]} vertices',
            })

        if num_tests > 3:
            # Test 4: Input as torch tensor
            verts, faces = self._generate_valid_tetrahedra(15, 4, scale=3.0)
            verts = torch.from_numpy(verts).to(torch.float64)
            test_cases.append({
                'verts_init': verts,
                'faces': faces,
                'description': 'Torch tensor input: 4 tets, 15 vertices',
            })

        if num_tests > 4:
            # Test 5: Larger mesh
            verts, faces = self._generate_valid_tetrahedra(50, 10, scale=5.0)
            test_cases.append({
                'verts_init': verts,
                'faces': faces,
                'description': f'Large mesh: {faces.shape[0]} tets, {verts.shape[0]} vertices',
            })

        if num_tests > 5:
            # Test 6: Regular grid-like structure - use valid tetrahedra
            verts = np.array([
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]
            ], dtype=np.float64)
            faces = np.array([
                [0, 1, 2, 3],
                [1, 5, 3, 7],
                [2, 6, 3, 7]
            ], dtype=np.int64)
            test_cases.append({
                'verts_init': verts,
                'faces': faces,
                'description': 'Grid structure: 3 tets in unit cube',
            })

        if num_tests > 6:
            # Test 7: Scaled vertices
            verts, faces = self._generate_valid_tetrahedra(25, 6, scale=10.0)
            test_cases.append({
                'verts_init': verts,
                'faces': faces,
                'description': 'Scaled mesh: 6 tets, vertices scaled by 10',
            })

        if num_tests > 7:
            # Test 8: Small perturbations
            verts = np.array([
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                [2.0, 0.1, 0.1], [1.9, 1.0, 0.1], [2.1, 0.0, 1.0]
            ], dtype=np.float64)
            faces = np.array([
                [0, 1, 2, 3],
                [1, 4, 5, 6]
            ], dtype=np.int64)
            test_cases.append({
                'verts_init': verts,
                'faces': faces,
                'description': 'Perturbed mesh: 2 tets with small perturbations',
            })

        if num_tests > 8:
            # Test 9: Many tetrahedra
            verts, faces = self._generate_valid_tetrahedra(100, 20, scale=3.0)
            test_cases.append({
                'verts_init': verts,
                'faces': faces,
                'description': f'Many tetrahedra: {faces.shape[0]} tets, {verts.shape[0]} vertices',
            })

        if num_tests > 9:
            # Test 10: Very large mesh
            verts, faces = self._generate_valid_tetrahedra(200, 50, scale=8.0)
            test_cases.append({
                'verts_init': verts,
                'faces': faces,
                'description': f'Very large mesh: {faces.shape[0]} tets, {verts.shape[0]} vertices',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            num_verts = 30 + i * 10
            num_tets = 8 + i * 2
            verts, faces = self._generate_valid_tetrahedra(num_verts, num_tets, scale=2.0 + i)

            test_cases.append({
                'verts_init': verts,
                'faces': faces,
                'description': f'Additional test {i+1}: {faces.shape[0]} tets, {verts.shape[0]} vertices',
            })

        return test_cases[:num_tests]
