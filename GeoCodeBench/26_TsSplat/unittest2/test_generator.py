"""
Test Data Generator for get_surface_vf function.
Generates test cases with different tetrahedral face configurations.
"""

import numpy as np


class TestDataGenerator:
    """Generate test data for get_surface_vf function."""

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Simple single tetrahedron
        faces = np.array([[0, 1, 2, 3]])
        test_cases.append({
            'faces': faces,
            'description': f'Single tetrahedron: 1 tet, 4 vertices',
        })

        if num_tests > 1:
            # Test 2: Two separate tetrahedra
            faces = np.array([
                [0, 1, 2, 3],
                [4, 5, 6, 7]
            ])
            test_cases.append({
                'faces': faces,
                'description': f'Two separate tets: 2 tets, 8 vertices',
            })

        if num_tests > 2:
            # Test 3: Two tetrahedra sharing a face
            faces = np.array([
                [0, 1, 2, 3],
                [0, 1, 2, 4]
            ])
            test_cases.append({
                'faces': faces,
                'description': f'Two adjacent tets: 2 tets sharing a face',
            })

        if num_tests > 3:
            # Test 4: Small mesh
            num_tets = 5
            faces = np.random.randint(0, 15, size=(num_tets, 4))
            test_cases.append({
                'faces': faces,
                'description': f'Small mesh: {num_tets} tets, random connections',
            })

        if num_tests > 4:
            # Test 5: Medium mesh
            num_tets = 20
            max_vertex = 30
            faces = np.random.randint(0, max_vertex, size=(num_tets, 4))
            test_cases.append({
                'faces': faces,
                'description': f'Medium mesh: {num_tets} tets, {max_vertex} vertex range',
            })

        if num_tests > 5:
            # Test 6: Larger mesh with more structure
            num_tets = 50
            max_vertex = 40
            faces = np.random.randint(0, max_vertex, size=(num_tets, 4))
            test_cases.append({
                'faces': faces,
                'description': f'Large mesh: {num_tets} tets, {max_vertex} vertex range',
            })

        if num_tests > 6:
            # Test 7: Dense mesh
            num_tets = 100
            max_vertex = 60
            faces = np.random.randint(0, max_vertex, size=(num_tets, 4))
            test_cases.append({
                'faces': faces,
                'description': f'Dense mesh: {num_tets} tets, {max_vertex} vertex range',
            })

        if num_tests > 7:
            # Test 8: Chain of tetrahedra
            num_tets = 10
            faces = []
            for i in range(num_tets):
                base = i * 2
                faces.append([base, base+1, base+2, base+3])
            faces = np.array(faces)
            test_cases.append({
                'faces': faces,
                'description': f'Chain structure: {num_tets} tets in sequence',
            })

        if num_tests > 8:
            # Test 9: Large random mesh
            num_tets = 200
            max_vertex = 100
            faces = np.random.randint(0, max_vertex, size=(num_tets, 4))
            test_cases.append({
                'faces': faces,
                'description': f'Very large mesh: {num_tets} tets, {max_vertex} vertex range',
            })

        if num_tests > 9:
            # Test 10: Complex connectivity
            num_tets = 150
            max_vertex = 80
            faces = np.random.randint(0, max_vertex, size=(num_tets, 4))
            test_cases.append({
                'faces': faces,
                'description': f'Complex mesh: {num_tets} tets, {max_vertex} vertex range',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            num_tets = 30 + i * 10
            max_vertex = 50 + i * 5
            faces = np.random.randint(0, max_vertex, size=(num_tets, 4))

            test_cases.append({
                'faces': faces,
                'description': f'Additional test {i+1}: {num_tets} tets, {max_vertex} vertex range',
            })

        return test_cases[:num_tests]
