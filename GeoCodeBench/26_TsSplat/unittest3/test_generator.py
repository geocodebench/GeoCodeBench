"""
Test Data Generator for _compute_vertex_normal and _compute_vertex_tangent.
Generates test cases with different mesh configurations.
"""

import torch

from reference_implementation import _compute_vertex_normal as ref_compute_vertex_normal


class MockObject:
    """Mock object to hold mesh data for testing."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestDataGenerator:
    """Generate test data for vertex normal and tangent computation."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Simple cube-like mesh (8 vertices, 12 triangles)
        num_vertices = 8
        num_faces = 12
        v_pos = torch.randn(num_vertices, 3)
        t_pos_idx = torch.randint(0, num_vertices, (num_faces, 3), dtype=torch.long)

        num_tex_vertices = num_vertices
        v_tex = torch.rand(num_tex_vertices, 2) * 0.8 + 0.1
        t_tex_idx = t_pos_idx.clone()

        mock_obj = MockObject(v_pos=v_pos, t_pos_idx=t_pos_idx)
        v_nrm = ref_compute_vertex_normal(mock_obj)

        test_cases.append({
            'v_pos': v_pos,
            't_pos_idx': t_pos_idx,
            'v_tex': v_tex,
            't_tex_idx': t_tex_idx,
            'v_nrm': v_nrm,
            'description': f'Basic: {num_vertices} vertices, {num_faces} triangles',
        })

        if num_tests > 1:
            num_vertices = 10
            num_faces = 16
            v_pos = torch.randn(num_vertices, 3) * 2.0
            t_pos_idx = torch.randint(0, num_vertices, (num_faces, 3), dtype=torch.long)
            v_tex = torch.rand(num_vertices, 2) * 0.8 + 0.1
            t_tex_idx = t_pos_idx.clone()

            mock_obj = MockObject(v_pos=v_pos, t_pos_idx=t_pos_idx)
            v_nrm = ref_compute_vertex_normal(mock_obj)

            test_cases.append({
                'v_pos': v_pos,
                't_pos_idx': t_pos_idx,
                'v_tex': v_tex,
                't_tex_idx': t_tex_idx,
                'v_nrm': v_nrm,
                'description': f'Small: {num_vertices} vertices, {num_faces} triangles',
            })

        if num_tests > 2:
            num_vertices = 50
            num_faces = 80
            v_pos = torch.randn(num_vertices, 3) * 5.0
            t_pos_idx = torch.randint(0, num_vertices, (num_faces, 3), dtype=torch.long)
            v_tex = torch.rand(num_vertices, 2) * 0.8 + 0.1
            t_tex_idx = t_pos_idx.clone()

            mock_obj = MockObject(v_pos=v_pos, t_pos_idx=t_pos_idx)
            v_nrm = ref_compute_vertex_normal(mock_obj)

            test_cases.append({
                'v_pos': v_pos,
                't_pos_idx': t_pos_idx,
                'v_tex': v_tex,
                't_tex_idx': t_tex_idx,
                'v_nrm': v_nrm,
                'description': f'Medium: {num_vertices} vertices, {num_faces} triangles',
            })

        if num_tests > 3:
            num_vertices = 100
            num_faces = 150
            num_tex_vertices = 120
            v_pos = torch.randn(num_vertices, 3) * 3.0
            t_pos_idx = torch.randint(0, num_vertices, (num_faces, 3), dtype=torch.long)
            v_tex = torch.rand(num_tex_vertices, 2) * 0.8 + 0.1
            t_tex_idx = torch.randint(0, num_tex_vertices, (num_faces, 3), dtype=torch.long)

            mock_obj = MockObject(v_pos=v_pos, t_pos_idx=t_pos_idx)
            v_nrm = ref_compute_vertex_normal(mock_obj)

            test_cases.append({
                'v_pos': v_pos,
                't_pos_idx': t_pos_idx,
                'v_tex': v_tex,
                't_tex_idx': t_tex_idx,
                'v_nrm': v_nrm,
                'description': f'Large: {num_vertices} vertices, {num_faces} triangles, different tex indices',
            })

        if num_tests > 4:
            num_vertices = 200
            num_faces = 350
            v_pos = torch.randn(num_vertices, 3) * 10.0
            t_pos_idx = torch.randint(0, num_vertices, (num_faces, 3), dtype=torch.long)
            v_tex = torch.rand(num_vertices, 2) * 0.8 + 0.1
            t_tex_idx = t_pos_idx.clone()

            mock_obj = MockObject(v_pos=v_pos, t_pos_idx=t_pos_idx)
            v_nrm = ref_compute_vertex_normal(mock_obj)

            test_cases.append({
                'v_pos': v_pos,
                't_pos_idx': t_pos_idx,
                'v_tex': v_tex,
                't_tex_idx': t_tex_idx,
                'v_nrm': v_nrm,
                'description': f'Very large: {num_vertices} vertices, {num_faces} triangles',
            })

        if num_tests > 5:
            num_vertices = 20
            num_faces = 40
            v_pos = torch.randn(num_vertices, 3) * 2.0
            t_pos_idx = torch.randint(0, num_vertices, (num_faces, 3), dtype=torch.long)
            v_tex = torch.rand(num_vertices, 2) * 0.8 + 0.1
            t_tex_idx = t_pos_idx.clone()

            mock_obj = MockObject(v_pos=v_pos, t_pos_idx=t_pos_idx)
            v_nrm = ref_compute_vertex_normal(mock_obj)

            test_cases.append({
                'v_pos': v_pos,
                't_pos_idx': t_pos_idx,
                'v_tex': v_tex,
                't_tex_idx': t_tex_idx,
                'v_nrm': v_nrm,
                'description': f'Many faces: {num_vertices} vertices, {num_faces} triangles',
            })

        if num_tests > 6:
            num_vertices = 120
            num_faces = 200
            v_pos = torch.randn(num_vertices, 3) * 4.0
            t_pos_idx = torch.randint(0, num_vertices, (num_faces, 3), dtype=torch.long)
            v_tex = torch.rand(num_vertices, 2) * 0.8 + 0.1
            t_tex_idx = t_pos_idx.clone()

            mock_obj = MockObject(v_pos=v_pos, t_pos_idx=t_pos_idx)
            v_nrm = ref_compute_vertex_normal(mock_obj)

            test_cases.append({
                'v_pos': v_pos,
                't_pos_idx': t_pos_idx,
                'v_tex': v_tex,
                't_tex_idx': t_tex_idx,
                'v_nrm': v_nrm,
                'description': f'Medium-large: {num_vertices} vertices, {num_faces} triangles',
            })

        if num_tests > 7:
            num_vertices = 80
            num_faces = 120
            num_tex_vertices = 100
            v_pos = torch.randn(num_vertices, 3) * 3.0
            t_pos_idx = torch.randint(0, num_vertices, (num_faces, 3), dtype=torch.long)
            v_tex = torch.rand(num_tex_vertices, 2) * 0.8 + 0.1
            t_tex_idx = torch.randint(0, num_tex_vertices, (num_faces, 3), dtype=torch.long)

            mock_obj = MockObject(v_pos=v_pos, t_pos_idx=t_pos_idx)
            v_nrm = ref_compute_vertex_normal(mock_obj)

            test_cases.append({
                'v_pos': v_pos,
                't_pos_idx': t_pos_idx,
                'v_tex': v_tex,
                't_tex_idx': t_tex_idx,
                'v_nrm': v_nrm,
                'description': f'Different UV: {num_vertices} vertices, {num_faces} triangles, {num_tex_vertices} tex vertices',
            })

        if num_tests > 8:
            num_vertices = 40
            num_faces = 70
            v_pos = torch.randn(num_vertices, 3) * 2.5
            t_pos_idx = torch.randint(0, num_vertices, (num_faces, 3), dtype=torch.long)
            v_tex = torch.rand(num_vertices, 2) * 0.7 + 0.15
            t_tex_idx = t_pos_idx.clone()

            mock_obj = MockObject(v_pos=v_pos, t_pos_idx=t_pos_idx)
            v_nrm = ref_compute_vertex_normal(mock_obj)

            test_cases.append({
                'v_pos': v_pos,
                't_pos_idx': t_pos_idx,
                'v_tex': v_tex,
                't_tex_idx': t_tex_idx,
                'v_nrm': v_nrm,
                'description': f'Medium-small: {num_vertices} vertices, {num_faces} triangles',
            })

        if num_tests > 9:
            num_vertices = 60
            num_faces = 100
            v_pos = torch.randn(num_vertices, 3) * 3.5
            t_pos_idx = torch.randint(0, num_vertices, (num_faces, 3), dtype=torch.long)
            v_tex = torch.rand(num_vertices, 2) * 0.7 + 0.15
            t_tex_idx = t_pos_idx.clone()

            mock_obj = MockObject(v_pos=v_pos, t_pos_idx=t_pos_idx)
            v_nrm = ref_compute_vertex_normal(mock_obj)

            test_cases.append({
                'v_pos': v_pos,
                't_pos_idx': t_pos_idx,
                'v_tex': v_tex,
                't_tex_idx': t_tex_idx,
                'v_nrm': v_nrm,
                'description': f'Balanced: {num_vertices} vertices, {num_faces} triangles',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            num_vertices = 30 + i * 10
            num_faces = 40 + i * 15
            v_pos = torch.randn(num_vertices, 3) * (2.0 + i * 0.5)
            t_pos_idx = torch.randint(0, num_vertices, (num_faces, 3), dtype=torch.long)
            v_tex = torch.rand(num_vertices, 2) * 0.8 + 0.1
            t_tex_idx = t_pos_idx.clone()

            mock_obj = MockObject(v_pos=v_pos, t_pos_idx=t_pos_idx)
            v_nrm = ref_compute_vertex_normal(mock_obj)

            test_cases.append({
                'v_pos': v_pos,
                't_pos_idx': t_pos_idx,
                'v_tex': v_tex,
                't_tex_idx': t_tex_idx,
                'v_nrm': v_nrm,
                'description': f'Additional test {i+1}: {num_vertices} vertices, {num_faces} triangles',
            })

        return test_cases[:num_tests]
