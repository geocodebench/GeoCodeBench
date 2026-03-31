"""
Test Data Generator for stokes_fac_from_normal function.
"""

import torch


class TestDataGenerator:
    """Generate test data for stokes_fac_from_normal function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def _normalize(self, v):
        """Normalize vectors."""
        return v / torch.norm(v, dim=-1, keepdim=True)

    def _random_unit_vectors(self, *shape):
        """Generate random unit vectors."""
        v = torch.randn(*shape, 3)
        return self._normalize(v)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case - single ray
        rays_o = torch.randn(3)
        rays_d = self._random_unit_vectors()
        normal = self._random_unit_vectors()
        test_cases.append({
            'rays_o': rays_o,
            'rays_d': rays_d,
            'normal': normal,
            'train_mode': False,
            'ret_spec': False,
            'clip_spec': False,
            'description': 'Basic: single ray, no clip',
        })

        if num_tests > 1:
            # Test 2: Batch of rays
            batch_size = 10
            rays_o = torch.randn(batch_size, 3)
            rays_d = self._random_unit_vectors(batch_size)
            normal = self._random_unit_vectors(batch_size)
            test_cases.append({
                'rays_o': rays_o,
                'rays_d': rays_d,
                'normal': normal,
                'train_mode': False,
                'ret_spec': False,
                'clip_spec': False,
                'description': f'Batch: {batch_size} rays, no clip',
            })

        if num_tests > 2:
            # Test 3: With clip_spec enabled
            batch_size = 20
            rays_o = torch.randn(batch_size, 3)
            rays_d = self._random_unit_vectors(batch_size)
            normal = self._random_unit_vectors(batch_size)
            test_cases.append({
                'rays_o': rays_o,
                'rays_d': rays_d,
                'normal': normal,
                'train_mode': False,
                'ret_spec': False,
                'clip_spec': True,
                'description': f'Batch: {batch_size} rays, with clip_spec',
            })

        if num_tests > 3:
            # Test 4: 2D batch
            batch_shape = (5, 8)
            rays_o = torch.randn(*batch_shape, 3)
            rays_d = self._random_unit_vectors(*batch_shape)
            normal = self._random_unit_vectors(*batch_shape)
            test_cases.append({
                'rays_o': rays_o,
                'rays_d': rays_d,
                'normal': normal,
                'train_mode': False,
                'ret_spec': False,
                'clip_spec': False,
                'description': f'2D Batch: shape {batch_shape}',
            })

        if num_tests > 4:
            # Test 5: Medium batch with varied angles
            batch_size = 15
            rays_o = torch.randn(batch_size, 3)
            rays_d = self._random_unit_vectors(batch_size)
            normal = self._random_unit_vectors(batch_size)
            test_cases.append({
                'rays_o': rays_o,
                'rays_d': rays_d,
                'normal': normal,
                'train_mode': False,
                'ret_spec': False,
                'clip_spec': True,
                'description': f'Medium batch: {batch_size} rays with clip_spec',
            })

        if num_tests > 5:
            # Test 6: Large batch
            batch_size = 100
            rays_o = torch.randn(batch_size, 3)
            rays_d = self._random_unit_vectors(batch_size)
            normal = self._random_unit_vectors(batch_size)
            test_cases.append({
                'rays_o': rays_o,
                'rays_d': rays_d,
                'normal': normal,
                'train_mode': False,
                'ret_spec': False,
                'clip_spec': False,
                'description': f'Large batch: {batch_size} rays',
            })

        if num_tests > 6:
            # Test 7: 3D batch
            batch_shape = (4, 5, 6)
            rays_o = torch.randn(*batch_shape, 3)
            rays_d = self._random_unit_vectors(*batch_shape)
            normal = self._random_unit_vectors(*batch_shape)
            test_cases.append({
                'rays_o': rays_o,
                'rays_d': rays_d,
                'normal': normal,
                'train_mode': False,
                'ret_spec': False,
                'clip_spec': True,
                'description': f'3D Batch: shape {batch_shape}',
            })

        if num_tests > 7:
            # Test 8: Large batch with clip_spec
            batch_size = 50
            rays_o = torch.randn(batch_size, 3)
            rays_d = self._random_unit_vectors(batch_size)
            normal = self._random_unit_vectors(batch_size)
            test_cases.append({
                'rays_o': rays_o,
                'rays_d': rays_d,
                'normal': normal,
                'train_mode': False,
                'ret_spec': False,
                'clip_spec': True,
                'description': f'Large batch with clip: {batch_size} rays',
            })

        if num_tests > 8:
            # Test 9: Random angles (avoid extreme cases)
            batch_size = 25
            rays_o = torch.randn(batch_size, 3)
            rays_d = self._random_unit_vectors(batch_size)
            normal = self._random_unit_vectors(batch_size)
            # Ensure angles are not too extreme (dot product between -0.9 and 0.9)
            for i in range(batch_size):
                dot_prod = (normal[i] * rays_d[i]).sum()
                if abs(dot_prod) > 0.9:
                    # Re-generate normal to avoid near-parallel case
                    normal[i] = self._random_unit_vectors(1)[0]
            test_cases.append({
                'rays_o': rays_o,
                'rays_d': rays_d,
                'normal': normal,
                'train_mode': False,
                'ret_spec': False,
                'clip_spec': False,
                'description': f'Random angles: {batch_size} rays (avoiding extreme angles)',
            })

        if num_tests > 9:
            # Test 10: Small batch with high precision
            batch_size = 5
            rays_o = torch.randn(batch_size, 3)
            rays_d = self._random_unit_vectors(batch_size)
            normal = self._random_unit_vectors(batch_size)
            test_cases.append({
                'rays_o': rays_o,
                'rays_d': rays_d,
                'normal': normal,
                'train_mode': False,
                'ret_spec': False,
                'clip_spec': False,
                'description': f'Small batch: {batch_size} rays for precision test',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            batch_size = 10 + i * 5
            batch_shape = (batch_size,) if i % 2 == 0 else (2 + i % 3, batch_size // (2 + i % 3))
            rays_o = torch.randn(*batch_shape, 3)
            rays_d = self._random_unit_vectors(*batch_shape)
            normal = self._random_unit_vectors(*batch_shape)

            test_cases.append({
                'rays_o': rays_o,
                'rays_d': rays_d,
                'normal': normal,
                'train_mode': False,
                'ret_spec': False,
                'clip_spec': i % 2 == 1,
                'description': f'Additional test {i+1}: shape {batch_shape}, clip={i % 2 == 1}',
            })

        return test_cases[:num_tests]
