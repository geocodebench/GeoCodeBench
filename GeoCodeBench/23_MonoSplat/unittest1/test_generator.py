"""
Test Data Generator for depth-disparity conversion functions.
Generates various test cases with different configurations.
"""

import torch


class TestDataGenerator:
    """Generate test data for depth-disparity conversion functions."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with scalar values
        near = torch.tensor(0.5)
        far = torch.tensor(10.0)
        relative_disparity = torch.tensor(0.5)
        depth = torch.tensor(2.0)
        test_cases.append({
            'near': near,
            'far': far,
            'relative_disparity': relative_disparity,
            'depth': depth,
            'eps': 1e-10,
            'description': 'Basic: scalar values',
        })

        if num_tests > 1:
            # Test 2: 1D tensor
            near = torch.ones(10) * 0.5
            far = torch.ones(10) * 10.0
            relative_disparity = torch.linspace(0, 1, 10)
            depth = torch.linspace(0.5, 10.0, 10)
            test_cases.append({
                'near': near,
                'far': far,
                'relative_disparity': relative_disparity,
                'depth': depth,
                'eps': 1e-10,
                'description': '1D tensor: 10 elements',
            })

        if num_tests > 2:
            # Test 3: 2D tensor
            batch_size = (5, 8)
            near = torch.rand(*batch_size) * 0.5 + 0.1
            far = torch.rand(*batch_size) * 5.0 + 5.0
            relative_disparity = torch.rand(*batch_size)
            depth = torch.rand(*batch_size) * 9.0 + 0.5
            test_cases.append({
                'near': near,
                'far': far,
                'relative_disparity': relative_disparity,
                'depth': depth,
                'eps': 1e-10,
                'description': f'2D tensor: {batch_size}',
            })

        if num_tests > 3:
            # Test 4: 3D tensor
            batch_size = (4, 3, 6)
            near = torch.rand(*batch_size) * 0.5 + 0.1
            far = torch.rand(*batch_size) * 5.0 + 5.0
            relative_disparity = torch.rand(*batch_size)
            depth = torch.rand(*batch_size) * 9.0 + 0.5
            test_cases.append({
                'near': near,
                'far': far,
                'relative_disparity': relative_disparity,
                'depth': depth,
                'eps': 1e-10,
                'description': f'3D tensor: {batch_size}',
            })

        if num_tests > 4:
            # Test 5: Edge case - near=0 (relative disparity)
            batch_size = (8,)
            near = torch.full(batch_size, 0.01)
            far = torch.full(batch_size, 100.0)
            relative_disparity = torch.tensor([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0])
            depth = torch.tensor([0.01, 0.5, 1.0, 5.0, 10.0, 50.0, 90.0, 100.0])
            test_cases.append({
                'near': near,
                'far': far,
                'relative_disparity': relative_disparity,
                'depth': depth,
                'eps': 1e-10,
                'description': 'Edge case: boundary values',
            })

        if num_tests > 5:
            # Test 6: Large batch
            batch_size = (32, 32)
            near = torch.rand(*batch_size) * 0.5 + 0.1
            far = torch.rand(*batch_size) * 10.0 + 10.0
            relative_disparity = torch.rand(*batch_size)
            depth = torch.rand(*batch_size) * 19.0 + 0.5
            test_cases.append({
                'near': near,
                'far': far,
                'relative_disparity': relative_disparity,
                'depth': depth,
                'eps': 1e-10,
                'description': f'Large batch: {batch_size}',
            })

        if num_tests > 6:
            # Test 7: Different eps value
            batch_size = (10,)
            near = torch.rand(batch_size) * 0.5 + 0.1
            far = torch.rand(batch_size) * 5.0 + 5.0
            relative_disparity = torch.rand(batch_size)
            depth = torch.rand(batch_size) * 9.0 + 0.5
            test_cases.append({
                'near': near,
                'far': far,
                'relative_disparity': relative_disparity,
                'depth': depth,
                'eps': 1e-8,
                'description': 'Different eps: 1e-8',
            })

        if num_tests > 7:
            # Test 8: Very close near and far
            batch_size = (6,)
            near = torch.full(batch_size, 5.0)
            far = torch.full(batch_size, 5.1)
            relative_disparity = torch.linspace(0, 1, 6)
            depth = torch.linspace(5.0, 5.1, 6)
            test_cases.append({
                'near': near,
                'far': far,
                'relative_disparity': relative_disparity,
                'depth': depth,
                'eps': 1e-10,
                'description': 'Numerical stability: close near and far',
            })

        if num_tests > 8:
            # Test 9: 4D tensor
            batch_size = (2, 3, 4, 5)
            near = torch.rand(*batch_size) * 0.5 + 0.1
            far = torch.rand(*batch_size) * 5.0 + 5.0
            relative_disparity = torch.rand(*batch_size)
            depth = torch.rand(*batch_size) * 9.0 + 0.5
            test_cases.append({
                'near': near,
                'far': far,
                'relative_disparity': relative_disparity,
                'depth': depth,
                'eps': 1e-10,
                'description': f'4D tensor: {batch_size}',
            })

        if num_tests > 9:
            # Test 10: Broadcasting - scalar near/far with tensor disparity
            near = torch.tensor(1.0)
            far = torch.tensor(10.0)
            relative_disparity = torch.rand(12)
            depth = torch.rand(12) * 9.0 + 1.0
            test_cases.append({
                'near': near,
                'far': far,
                'relative_disparity': relative_disparity,
                'depth': depth,
                'eps': 1e-10,
                'description': 'Broadcasting: scalar near/far with tensor',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            batch_size = tuple([3 + (i % 4)] * (1 + i % 3))
            near = torch.rand(*batch_size) * 0.5 + 0.1
            far = torch.rand(*batch_size) * 5.0 + 5.0
            relative_disparity = torch.rand(*batch_size)
            depth = torch.rand(*batch_size) * 9.0 + 0.5

            test_cases.append({
                'near': near,
                'far': far,
                'relative_disparity': relative_disparity,
                'depth': depth,
                'eps': 1e-10,
                'description': f'Additional test {i+1}: batch_size={batch_size}',
            })

        return test_cases[:num_tests]
