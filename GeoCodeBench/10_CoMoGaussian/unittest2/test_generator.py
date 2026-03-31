"""
Test Data Generator for skew_symmetric, transform_SE3, and rodrigues_formula functions.
"""

import math

import torch


class TestDataGenerator:
    """Generate test data for the three functions."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_skew_symmetric_tests(self, num_tests=5):
        """Generate test cases for skew_symmetric."""
        test_cases = []

        # Test 1: Basic single vector
        w = torch.randn(3)
        test_cases.append({
            'w': w,
            'description': 'Basic: single 3D vector',
        })

        if num_tests > 1:
            # Test 2: Batch of vectors
            w = torch.randn(5, 3)
            test_cases.append({
                'w': w,
                'description': f'Batch: 5 vectors (5, 3)',
            })

        if num_tests > 2:
            # Test 3: Larger batch
            w = torch.randn(10, 3)
            test_cases.append({
                'w': w,
                'description': f'Large batch: 10 vectors (10, 3)',
            })

        if num_tests > 3:
            # Test 4: Edge case - zeros
            w = torch.zeros(3)
            test_cases.append({
                'w': w,
                'description': 'Edge case: zero vector',
            })

        if num_tests > 4:
            # Test 5: Edge case - small values
            w = torch.tensor([1e-6, 1e-6, 1e-6])
            test_cases.append({
                'w': w,
                'description': 'Edge case: very small values',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            batch_size = 3 + i * 2
            w = torch.randn(batch_size, 3)
            test_cases.append({
                'w': w,
                'description': f'Additional test {i+1}: batch_size={batch_size}',
            })

        return test_cases[:num_tests]

    def generate_transform_SE3_tests(self, num_tests=5):
        """Generate test cases for transform_SE3."""
        test_cases = []

        # Test 1: Basic single transformation
        exp_w_skew = torch.eye(3)
        p = torch.randn(3, 1)
        test_cases.append({
            'exp_w_skew': exp_w_skew,
            'p': p,
            'description': 'Basic: identity rotation',
        })

        if num_tests > 1:
            # Test 2: Batch transformations
            batch_size = 5
            exp_w_skew = torch.randn(batch_size, 3, 3)
            p = torch.randn(batch_size, 3, 1)
            test_cases.append({
                'exp_w_skew': exp_w_skew,
                'p': p,
                'description': f'Batch: {batch_size} transformations',
            })

        if num_tests > 2:
            # Test 3: Larger batch
            batch_size = 10
            exp_w_skew = torch.randn(batch_size, 3, 3)
            p = torch.randn(batch_size, 3, 1)
            test_cases.append({
                'exp_w_skew': exp_w_skew,
                'p': p,
                'description': f'Large batch: {batch_size} transformations',
            })

        if num_tests > 3:
            # Test 4: Edge case - zero translation
            exp_w_skew = torch.randn(1, 3, 3)
            p = torch.zeros(1, 3, 1)
            test_cases.append({
                'exp_w_skew': exp_w_skew,
                'p': p,
                'description': 'Edge case: zero translation',
            })

        if num_tests > 4:
            # Test 5: Orthogonal rotation matrix
            u, _, v = torch.svd(torch.randn(3, 3))
            exp_w_skew = torch.matmul(u, v.T)
            p = torch.randn(3, 1)
            test_cases.append({
                'exp_w_skew': exp_w_skew,
                'p': p,
                'description': 'Orthogonal rotation matrix',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            batch_size = 3 + i * 2
            exp_w_skew = torch.randn(batch_size, 3, 3)
            p = torch.randn(batch_size, 3, 1)
            test_cases.append({
                'exp_w_skew': exp_w_skew,
                'p': p,
                'description': f'Additional test {i+1}: batch_size={batch_size}',
            })

        return test_cases[:num_tests]

    def generate_rodrigues_formula_tests(self, num_tests=5):
        """Generate test cases for rodrigues_formula."""
        test_cases = []

        # Test 1: Basic case - create a skew-symmetric matrix
        w_vec = torch.tensor([1.0, 0.0, 0.0])  # axis
        theta = torch.tensor([[math.pi / 2]])  # 90 degrees
        w = torch.zeros(1, 3, 3)
        w[0, 0, 1] = -w_vec[2]
        w[0, 0, 2] = w_vec[1]
        w[0, 1, 0] = w_vec[2]
        w[0, 1, 2] = -w_vec[0]
        w[0, 2, 0] = -w_vec[1]
        w[0, 2, 1] = w_vec[0]
        test_cases.append({
            'w': w,
            'theta': theta,
            'description': 'Basic: 90-degree rotation around x-axis',
        })

        if num_tests > 1:
            # Test 2: Batch rotations
            batch_size = 5
            w = torch.randn(batch_size, 3, 3)
            # Make skew-symmetric
            w = (w - w.transpose(-2, -1)) / 2
            theta = torch.rand(batch_size, 1) * math.pi
            test_cases.append({
                'w': w,
                'theta': theta,
                'description': f'Batch: {batch_size} rotations',
            })

        if num_tests > 2:
            # Test 3: Larger batch
            batch_size = 10
            w = torch.randn(batch_size, 3, 3)
            w = (w - w.transpose(-2, -1)) / 2
            theta = torch.rand(batch_size, 1) * math.pi
            test_cases.append({
                'w': w,
                'theta': theta,
                'description': f'Large batch: {batch_size} rotations',
            })

        if num_tests > 3:
            # Test 4: Edge case - zero rotation
            w = torch.zeros(1, 3, 3)
            theta = torch.zeros(1, 1)
            test_cases.append({
                'w': w,
                'theta': theta,
                'description': 'Edge case: zero rotation (theta=0)',
            })

        if num_tests > 4:
            # Test 5: Small rotation
            w = torch.randn(1, 3, 3)
            w = (w - w.transpose(-2, -1)) / 2
            theta = torch.tensor([[1e-3]])  # Very small rotation
            test_cases.append({
                'w': w,
                'theta': theta,
                'description': 'Edge case: very small rotation',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            batch_size = 3 + i * 2
            w = torch.randn(batch_size, 3, 3)
            w = (w - w.transpose(-2, -1)) / 2
            theta = torch.rand(batch_size, 1) * math.pi
            test_cases.append({
                'w': w,
                'theta': theta,
                'description': f'Additional test {i+1}: batch_size={batch_size}',
            })

        return test_cases[:num_tests]
