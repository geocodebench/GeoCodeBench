"""
Test Data Generator for de_casteljau_split function.
"""

from __future__ import annotations

import torch


class TestDataGenerator:
    """Generate test data for de_casteljau_split function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic Bezier curve split at t=0.5
        batch_size = 2
        curves = torch.randn(batch_size, 4, 3)
        t = torch.ones(batch_size) * 0.5
        is_bezier = torch.ones(batch_size, dtype=torch.bool)
        test_cases.append({
            'curves': curves,
            't': t,
            'is_bezier': is_bezier,
            'description': f'Basic Bezier: batch={batch_size}, t=0.5',
        })

        if num_tests > 1:
            # Test 2: Mixed Bezier and line
            batch_size = 3
            curves = torch.randn(batch_size, 4, 3)
            t = torch.tensor([0.3, 0.5, 0.7])
            is_bezier = torch.tensor([True, False, True], dtype=torch.bool)
            test_cases.append({
                'curves': curves,
                't': t,
                'is_bezier': is_bezier,
                'description': f'Mixed: batch={batch_size}, mixed types',
            })

        if num_tests > 2:
            # Test 3: All straight lines
            batch_size = 4
            curves = torch.randn(batch_size, 4, 3)
            t = torch.tensor([0.1, 0.3, 0.5, 0.9])
            is_bezier = torch.zeros(batch_size, dtype=torch.bool)
            test_cases.append({
                'curves': curves,
                't': t,
                'is_bezier': is_bezier,
                'description': f'All lines: batch={batch_size}, t varies',
            })

        if num_tests > 3:
            # Test 4: Large batch
            batch_size = 8
            curves = torch.randn(batch_size, 4, 3)
            t = torch.linspace(0.1, 0.9, batch_size)
            is_bezier = torch.ones(batch_size, dtype=torch.bool)
            test_cases.append({
                'curves': curves,
                't': t,
                'is_bezier': is_bezier,
                'description': f'Large batch: batch={batch_size}, all Bezier',
            })

        if num_tests > 4:
            # Test 5: Single curve
            batch_size = 1
            curves = torch.randn(batch_size, 4, 3)
            t = torch.tensor([0.5])
            is_bezier = torch.tensor([True], dtype=torch.bool)
            test_cases.append({
                'curves': curves,
                't': t,
                'is_bezier': is_bezier,
                'description': f'Single curve: batch=1',
            })

        if num_tests > 5:
            # Test 6: Edge case - t=0
            batch_size = 2
            curves = torch.randn(batch_size, 4, 3)
            t = torch.zeros(batch_size)
            is_bezier = torch.ones(batch_size, dtype=torch.bool)
            test_cases.append({
                'curves': curves,
                't': t,
                'is_bezier': is_bezier,
                'description': f'Edge case: t=0',
            })

        if num_tests > 6:
            # Test 7: Edge case - t=1
            batch_size = 2
            curves = torch.randn(batch_size, 4, 3)
            t = torch.ones(batch_size)
            is_bezier = torch.ones(batch_size, dtype=torch.bool)
            test_cases.append({
                'curves': curves,
                't': t,
                'is_bezier': is_bezier,
                'description': f'Edge case: t=1',
            })

        if num_tests > 7:
            # Test 8: Different t values for different curves
            batch_size = 5
            curves = torch.randn(batch_size, 4, 3)
            t = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
            is_bezier = torch.tensor([True, False, True, False, True], dtype=torch.bool)
            test_cases.append({
                'curves': curves,
                't': t,
                'is_bezier': is_bezier,
                'description': f'Varied t: batch={batch_size}, mixed types',
            })

        if num_tests > 8:
            # Test 9: Scalar t (should be broadcasted)
            batch_size = 3
            curves = torch.randn(batch_size, 4, 3)
            t = torch.tensor(0.5)  # Scalar
            is_bezier = torch.ones(batch_size, dtype=torch.bool)
            test_cases.append({
                'curves': curves,
                't': t,
                'is_bezier': is_bezier,
                'description': f'Scalar t: batch={batch_size}, t=0.5',
            })

        if num_tests > 9:
            # Test 10: More complex mixed case
            batch_size = 6
            curves = torch.randn(batch_size, 4, 3)
            t = torch.rand(batch_size)  # Random t values
            is_bezier = torch.tensor([True, False, True, True, False, False], dtype=torch.bool)
            test_cases.append({
                'curves': curves,
                't': t,
                'is_bezier': is_bezier,
                'description': f'Complex: batch={batch_size}, random t, mixed types',
            })

        return test_cases[:num_tests]
