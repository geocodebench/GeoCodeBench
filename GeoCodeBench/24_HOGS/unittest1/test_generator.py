"""
Test data generator for HOGS Gaussian Model unit tests.
"""

from __future__ import annotations

import torch
import numpy as np


class TestDataGenerator:
    """Generate test data for HOGS Gaussian Model functions."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small number of points
        num_points = 10
        _xyz = torch.randn(num_points, 3, dtype=torch.float32)
        _w = torch.randn(num_points, dtype=torch.float32) * 0.5  # Keep reasonable range
        test_cases.append({
            '_xyz': _xyz,
            '_w': _w,
            'description': f'Basic: {num_points} points, small scale',
        })

        if num_tests > 1:
            # Test 2: Larger number of points
            num_points = 100
            _xyz = torch.randn(num_points, 3, dtype=torch.float32) * 10
            _w = torch.randn(num_points, dtype=torch.float32) * 2
            test_cases.append({
                '_xyz': _xyz,
                '_w': _w,
                'description': f'Medium: {num_points} points, larger scale',
            })

        if num_tests > 2:
            # Test 3: Points near origin
            num_points = 50
            _xyz = torch.randn(num_points, 3, dtype=torch.float32) * 0.1
            _w = torch.randn(num_points, dtype=torch.float32) * 0.1
            test_cases.append({
                '_xyz': _xyz,
                '_w': _w,
                'description': f'Near origin: {num_points} points, small coordinates',
            })

        if num_tests > 3:
            # Test 4: Points far from origin
            num_points = 50
            _xyz = torch.randn(num_points, 3, dtype=torch.float32) * 100
            _w = torch.randn(num_points, dtype=torch.float32) * 5
            test_cases.append({
                '_xyz': _xyz,
                '_w': _w,
                'description': f'Far from origin: {num_points} points, large coordinates',
            })

        if num_tests > 4:
            # Test 5: Mixed positive and negative w
            num_points = 75
            _xyz = torch.randn(num_points, 3, dtype=torch.float32) * 5
            _w = torch.randn(num_points, dtype=torch.float32) * 3
            test_cases.append({
                '_xyz': _xyz,
                '_w': _w,
                'description': f'Mixed: {num_points} points, varied range',
            })

        if num_tests > 5:
            # Test 6: Single point
            num_points = 1
            _xyz = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
            _w = torch.tensor([0.5], dtype=torch.float32)
            test_cases.append({
                '_xyz': _xyz,
                '_w': _w,
                'description': f'Edge case: single point',
            })

        if num_tests > 6:
            # Test 7: Points on axes
            num_points = 6
            _xyz = torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0]
            ], dtype=torch.float32)
            _w = torch.ones(num_points, dtype=torch.float32) * 0.5
            test_cases.append({
                '_xyz': _xyz,
                '_w': _w,
                'description': f'Special: points on coordinate axes',
            })

        if num_tests > 7:
            # Test 8: Large batch
            num_points = 500
            _xyz = torch.randn(num_points, 3, dtype=torch.float32) * 20
            _w = torch.randn(num_points, dtype=torch.float32) * 2
            test_cases.append({
                '_xyz': _xyz,
                '_w': _w,
                'description': f'Large: {num_points} points',
            })

        if num_tests > 8:
            # Test 9: Very small w values (large inv_w)
            num_points = 30
            _xyz = torch.randn(num_points, 3, dtype=torch.float32) * 5
            _w = torch.randn(num_points, dtype=torch.float32) * 0.01 - 5  # exp will be small
            test_cases.append({
                '_xyz': _xyz,
                '_w': _w,
                'description': f'Small w: {num_points} points, large inverse w',
            })

        if num_tests > 9:
            # Test 10: Points in specific regions
            num_points = 40
            _xyz = torch.randn(num_points, 3, dtype=torch.float32).abs() * 10  # All positive quadrant
            _w = torch.randn(num_points, dtype=torch.float32) * 1.5
            test_cases.append({
                '_xyz': _xyz,
                '_w': _w,
                'description': f'Positive quadrant: {num_points} points',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            num_points = 20 + i * 10
            scale = 1 + i * 2
            _xyz = torch.randn(num_points, 3, dtype=torch.float32) * scale
            _w = torch.randn(num_points, dtype=torch.float32) * (scale / 2)

            test_cases.append({
                '_xyz': _xyz,
                '_w': _w,
                'description': f'Additional test {i+1}: {num_points} points, scale={scale}',
            })

        return test_cases[:num_tests]
