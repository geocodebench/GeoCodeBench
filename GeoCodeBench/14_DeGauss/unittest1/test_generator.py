"""
Test data generator for BrightnessActivation function.
"""

from __future__ import annotations

import torch


class TestDataGenerator:
    """Generate test data for BrightnessActivation function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=10):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with single tensor
        x = torch.rand(10, 1, 32, 32)
        test_cases.append({
            'x': x,
            'description': 'Basic: shape (10, 1, 32, 32), random values',
        })

        if num_tests > 1:
            # Test 2: Below threshold only
            x = torch.rand(10, 1, 32, 32) * 0.5  # All values < 0.75
            test_cases.append({
                'x': x,
                'description': 'Below threshold: all values in [0, 0.5]',
            })

        if num_tests > 2:
            # Test 3: Above threshold only
            x = torch.rand(10, 1, 32, 32) * 0.2 + 0.8  # All values > 0.75
            test_cases.append({
                'x': x,
                'description': 'Above threshold: all values in [0.8, 1.0]',
            })

        if num_tests > 3:
            # Test 4: Mixed values
            x = torch.rand(10, 1, 32, 32)  # Mixed values
            test_cases.append({
                'x': x,
                'description': 'Mixed: random values in [0, 1]',
            })

        if num_tests > 4:
            # Test 5: Edge case - exactly at threshold
            x = torch.full((10, 1, 32, 32), 0.75)
            test_cases.append({
                'x': x,
                'description': 'Edge case: exactly at threshold (0.75)',
            })

        if num_tests > 5:
            # Test 6: Small tensor
            x = torch.rand(1, 1, 4, 4)
            test_cases.append({
                'x': x,
                'description': 'Small tensor: shape (1, 1, 4, 4)',
            })

        if num_tests > 6:
            # Test 7: Large tensor
            x = torch.rand(4, 3, 512, 512)
            test_cases.append({
                'x': x,
                'description': 'Large tensor: shape (4, 3, 512, 512)',
            })

        if num_tests > 7:
            # Test 8: Boundary values (0 and 1)
            x = torch.zeros(10, 1, 32, 32)
            x[::2] = 1.0
            test_cases.append({
                'x': x,
                'description': 'Boundary values: alternating 0 and 1',
            })

        if num_tests > 8:
            # Test 9: Slightly above and below threshold
            x = torch.cat([
                torch.full((5, 1, 32, 32), 0.74),
                torch.full((5, 1, 32, 32), 0.76)
            ], dim=0)
            test_cases.append({
                'x': x,
                'description': 'Near threshold: 0.74 and 0.76',
            })

        if num_tests > 9:
            # Test 10: Single element tensor
            x = torch.tensor([[[[0.8]]]])
            test_cases.append({
                'x': x,
                'description': 'Single element: shape (1, 1, 1, 1)',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            # Random shapes and values
            batch_size = 2 + (i % 3)
            channels = 1 + (i % 3)
            height = 16 + (i % 32) * 16
            width = 16 + (i % 32) * 16
            x = torch.rand(batch_size, channels, height, width)

            test_cases.append({
                'x': x,
                'description': f'Additional test {i+1}: shape ({batch_size}, {channels}, {height}, {width})',
            })

        return test_cases[:num_tests]
