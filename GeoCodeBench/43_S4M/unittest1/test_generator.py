"""
Test Data Generator for compute_axial_cis() function.
Generates test cases with different configurations.
"""

from __future__ import annotations

import torch


class TestDataGenerator:
    """Generate test data for compute_axial_cis() function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small dimensions
        test_cases.append({
            'dim': 64,
            'end_x': 8,
            'end_y': 8,
            'theta': 10000.0,
            'description': 'Basic: dim=64, end_x=8, end_y=8, theta=10000.0',
        })

        if num_tests > 1:
            # Test 2: Different dimensions
            test_cases.append({
                'dim': 128,
                'end_x': 16,
                'end_y': 16,
                'theta': 10000.0,
                'description': 'Larger: dim=128, end_x=16, end_y=16, theta=10000.0',
            })

        if num_tests > 2:
            # Test 3: Different theta
            test_cases.append({
                'dim': 64,
                'end_x': 10,
                'end_y': 10,
                'theta': 5000.0,
                'description': 'Different theta: dim=64, end_x=10, end_y=10, theta=5000.0',
            })

        if num_tests > 3:
            # Test 4: Rectangular grid
            test_cases.append({
                'dim': 64,
                'end_x': 12,
                'end_y': 8,
                'theta': 10000.0,
                'description': 'Rectangular: dim=64, end_x=12, end_y=8, theta=10000.0',
            })

        if num_tests > 4:
            # Test 5: Small grid
            test_cases.append({
                'dim': 32,
                'end_x': 4,
                'end_y': 4,
                'theta': 10000.0,
                'description': 'Small: dim=32, end_x=4, end_y=4, theta=10000.0',
            })

        if num_tests > 5:
            # Test 6: Large theta
            test_cases.append({
                'dim': 64,
                'end_x': 8,
                'end_y': 8,
                'theta': 20000.0,
                'description': 'Large theta: dim=64, end_x=8, end_y=8, theta=20000.0',
            })

        if num_tests > 6:
            # Test 7: Odd dimensions
            test_cases.append({
                'dim': 96,
                'end_x': 7,
                'end_y': 7,
                'theta': 10000.0,
                'description': 'Odd dims: dim=96, end_x=7, end_y=7, theta=10000.0',
            })

        if num_tests > 7:
            # Test 8: Very small theta
            test_cases.append({
                'dim': 64,
                'end_x': 8,
                'end_y': 8,
                'theta': 1000.0,
                'description': 'Small theta: dim=64, end_x=8, end_y=8, theta=1000.0',
            })

        if num_tests > 8:
            # Test 9: Large dimensions
            test_cases.append({
                'dim': 256,
                'end_x': 20,
                'end_y': 20,
                'theta': 10000.0,
                'description': 'Large: dim=256, end_x=20, end_y=20, theta=10000.0',
            })

        if num_tests > 9:
            # Test 10: Different aspect ratio
            test_cases.append({
                'dim': 64,
                'end_x': 16,
                'end_y': 4,
                'theta': 10000.0,
                'description': 'Wide: dim=64, end_x=16, end_y=4, theta=10000.0',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            dim = 32 * (1 + (i % 8))  # 32, 64, 96, 128, 160, 192, 224, 256
            end_x = 4 + (i % 12)
            end_y = 4 + ((i + 1) % 12)
            theta = 1000.0 * (1 + (i % 20))  # 1000 to 20000

            test_cases.append({
                'dim': dim,
                'end_x': end_x,
                'end_y': end_y,
                'theta': theta,
                'description': f'Additional test {i+1}: dim={dim}, end_x={end_x}, end_y={end_y}, theta={theta}',
            })

        return test_cases[:num_tests]
