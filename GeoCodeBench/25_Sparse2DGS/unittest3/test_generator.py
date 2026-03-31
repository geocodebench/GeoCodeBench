"""
Test Data Generator for build_rotation function.
Generates test cases with different configurations.
"""

import torch
import math


class TestDataGenerator:
    """Generate test data for build_rotation function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small batch
        batch_size = 5
        quaternions = torch.randn(batch_size, 4)
        test_cases.append({
            'quaternions': quaternions,
            'description': f'Basic: batch_size={batch_size}',
        })

        if num_tests > 1:
            batch_size = 1
            quaternions = torch.randn(batch_size, 4)
            test_cases.append({
                'quaternions': quaternions,
                'description': f'Single: batch_size={batch_size}',
            })

        if num_tests > 2:
            batch_size = 3
            quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * batch_size, dtype=torch.float32)
            test_cases.append({
                'quaternions': quaternions,
                'description': f'Identity: batch_size={batch_size}, identity quaternions',
            })

        if num_tests > 3:
            batch_size = 2
            angle = math.pi / 4
            quaternions = torch.tensor([
                [math.cos(angle), 0.0, 0.0, math.sin(angle)],
                [math.cos(angle), 0.0, 0.0, math.sin(angle)]
            ], dtype=torch.float32)
            test_cases.append({
                'quaternions': quaternions,
                'description': f'90° Z-rotation: batch_size={batch_size}',
            })

        if num_tests > 4:
            batch_size = 50
            quaternions = torch.randn(batch_size, 4)
            test_cases.append({
                'quaternions': quaternions,
                'description': f'Large batch: batch_size={batch_size}',
            })

        if num_tests > 5:
            batch_size = 10
            quaternions = torch.randn(batch_size, 4)
            quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)
            test_cases.append({
                'quaternions': quaternions,
                'description': f'Pre-normalized: batch_size={batch_size}, already normalized',
            })

        if num_tests > 6:
            batch_size = 8
            quaternions = torch.randn(batch_size, 4) * 0.1
            test_cases.append({
                'quaternions': quaternions,
                'description': f'Small magnitude: batch_size={batch_size}, small values',
            })

        if num_tests > 7:
            batch_size = 12
            quaternions = torch.randn(batch_size, 4)
            quaternions[::2] = torch.abs(quaternions[::2])
            test_cases.append({
                'quaternions': quaternions,
                'description': f'Mixed signs: batch_size={batch_size}',
            })

        if num_tests > 8:
            batch_size = 4
            quaternions = torch.tensor([
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.7071, 0.7071, 0.0],
            ], dtype=torch.float32)
            test_cases.append({
                'quaternions': quaternions,
                'description': f'180° rotations: batch_size={batch_size}, various axes',
            })

        if num_tests > 9:
            batch_size = 200
            quaternions = torch.randn(batch_size, 4)
            test_cases.append({
                'quaternions': quaternions,
                'description': f'Very large batch: batch_size={batch_size}',
            })

        for i in range(num_tests - len(test_cases)):
            batch_size = 10 + i * 5
            quaternions = torch.randn(batch_size, 4)

            test_cases.append({
                'quaternions': quaternions,
                'description': f'Additional test {i+1}: batch_size={batch_size}',
            })

        return test_cases[:num_tests]
