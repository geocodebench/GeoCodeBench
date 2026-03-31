"""
Test Data Generator for wahba() function.
Generates various test cases with different configurations.
"""

import torch


class TestDataGenerator:
    """Generate test data for wahba() function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small batch and small number of points
        B = 2
        N = 10
        source_points = torch.randn(B, 3, N)
        target_points = torch.randn(B, 3, N)
        test_cases.append({
            'source_points': source_points,
            'target_points': target_points,
            'device_': None,
            'description': f'Basic: B={B}, N={N}',
        })

        if num_tests > 1:
            # Test 2: Single batch, more points
            B = 1
            N = 20
            source_points = torch.randn(B, 3, N)
            target_points = torch.randn(B, 3, N)
            test_cases.append({
                'source_points': source_points,
                'target_points': target_points,
                'device_': None,
                'description': f'Single batch: B={B}, N={N}',
            })

        if num_tests > 2:
            # Test 3: Larger batch, medium points
            B = 4
            N = 15
            source_points = torch.randn(B, 3, N)
            target_points = torch.randn(B, 3, N)
            test_cases.append({
                'source_points': source_points,
                'target_points': target_points,
                'device_': None,
                'description': f'Larger batch: B={B}, N={N}',
            })

        if num_tests > 3:
            # Test 4: Small batch, many points
            B = 2
            N = 50
            source_points = torch.randn(B, 3, N)
            target_points = torch.randn(B, 3, N)
            test_cases.append({
                'source_points': source_points,
                'target_points': target_points,
                'device_': None,
                'description': f'Many points: B={B}, N={N}',
            })

        if num_tests > 4:
            # Test 5: Large batch
            B = 8
            N = 12
            source_points = torch.randn(B, 3, N)
            target_points = torch.randn(B, 3, N)
            test_cases.append({
                'source_points': source_points,
                'target_points': target_points,
                'device_': None,
                'description': f'Large batch: B={B}, N={N}',
            })

        if num_tests > 5:
            # Test 6: Edge case - minimal points (N=3)
            B = 3
            N = 3
            source_points = torch.randn(B, 3, N)
            target_points = torch.randn(B, 3, N)
            test_cases.append({
                'source_points': source_points,
                'target_points': target_points,
                'device_': None,
                'description': f'Minimal points: B={B}, N={N}',
            })

        if num_tests > 6:
            # Test 7: Edge case - very many points
            B = 2
            N = 100
            source_points = torch.randn(B, 3, N)
            target_points = torch.randn(B, 3, N)
            test_cases.append({
                'source_points': source_points,
                'target_points': target_points,
                'device_': None,
                'description': f'Very many points: B={B}, N={N}',
            })

        if num_tests > 7:
            # Test 8: Float32 dtype
            B = 3
            N = 8
            source_points = torch.randn(B, 3, N, dtype=torch.float32)
            target_points = torch.randn(B, 3, N, dtype=torch.float32)
            test_cases.append({
                'source_points': source_points,
                'target_points': target_points,
                'device_': None,
                'description': f'Float32: B={B}, N={N}',
            })

        if num_tests > 8:
            # Test 9: Float64 dtype
            B = 2
            N = 10
            source_points = torch.randn(B, 3, N, dtype=torch.float64)
            target_points = torch.randn(B, 3, N, dtype=torch.float64)
            test_cases.append({
                'source_points': source_points,
                'target_points': target_points,
                'device_': None,
                'description': f'Float64: B={B}, N={N}',
            })

        if num_tests > 9:
            # Test 10: Normalized input (unit vectors)
            B = 4
            N = 15
            source_points = torch.randn(B, 3, N)
            source_points = source_points / (torch.norm(source_points, dim=1, keepdim=True) + 1e-8)
            target_points = torch.randn(B, 3, N)
            target_points = target_points / (torch.norm(target_points, dim=1, keepdim=True) + 1e-8)
            test_cases.append({
                'source_points': source_points,
                'target_points': target_points,
                'device_': None,
                'description': f'Normalized: B={B}, N={N}',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            B = 2 + (i % 5)
            N = 5 + (i % 20) * 2
            source_points = torch.randn(B, 3, N)
            target_points = torch.randn(B, 3, N)

            test_cases.append({
                'source_points': source_points,
                'target_points': target_points,
                'device_': None,
                'description': f'Additional test {i+1}: B={B}, N={N}',
            })

        return test_cases[:num_tests]
