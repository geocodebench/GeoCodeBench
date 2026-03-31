"""
Test Data Generator for arun_batched() function.
Generates various test cases with different configurations.
"""

import torch


class TestDataGenerator:
    """Generate test data for arun_batched() function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small batch and small point set
        B = 2
        N = 10
        source_points = torch.randn(B, 3, N)
        target_points = torch.randn(B, 3, N)
        test_cases.append({
            'source_points': source_points,
            'target_points': target_points,
            'description': f'Basic: B={B}, N={N}',
        })

        if num_tests > 1:
            # Test 2: Single batch, larger point set
            B = 1
            N = 50
            source_points = torch.randn(B, 3, N)
            target_points = torch.randn(B, 3, N)
            test_cases.append({
                'source_points': source_points,
                'target_points': target_points,
                'description': f'Single batch: B={B}, N={N}',
            })

        if num_tests > 2:
            # Test 3: Larger batch, medium point set
            B = 5
            N = 20
            source_points = torch.randn(B, 3, N)
            target_points = torch.randn(B, 3, N)
            test_cases.append({
                'source_points': source_points,
                'target_points': target_points,
                'description': f'Larger batch: B={B}, N={N}',
            })

        if num_tests > 3:
            # Test 4: Small point set
            B = 3
            N = 5
            source_points = torch.randn(B, 3, N)
            target_points = torch.randn(B, 3, N)
            test_cases.append({
                'source_points': source_points,
                'target_points': target_points,
                'description': f'Small point set: B={B}, N={N}',
            })

        if num_tests > 4:
            # Test 5: Large point set
            B = 2
            N = 100
            source_points = torch.randn(B, 3, N)
            target_points = torch.randn(B, 3, N)
            test_cases.append({
                'source_points': source_points,
                'target_points': target_points,
                'description': f'Large point set: B={B}, N={N}',
            })

        if num_tests > 5:
            # Test 6: Very large batch
            B = 10
            N = 15
            source_points = torch.randn(B, 3, N)
            target_points = torch.randn(B, 3, N)
            test_cases.append({
                'source_points': source_points,
                'target_points': target_points,
                'description': f'Very large batch: B={B}, N={N}',
            })

        if num_tests > 6:
            # Test 7: Edge case - all zeros input
            B = 2
            N = 10
            source_points = torch.zeros(B, 3, N)
            target_points = torch.zeros(B, 3, N)
            test_cases.append({
                'source_points': source_points,
                'target_points': target_points,
                'description': f'Edge case: B={B}, N={N}, zero input',
            })

        if num_tests > 7:
            # Test 8: Small values
            B = 2
            N = 10
            source_points = torch.randn(B, 3, N) * 0.01
            target_points = torch.randn(B, 3, N) * 0.01
            test_cases.append({
                'source_points': source_points,
                'target_points': target_points,
                'description': f'Small values: B={B}, N={N}',
            })

        if num_tests > 8:
            # Test 9: Large values
            B = 2
            N = 10
            source_points = torch.randn(B, 3, N) * 100
            target_points = torch.randn(B, 3, N) * 100
            test_cases.append({
                'source_points': source_points,
                'target_points': target_points,
                'description': f'Large values: B={B}, N={N}',
            })

        if num_tests > 9:
            # Test 10: Normalized input
            B = 3
            N = 20
            source_points = torch.randn(B, 3, N)
            source_points = source_points / (torch.norm(source_points, dim=1, keepdim=True) + 1e-8)
            target_points = torch.randn(B, 3, N)
            target_points = target_points / (torch.norm(target_points, dim=1, keepdim=True) + 1e-8)
            test_cases.append({
                'source_points': source_points,
                'target_points': target_points,
                'description': f'Normalized input: B={B}, N={N}',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            B = 2 + (i % 8)
            N = 10 + (i % 50) * 2
            source_points = torch.randn(B, 3, N)
            target_points = torch.randn(B, 3, N)

            test_cases.append({
                'source_points': source_points,
                'target_points': target_points,
                'description': f'Additional test {i+1}: B={B}, N={N}',
            })

        return test_cases[:num_tests]
