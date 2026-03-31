"""
Test Data Generator for Channel_CTX_fea.forward() function.
Generates test cases with different configurations.
"""

import torch


class TestDataGenerator:
    """Generate test data for Channel_CTX_fea.forward() function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small batch, to_dec=-1
        N = 4
        fea_q = torch.randn(N, 256)
        to_dec = -1
        test_cases.append({
            'fea_q': fea_q,
            'to_dec': to_dec,
            'description': f'Basic: N={N}, to_dec=-1 (return all levels)',
        })

        if num_tests > 1:
            # Test 2: Single sample, to_dec=0
            N = 1
            fea_q = torch.randn(N, 256)
            to_dec = 0
            test_cases.append({
                'fea_q': fea_q,
                'to_dec': to_dec,
                'description': f'Single sample: N={N}, to_dec=0 (return level 0)',
            })

        if num_tests > 2:
            # Test 3: Larger batch, to_dec=1
            N = 8
            fea_q = torch.randn(N, 256)
            to_dec = 1
            test_cases.append({
                'fea_q': fea_q,
                'to_dec': to_dec,
                'description': f'Larger batch: N={N}, to_dec=1 (return level 1)',
            })

        if num_tests > 3:
            # Test 4: Medium batch, to_dec=2
            N = 5
            fea_q = torch.randn(N, 256)
            to_dec = 2
            test_cases.append({
                'fea_q': fea_q,
                'to_dec': to_dec,
                'description': f'Medium batch: N={N}, to_dec=2 (return level 2)',
            })

        if num_tests > 4:
            # Test 5: to_dec=3
            N = 6
            fea_q = torch.randn(N, 256)
            to_dec = 3
            test_cases.append({
                'fea_q': fea_q,
                'to_dec': to_dec,
                'description': f'Test to_dec=3: N={N}, to_dec=3 (return level 3)',
            })

        if num_tests > 5:
            # Test 6: Large batch, to_dec=-1
            N = 16
            fea_q = torch.randn(N, 256)
            to_dec = -1
            test_cases.append({
                'fea_q': fea_q,
                'to_dec': to_dec,
                'description': f'Large batch: N={N}, to_dec=-1',
            })

        if num_tests > 6:
            # Test 7: Edge case - all zeros input
            N = 3
            fea_q = torch.zeros(N, 256)
            to_dec = -1
            test_cases.append({
                'fea_q': fea_q,
                'to_dec': to_dec,
                'description': f'Edge case: N={N}, zero input, to_dec=-1',
            })

        if num_tests > 7:
            # Test 8: Different to_dec values on same input
            N = 4
            fea_q = torch.randn(N, 256)
            to_dec = 0
            test_cases.append({
                'fea_q': fea_q,
                'to_dec': to_dec,
                'description': f'Same input: N={N}, to_dec=0',
            })

        if num_tests > 8:
            # Test 9: Very large batch
            N = 32
            fea_q = torch.randn(N, 256)
            to_dec = -1
            test_cases.append({
                'fea_q': fea_q,
                'to_dec': to_dec,
                'description': f'Very large batch: N={N}, to_dec=-1',
            })

        if num_tests > 9:
            # Test 10: Normalized input
            N = 7
            fea_q = torch.randn(N, 256)
            fea_q = (fea_q - fea_q.mean()) / (fea_q.std() + 1e-8)
            to_dec = 2
            test_cases.append({
                'fea_q': fea_q,
                'to_dec': to_dec,
                'description': f'Normalized input: N={N}, to_dec=2',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            N = 3 + (i % 10)
            fea_q = torch.randn(N, 256)
            to_dec = (i % 5) - 1  # -1, 0, 1, 2, 3

            test_cases.append({
                'fea_q': fea_q,
                'to_dec': to_dec,
                'description': f'Additional test {i+1}: N={N}, to_dec={to_dec}',
            })

        return test_cases[:num_tests]
