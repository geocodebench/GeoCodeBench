"""
Test Data Generator for Gumbel_Network.
Generates test cases with different batch sizes and tau values.
"""

import torch


class TestDataGenerator:
    """Generate test data for Gumbel_Network."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small batch
        batch_size = 8
        test_cases.append({
            'batch_size': batch_size,
            'tau': 1.0,
            'description': f'Basic: batch_size={batch_size}, tau=1.0',
        })

        if num_tests > 1:
            # Test 2: Different tau value
            batch_size = 16
            test_cases.append({
                'batch_size': batch_size,
                'tau': 0.5,
                'description': f'Lower tau: batch_size={batch_size}, tau=0.5',
            })

        if num_tests > 2:
            # Test 3: Higher tau
            batch_size = 12
            test_cases.append({
                'batch_size': batch_size,
                'tau': 2.0,
                'description': f'Higher tau: batch_size={batch_size}, tau=2.0',
            })

        if num_tests > 3:
            # Test 4: Larger batch
            batch_size = 32
            test_cases.append({
                'batch_size': batch_size,
                'tau': 1.0,
                'description': f'Large batch: batch_size={batch_size}, tau=1.0',
            })

        if num_tests > 4:
            # Test 5: Very small tau (sharper distribution)
            batch_size = 10
            test_cases.append({
                'batch_size': batch_size,
                'tau': 0.1,
                'description': f'Sharp distribution: batch_size={batch_size}, tau=0.1',
            })

        if num_tests > 5:
            # Test 6: Very large tau (smoother distribution)
            batch_size = 20
            test_cases.append({
                'batch_size': batch_size,
                'tau': 5.0,
                'description': f'Smooth distribution: batch_size={batch_size}, tau=5.0',
            })

        if num_tests > 6:
            # Test 7: Single element batch
            batch_size = 1
            test_cases.append({
                'batch_size': batch_size,
                'tau': 1.0,
                'description': f'Single element: batch_size={batch_size}, tau=1.0',
            })

        if num_tests > 7:
            # Test 8: Large batch with different tau
            batch_size = 64
            test_cases.append({
                'batch_size': batch_size,
                'tau': 0.7,
                'description': f'Large batch variant: batch_size={batch_size}, tau=0.7',
            })

        if num_tests > 8:
            # Test 9: Medium batch with medium tau
            batch_size = 24
            test_cases.append({
                'batch_size': batch_size,
                'tau': 1.5,
                'description': f'Medium config: batch_size={batch_size}, tau=1.5',
            })

        if num_tests > 9:
            # Test 10: Edge case with very small batch and extreme tau
            batch_size = 4
            test_cases.append({
                'batch_size': batch_size,
                'tau': 0.01,
                'description': f'Edge case: batch_size={batch_size}, tau=0.01',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            batch_size = 8 + i * 4
            tau = 0.5 + i * 0.3

            test_cases.append({
                'batch_size': batch_size,
                'tau': tau,
                'description': f'Additional test {i+1}: batch_size={batch_size}, tau={tau:.2f}',
            })

        return test_cases[:num_tests]
