"""
Test Data Generator for update_quaternion() function.
Generates various test cases with different configurations.
"""

from __future__ import annotations

import torch


class TestDataGenerator:
    """Generate test data for update_quaternion() function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small batch
        N = 4
        q = torch.randn(N, 4)
        # Normalize quaternion
        q = q / torch.norm(q, dim=1, keepdim=True)
        omega = torch.randn(N, 3) * 0.1  # Small angular velocity
        delta_t = 0.01
        test_cases.append({
            'q': q,
            'omega': omega,
            'delta_t': delta_t,
            'description': f'Basic: N={N}, small omega, delta_t={delta_t}',
        })

        if num_tests > 1:
            # Test 2: Single sample
            N = 1
            q = torch.randn(N, 4)
            q = q / torch.norm(q, dim=1, keepdim=True)
            omega = torch.randn(N, 3) * 0.5
            delta_t = 0.1
            test_cases.append({
                'q': q,
                'omega': omega,
                'delta_t': delta_t,
                'description': f'Single sample: N={N}, delta_t={delta_t}',
            })

        if num_tests > 2:
            # Test 3: Larger batch, larger omega
            N = 8
            q = torch.randn(N, 4)
            q = q / torch.norm(q, dim=1, keepdim=True)
            omega = torch.randn(N, 3) * 1.0
            delta_t = 0.05
            test_cases.append({
                'q': q,
                'omega': omega,
                'delta_t': delta_t,
                'description': f'Larger batch: N={N}, larger omega, delta_t={delta_t}',
            })

        if num_tests > 3:
            # Test 4: Medium batch, very small omega
            N = 5
            q = torch.randn(N, 4)
            q = q / torch.norm(q, dim=1, keepdim=True)
            omega = torch.randn(N, 3) * 0.01
            delta_t = 0.001
            test_cases.append({
                'q': q,
                'omega': omega,
                'delta_t': delta_t,
                'description': f'Medium batch: N={N}, very small omega, delta_t={delta_t}',
            })

        if num_tests > 4:
            # Test 5: Zero omega (no rotation)
            N = 6
            q = torch.randn(N, 4)
            q = q / torch.norm(q, dim=1, keepdim=True)
            omega = torch.zeros(N, 3)
            delta_t = 0.1
            test_cases.append({
                'q': q,
                'omega': omega,
                'description': f'Zero omega: N={N}, no rotation',
                'delta_t': delta_t,
            })

        if num_tests > 5:
            # Test 6: Large batch, different delta_t
            N = 16
            q = torch.randn(N, 4)
            q = q / torch.norm(q, dim=1, keepdim=True)
            omega = torch.randn(N, 3) * 0.2
            delta_t = 0.2
            test_cases.append({
                'q': q,
                'omega': omega,
                'delta_t': delta_t,
                'description': f'Large batch: N={N}, delta_t={delta_t}',
            })

        if num_tests > 6:
            # Test 7: Edge case - identity quaternion
            N = 3
            q = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * N)
            omega = torch.randn(N, 3) * 0.1
            delta_t = 0.01
            test_cases.append({
                'q': q,
                'omega': omega,
                'delta_t': delta_t,
                'description': f'Identity quaternion: N={N}',
            })

        if num_tests > 7:
            # Test 8: Large angular velocity
            N = 4
            q = torch.randn(N, 4)
            q = q / torch.norm(q, dim=1, keepdim=True)
            omega = torch.randn(N, 3) * 5.0
            delta_t = 0.01
            test_cases.append({
                'q': q,
                'omega': omega,
                'delta_t': delta_t,
                'description': f'Large omega: N={N}, omega magnitude ~5.0',
            })

        if num_tests > 8:
            # Test 9: Very large batch
            N = 32
            q = torch.randn(N, 4)
            q = q / torch.norm(q, dim=1, keepdim=True)
            omega = torch.randn(N, 3) * 0.1
            delta_t = 0.01
            test_cases.append({
                'q': q,
                'omega': omega,
                'delta_t': delta_t,
                'description': f'Very large batch: N={N}',
            })

        if num_tests > 9:
            # Test 10: Small delta_t
            N = 7
            q = torch.randn(N, 4)
            q = q / torch.norm(q, dim=1, keepdim=True)
            omega = torch.randn(N, 3) * 0.1
            delta_t = 0.0001
            test_cases.append({
                'q': q,
                'omega': omega,
                'delta_t': delta_t,
                'description': f'Small delta_t: N={N}, delta_t={delta_t}',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            N = 3 + (i % 10)
            q = torch.randn(N, 4)
            q = q / torch.norm(q, dim=1, keepdim=True)
            omega = torch.randn(N, 3) * (0.1 + (i % 5) * 0.2)
            delta_t = 0.01 + (i % 3) * 0.05

            test_cases.append({
                'q': q,
                'omega': omega,
                'delta_t': delta_t,
                'description': f'Additional test {i+1}: N={N}, delta_t={delta_t}',
            })

        return test_cases[:num_tests]
