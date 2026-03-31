"""
Test Data Generator for UnifiedPnPCoeff() function.
Generates various test cases with different configurations.
"""

from __future__ import annotations

import torch


class TestDataGenerator:
    """Generate test data for UnifiedPnPCoeff() function."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests: int = 5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small batch and few points
        batch_size = 2
        nPts = 4
        f_batch = torch.randn(batch_size, nPts, 3)
        f_batch = f_batch / torch.norm(f_batch, dim=-1, keepdim=True)
        p_batch = torch.randn(batch_size, nPts, 3, 1)
        v_batch = torch.randn(batch_size, nPts, 3)
        test_cases.append({
            'f_batch': f_batch,
            'p_batch': p_batch,
            'v_batch': v_batch,
            'description': f'Basic: batch_size={batch_size}, nPts={nPts}',
        })

        if num_tests > 1:
            batch_size = 1
            nPts = 8
            f_batch = torch.randn(batch_size, nPts, 3)
            f_batch = f_batch / torch.norm(f_batch, dim=-1, keepdim=True)
            p_batch = torch.randn(batch_size, nPts, 3, 1)
            v_batch = torch.randn(batch_size, nPts, 3)
            test_cases.append({
                'f_batch': f_batch,
                'p_batch': p_batch,
                'v_batch': v_batch,
                'description': f'Single batch: batch_size={batch_size}, nPts={nPts}',
            })

        if num_tests > 2:
            batch_size = 4
            nPts = 6
            f_batch = torch.randn(batch_size, nPts, 3)
            f_batch = f_batch / torch.norm(f_batch, dim=-1, keepdim=True)
            p_batch = torch.randn(batch_size, nPts, 3, 1)
            v_batch = torch.randn(batch_size, nPts, 3)
            test_cases.append({
                'f_batch': f_batch,
                'p_batch': p_batch,
                'v_batch': v_batch,
                'description': f'Larger batch: batch_size={batch_size}, nPts={nPts}',
            })

        if num_tests > 3:
            batch_size = 3
            nPts = 10
            f_batch = torch.randn(batch_size, nPts, 3)
            f_batch = f_batch / torch.norm(f_batch, dim=-1, keepdim=True)
            p_batch = torch.randn(batch_size, nPts, 3, 1)
            v_batch = torch.randn(batch_size, nPts, 3)
            test_cases.append({
                'f_batch': f_batch,
                'p_batch': p_batch,
                'v_batch': v_batch,
                'description': f'Many points: batch_size={batch_size}, nPts={nPts}',
            })

        if num_tests > 4:
            batch_size = 1
            nPts = 3
            f_batch = torch.randn(batch_size, nPts, 3)
            f_batch = f_batch / torch.norm(f_batch, dim=-1, keepdim=True)
            p_batch = torch.randn(batch_size, nPts, 3, 1)
            v_batch = torch.randn(batch_size, nPts, 3)
            test_cases.append({
                'f_batch': f_batch,
                'p_batch': p_batch,
                'v_batch': v_batch,
                'description': f'Minimal: batch_size={batch_size}, nPts={nPts}',
            })

        if num_tests > 5:
            batch_size = 8
            nPts = 5
            f_batch = torch.randn(batch_size, nPts, 3)
            f_batch = f_batch / torch.norm(f_batch, dim=-1, keepdim=True)
            p_batch = torch.randn(batch_size, nPts, 3, 1)
            v_batch = torch.randn(batch_size, nPts, 3)
            test_cases.append({
                'f_batch': f_batch,
                'p_batch': p_batch,
                'v_batch': v_batch,
                'description': f'Large batch: batch_size={batch_size}, nPts={nPts}',
            })

        if num_tests > 6:
            batch_size = 2
            nPts = 4
            f_batch = torch.randn(batch_size, nPts, 3)
            f_batch = f_batch / torch.norm(f_batch, dim=-1, keepdim=True)
            p_batch = torch.randn(batch_size, nPts, 3, 1)
            v_batch = torch.zeros(batch_size, nPts, 3)
            test_cases.append({
                'f_batch': f_batch,
                'p_batch': p_batch,
                'v_batch': v_batch,
                'description': f'Zero v_batch: batch_size={batch_size}, nPts={nPts}',
            })

        if num_tests > 7:
            batch_size = 2
            nPts = 15
            f_batch = torch.randn(batch_size, nPts, 3)
            f_batch = f_batch / torch.norm(f_batch, dim=-1, keepdim=True)
            p_batch = torch.randn(batch_size, nPts, 3, 1)
            v_batch = torch.randn(batch_size, nPts, 3)
            test_cases.append({
                'f_batch': f_batch,
                'p_batch': p_batch,
                'v_batch': v_batch,
                'description': f'Many points: batch_size={batch_size}, nPts={nPts}',
            })

        if num_tests > 8:
            batch_size = 3
            nPts = 5
            f_batch = torch.randn(batch_size, nPts, 3)
            f_batch = f_batch / torch.norm(f_batch, dim=-1, keepdim=True)
            p_batch = torch.randn(batch_size, nPts, 3, 1)
            p_batch = p_batch / (torch.norm(p_batch, dim=2, keepdim=True) + 1e-8)
            v_batch = torch.randn(batch_size, nPts, 3)
            v_batch = v_batch / (torch.norm(v_batch, dim=-1, keepdim=True) + 1e-8)
            test_cases.append({
                'f_batch': f_batch,
                'p_batch': p_batch,
                'v_batch': v_batch,
                'description': f'Normalized inputs: batch_size={batch_size}, nPts={nPts}',
            })

        for i in range(num_tests - len(test_cases)):
            batch_size = 1 + (i % 5)
            nPts = 3 + (i % 10)
            f_batch = torch.randn(batch_size, nPts, 3)
            f_batch = f_batch / torch.norm(f_batch, dim=-1, keepdim=True)
            p_batch = torch.randn(batch_size, nPts, 3, 1)
            v_batch = torch.randn(batch_size, nPts, 3)
            test_cases.append({
                'f_batch': f_batch,
                'p_batch': p_batch,
                'v_batch': v_batch,
                'description': f'Additional test {i+1}: batch_size={batch_size}, nPts={nPts}',
            })

        return test_cases[:num_tests]
