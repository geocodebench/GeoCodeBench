"""
Test data generator for simplify_gmm_vectorized.
"""

from __future__ import annotations

import torch
from typing import Any, Dict, List


class TestDataGenerator:
    """Generate test data for simplify_gmm_vectorized."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        torch.manual_seed(seed)

    @staticmethod
    def _make_spd(batch: int) -> torch.Tensor:
        # Generate random SPD 3x3 matrices by A A^T + eps I
        A = torch.randn(batch, 3, 3)
        cov = torch.matmul(A, A.transpose(1, 2))
        cov = cov + 0.1 * torch.eye(3).unsqueeze(0)
        return cov

    def generate_test_suite(self, num_tests: int = 5) -> List[Dict[str, Any]]:
        tests: List[Dict[str, Any]] = []

        # Test 1: Small
        N = 64
        mu = torch.randn(N, 3)
        cov = self._make_spd(N)
        w = torch.softmax(torch.randn(N), dim=0)
        num_clusters = 8
        tests.append({
            'mu': mu,
            'cov': cov,
            'w': w,
            'num_clusters': num_clusters,
            'description': f'Small N={N}, K={num_clusters}'
        })

        if num_tests > 1:
            # Test 2: Medium with more clusters
            N = 200
            mu = torch.randn(N, 3)
            cov = self._make_spd(N)
            w = torch.softmax(torch.randn(N), dim=0)
            num_clusters = 20
            tests.append({
                'mu': mu,
                'cov': cov,
                'w': w,
                'num_clusters': num_clusters,
                'description': f'Medium N={N}, K={num_clusters}'
            })

        if num_tests > 2:
            # Test 3: Uneven weights (sparse)
            N = 150
            mu = torch.randn(N, 3)
            cov = self._make_spd(N)
            w = torch.zeros(N)
            w[:10] = torch.rand(10)
            w = w / (w.sum() + 1e-8)
            num_clusters = 12
            tests.append({
                'mu': mu,
                'cov': cov,
                'w': w,
                'num_clusters': num_clusters,
                'description': f'Sparse weights N={N}, K={num_clusters}'
            })

        if num_tests > 3:
            # Test 4: Larger N than clusters by far
            N = 512
            mu = torch.randn(N, 3)
            cov = self._make_spd(N)
            w = torch.softmax(torch.randn(N), dim=0)
            num_clusters = 16
            tests.append({
                'mu': mu,
                'cov': cov,
                'w': w,
                'num_clusters': num_clusters,
                'description': f'Large N={N}, K={num_clusters}'
            })

        if num_tests > 4:
            # Test 5: Clusters close to N
            N = 60
            mu = torch.randn(N, 3)
            cov = self._make_spd(N)
            w = torch.softmax(torch.randn(N), dim=0)
            num_clusters = 50
            tests.append({
                'mu': mu,
                'cov': cov,
                'w': w,
                'num_clusters': num_clusters,
                'description': f'Close K~N, N={N}, K={num_clusters}'
            })

        # Fill additional tests by pattern
        for i in range(num_tests - len(tests)):
            N = 100 + 20 * i
            mu = torch.randn(N, 3)
            cov = self._make_spd(N)
            w = torch.softmax(torch.randn(N), dim=0)
            K = max(5, N // (10 + (i % 5)))
            tests.append({
                'mu': mu,
                'cov': cov,
                'w': w,
                'num_clusters': K,
                'description': f'Auto-generated N={N}, K={K}'
            })

        return tests[:num_tests]
