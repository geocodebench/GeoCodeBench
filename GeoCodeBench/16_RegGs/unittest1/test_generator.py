"""
Test data generator for SinkhornDistance.compute_cost_matrix.
"""

import torch


class TestDataGenerator:
    def __init__(self, seed: int = 42):
        torch.manual_seed(seed)

    @staticmethod
    def _rand_psd(n: int, d: int):
        A = torch.randn(n, d, d)
        cov = A @ A.transpose(-1, -2)
        cov = cov + 0.1 * torch.eye(d).expand_as(cov)
        return cov

    def generate(self, num_tests: int = 5):
        cases = []
        dims = [3, 3, 3, 3, 3]
        sizes = [(16, 20), (8, 8), (32, 16), (5, 7), (50, 50)]
        for i in range(num_tests):
            d = dims[min(i, len(dims)-1)]
            na, nb = sizes[min(i, len(sizes)-1)]
            mu_A = torch.randn(na, d)
            mu_B = torch.randn(nb, d)
            cov_A = self._rand_psd(na, d)
            cov_B = self._rand_psd(nb, d)
            cases.append({
                'desc': f"case#{i+1}: A={na} B={nb} d={d}",
                'mu_A': mu_A,
                'mu_B': mu_B,
                'cov_A': cov_A,
                'cov_B': cov_B,
            })
        return cases[:num_tests]
