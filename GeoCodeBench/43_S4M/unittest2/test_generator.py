"""
Test Data Generator for SetCriterion.calc_similarity_map() function.
Generates test cases with different configurations.
"""

from __future__ import annotations

import torch


class TestDataGenerator:
    """Generate test data for SetCriterion.calc_similarity_map() function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small batch
        N, res, C, P = 2, 8, 64, 4
        feats = torch.randn(N, res * res, C)
        point_coords = torch.randint(0, res, (N, P, 2)).float()
        test_cases.append({
            'feats': feats,
            'point_coords': point_coords,
            'description': f'Basic: N={N}, res={res}, C={C}, P={P}',
        })

        if num_tests > 1:
            # Test 2: Single sample
            N, res, C, P = 1, 4, 32, 2
            feats = torch.randn(N, res * res, C)
            point_coords = torch.randint(0, res, (N, P, 2)).float()
            test_cases.append({
                'feats': feats,
                'point_coords': point_coords,
                'description': f'Single sample: N={N}, res={res}, C={C}, P={P}',
            })

        if num_tests > 2:
            # Test 3: Larger batch
            N, res, C, P = 4, 16, 128, 8
            feats = torch.randn(N, res * res, C)
            point_coords = torch.randint(0, res, (N, P, 2)).float()
            test_cases.append({
                'feats': feats,
                'point_coords': point_coords,
                'description': f'Larger batch: N={N}, res={res}, C={C}, P={P}',
            })

        if num_tests > 3:
            # Test 4: Medium batch, different resolution
            N, res, C, P = 3, 32, 256, 16
            feats = torch.randn(N, res * res, C)
            point_coords = torch.randint(0, res, (N, P, 2)).float()
            test_cases.append({
                'feats': feats,
                'point_coords': point_coords,
                'description': f'Medium batch: N={N}, res={res}, C={C}, P={P}',
            })

        if num_tests > 4:
            # Test 5: Small resolution
            N, res, C, P = 2, 2, 16, 1
            feats = torch.randn(N, res * res, C)
            point_coords = torch.randint(0, res, (N, P, 2)).float()
            test_cases.append({
                'feats': feats,
                'point_coords': point_coords,
                'description': f'Small resolution: N={N}, res={res}, C={C}, P={P}',
            })

        if num_tests > 5:
            # Test 6: Large batch
            N, res, C, P = 8, 8, 64, 32
            feats = torch.randn(N, res * res, C)
            point_coords = torch.randint(0, res, (N, P, 2)).float()
            test_cases.append({
                'feats': feats,
                'point_coords': point_coords,
                'description': f'Large batch: N={N}, res={res}, C={C}, P={P}',
            })

        if num_tests > 6:
            # Test 7: Edge case - all zeros input
            N, res, C, P = 2, 4, 32, 2
            feats = torch.zeros(N, res * res, C)
            point_coords = torch.randint(0, res, (N, P, 2)).float()
            test_cases.append({
                'feats': feats,
                'point_coords': point_coords,
                'description': f'Edge case: N={N}, zero feats, res={res}, C={C}, P={P}',
            })

        if num_tests > 7:
            # Test 8: Many points
            N, res, C, P = 2, 8, 64, 64
            feats = torch.randn(N, res * res, C)
            point_coords = torch.randint(0, res, (N, P, 2)).float()
            test_cases.append({
                'feats': feats,
                'point_coords': point_coords,
                'description': f'Many points: N={N}, res={res}, C={C}, P={P}',
            })

        if num_tests > 8:
            # Test 9: Very large resolution
            N, res, C, P = 1, 64, 128, 8
            feats = torch.randn(N, res * res, C)
            point_coords = torch.randint(0, res, (N, P, 2)).float()
            test_cases.append({
                'feats': feats,
                'point_coords': point_coords,
                'description': f'Very large resolution: N={N}, res={res}, C={C}, P={P}',
            })

        if num_tests > 9:
            # Test 10: Normalized input
            N, res, C, P = 3, 8, 64, 4
            feats = torch.randn(N, res * res, C)
            feats = (feats - feats.mean(dim=-1, keepdim=True)) / (feats.std(dim=-1, keepdim=True) + 1e-8)
            point_coords = torch.randint(0, res, (N, P, 2)).float()
            test_cases.append({
                'feats': feats,
                'point_coords': point_coords,
                'description': f'Normalized input: N={N}, res={res}, C={C}, P={P}',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            N = 2 + (i % 5)
            res = 4 + (i % 8) * 2
            C = 32 + (i % 8) * 32
            P = 2 + (i % 10)
            feats = torch.randn(N, res * res, C)
            point_coords = torch.randint(0, res, (N, P, 2)).float()

            test_cases.append({
                'feats': feats,
                'point_coords': point_coords,
                'description': f'Additional test {i+1}: N={N}, res={res}, C={C}, P={P}',
            })

        return test_cases[:num_tests]
