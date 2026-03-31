"""
Test data generator for interpolate_ms_features function.
"""

from __future__ import annotations

import itertools
import torch
import torch.nn as nn


def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso,
        a: float = 0.1,
        b: float = 0.5):
    """Helper function to initialize grid parameters."""
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


class TestDataGenerator:
    """Generate test data for interpolate_ms_features function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []
        device = torch.device('cpu')

        # Test 1: Basic case - 4D points (x,y,z,t), 2D grids, single scale
        pts = torch.randn(10, 4, device=device)  # 10 points, 4D (x,y,z,t)
        grid_dims = 2
        out_dim = 8
        reso = [16, 16, 16, 8]  # resolution for x,y,z,t
        ms_grids = [init_grid_param(grid_dims, 4, out_dim, reso)]
        test_cases.append({
            'pts': pts,
            'ms_grids': ms_grids,
            'grid_dimensions': grid_dims,
            'concat_features': True,
            'num_levels': None,
            'description': f'Basic: 4D points, 2D grids, 1 scale, concat=True',
        })

        if num_tests > 1:
            # Test 2: Multiple scales, concat features
            pts = torch.randn(5, 4, device=device)
            ms_grids = [
                init_grid_param(grid_dims, 4, 8, [16, 16, 16, 8]),
                init_grid_param(grid_dims, 4, 8, [32, 32, 32, 12]),
            ]
            test_cases.append({
                'pts': pts,
                'ms_grids': ms_grids,
                'grid_dimensions': grid_dims,
                'concat_features': True,
                'num_levels': None,
                'description': f'Multi-scale: 2 scales, concat=True',
            })

        if num_tests > 2:
            # Test 3: Single scale, sum features (not concat)
            pts = torch.randn(8, 4, device=device)
            ms_grids = [init_grid_param(grid_dims, 4, 4, [16, 16, 16, 8])]
            test_cases.append({
                'pts': pts,
                'ms_grids': ms_grids,
                'grid_dimensions': grid_dims,
                'concat_features': False,
                'num_levels': None,
                'description': f'Sum features: concat=False',
            })

        if num_tests > 3:
            # Test 4: Multiple scales, partial levels
            pts = torch.randn(12, 4, device=device)
            ms_grids = [
                init_grid_param(grid_dims, 4, 8, [16, 16, 16, 8]),
                init_grid_param(grid_dims, 4, 8, [32, 32, 32, 12]),
                init_grid_param(grid_dims, 4, 8, [64, 64, 64, 16]),
            ]
            test_cases.append({
                'pts': pts,
                'ms_grids': ms_grids,
                'grid_dimensions': grid_dims,
                'concat_features': True,
                'num_levels': 2,
                'description': f'Partial levels: 3 scales, use 2 levels',
            })

        if num_tests > 4:
            # Test 5: Larger input, different output dimension
            pts = torch.randn(20, 4, device=device)
            ms_grids = [init_grid_param(grid_dims, 4, 16, [32, 32, 32, 16])]
            test_cases.append({
                'pts': pts,
                'ms_grids': ms_grids,
                'grid_dimensions': grid_dims,
                'concat_features': True,
                'num_levels': None,
                'description': f'Large: 20 points, out_dim=16',
            })

        if num_tests > 5:
            # Test 6: Different resolution
            pts = torch.randn(6, 4, device=device)
            ms_grids = [init_grid_param(grid_dims, 4, 4, [8, 8, 8, 4])]
            test_cases.append({
                'pts': pts,
                'ms_grids': ms_grids,
                'grid_dimensions': grid_dims,
                'concat_features': True,
                'num_levels': None,
                'description': f'Small resolution: 8x8x8x4',
            })

        if num_tests > 6:
            # Test 7: Multiple scales with sum
            pts = torch.randn(7, 4, device=device)
            ms_grids = [
                init_grid_param(grid_dims, 4, 4, [16, 16, 16, 8]),
                init_grid_param(grid_dims, 4, 4, [24, 24, 24, 10]),
            ]
            test_cases.append({
                'pts': pts,
                'ms_grids': ms_grids,
                'grid_dimensions': grid_dims,
                'concat_features': False,
                'num_levels': None,
                'description': f'Multi-scale sum: 2 scales, concat=False',
            })

        if num_tests > 7:
            # Test 8: Edge case - single point
            pts = torch.randn(1, 4, device=device)
            ms_grids = [init_grid_param(grid_dims, 4, 8, [16, 16, 16, 8])]
            test_cases.append({
                'pts': pts,
                'ms_grids': ms_grids,
                'grid_dimensions': grid_dims,
                'concat_features': True,
                'num_levels': None,
                'description': f'Edge case: single point',
            })

        if num_tests > 8:
            # Test 9: Many points
            pts = torch.randn(50, 4, device=device)
            ms_grids = [init_grid_param(grid_dims, 4, 4, [32, 32, 32, 16])]
            test_cases.append({
                'pts': pts,
                'ms_grids': ms_grids,
                'grid_dimensions': grid_dims,
                'concat_features': True,
                'num_levels': None,
                'description': f'Many points: 50 points',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            pts = torch.randn(8 + i*2, 4, device=device)
            num_scales = (i % 3) + 1
            ms_grids = []
            for s in range(num_scales):
                ms_grids.append(init_grid_param(grid_dims, 4, 8, [16 + s*8, 16 + s*8, 16 + s*8, 8 + s*4]))

            test_cases.append({
                'pts': pts,
                'ms_grids': ms_grids,
                'grid_dimensions': grid_dims,
                'concat_features': i % 2 == 0,
                'num_levels': None,
                'description': f'Additional test {i+1}: {len(pts)} points, {num_scales} scales',
            })

        return test_cases[:num_tests]
