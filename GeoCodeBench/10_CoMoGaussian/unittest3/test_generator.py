"""
Test Data Generator for get_rigid_transformation and get_cmr_transformation functions.
"""

import torch


class TestDataGenerator:
    """Generate test data for functions."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5, num_views=29, view_dim=32, num_warp=9):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case
        idx_view = 0
        latent_rigid = torch.randn(num_warp, view_dim)
        latent_cmr = torch.randn(num_warp, view_dim)
        test_cases.append({
            'latent_rigid': latent_rigid,
            'latent_cmr': latent_cmr,
            'idx_view': idx_view,
            'description': f'Basic: view_dim={view_dim}, num_warp={num_warp}',
        })

        if num_tests > 1:
            # Test 2: Different view
            idx_view = 5
            latent_rigid = torch.randn(num_warp, view_dim)
            latent_cmr = torch.randn(num_warp, view_dim)
            test_cases.append({
                'latent_rigid': latent_rigid,
                'latent_cmr': latent_cmr,
                'idx_view': idx_view,
                'description': f'Different view: idx_view={idx_view}',
            })

        if num_tests > 2:
            # Test 3: Smaller num_warp
            idx_view = 0
            num_warp_small = 5
            latent_rigid = torch.randn(num_warp_small, view_dim)
            latent_cmr = torch.randn(num_warp_small, view_dim)
            test_cases.append({
                'latent_rigid': latent_rigid,
                'latent_cmr': latent_cmr,
                'idx_view': idx_view,
                'description': f'Small num_warp: {num_warp_small}',
            })

        if num_tests > 3:
            # Test 4: Larger view_dim
            idx_view = 0
            large_view_dim = 64
            large_num_warp = 10
            latent_rigid = torch.randn(large_num_warp, large_view_dim)
            latent_cmr = torch.randn(large_num_warp, large_view_dim)
            test_cases.append({
                'latent_rigid': latent_rigid,
                'latent_cmr': latent_cmr,
                'idx_view': idx_view,
                'description': f'Large view_dim: {large_view_dim}, num_warp={large_num_warp}',
            })

        if num_tests > 4:
            idx_view = 15
            latent_rigid = torch.randn(num_warp, view_dim) * 0.1
            latent_cmr = torch.randn(num_warp, view_dim) * 0.1
            test_cases.append({
                'latent_rigid': latent_rigid,
                'latent_cmr': latent_cmr,
                'idx_view': idx_view,
                'description': f'Small scale: view_idx={idx_view}',
            })

        # Generate additional tests
        for i in range(num_tests - len(test_cases)):
            idx_view = i % num_views
            test_view_dim = [16, 32, 64][i % 3]
            test_num_warp = [5, 9, 15][i % 3]
            latent_rigid = torch.randn(test_num_warp, test_view_dim)
            latent_cmr = torch.randn(test_num_warp, test_view_dim)

            test_cases.append({
                'latent_rigid': latent_rigid,
                'latent_cmr': latent_cmr,
                'idx_view': idx_view,
                'description': f'Additional test {i+1}: idx_view={idx_view}, view_dim={test_view_dim}, num_warp={test_num_warp}',
            })

        return test_cases[:num_tests]
