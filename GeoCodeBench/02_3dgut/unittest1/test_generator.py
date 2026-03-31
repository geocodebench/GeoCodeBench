"""
Test Data Generator for _isect_tiles and _isect_offset_encode functions.
Generates various test cases with different characteristics.
"""

import numpy as np
import torch


class TestDataGenerator:
    """Generate test data for isect functions."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def generate_simple_case(self):
        """Generate simple test case with few Gaussians."""
        n_gauss = 10
        means2d = torch.rand(1, n_gauss, 2) * 256
        radii = torch.rand(1, n_gauss, 2) * 20 + 5
        depths = torch.rand(1, n_gauss) * 10 + 1

        return {
            "means2d": means2d,
            "radii": radii,
            "depths": depths,
            "tile_size": 16,
            "tile_width": 16,
            "tile_height": 16,
            "image_width": 256,
            "image_height": 256,
            "description": "Simple case: 10 Gaussians, 16x16 tiles",
        }

    def generate_medium_case(self):
        """Generate medium complexity test case."""
        n_gauss = 100
        means2d = torch.rand(2, n_gauss, 2) * 512
        radii = torch.rand(2, n_gauss, 2) * 30 + 10
        depths = torch.rand(2, n_gauss) * 20 + 1

        return {
            "means2d": means2d,
            "radii": radii,
            "depths": depths,
            "tile_size": 16,
            "tile_width": 32,
            "tile_height": 32,
            "image_width": 512,
            "image_height": 512,
            "description": "Medium case: 100 Gaussians, 2 images, 32x32 tiles",
        }

    def generate_large_case(self):
        """Generate large test case."""
        n_gauss = 500
        means2d = torch.rand(1, n_gauss, 2) * 1024
        radii = torch.rand(1, n_gauss, 2) * 40 + 5
        depths = torch.rand(1, n_gauss) * 50 + 1

        return {
            "means2d": means2d,
            "radii": radii,
            "depths": depths,
            "tile_size": 16,
            "tile_width": 64,
            "tile_height": 64,
            "image_width": 1024,
            "image_height": 1024,
            "description": "Large case: 500 Gaussians, 64x64 tiles",
        }

    def generate_edge_case_zeros(self):
        """Generate edge case with some zero radii."""
        n_gauss = 20
        means2d = torch.rand(1, n_gauss, 2) * 256
        radii = torch.rand(1, n_gauss, 2) * 15 + 5
        radii[0, ::3, :] = 0.0
        depths = torch.rand(1, n_gauss) * 10 + 1

        return {
            "means2d": means2d,
            "radii": radii,
            "depths": depths,
            "tile_size": 16,
            "tile_width": 16,
            "tile_height": 16,
            "image_width": 256,
            "image_height": 256,
            "description": "Edge case: Some Gaussians with zero radii",
        }

    def generate_clustered_case(self):
        """Generate case with clustered Gaussians."""
        n_gauss = 48
        cluster_centers = torch.tensor([[64, 64], [192, 192], [64, 192], [192, 64]])
        means2d_list = []
        for center in cluster_centers:
            cluster_means = center + torch.randn(n_gauss // 4, 2) * 10
            means2d_list.append(cluster_means)
        means2d = torch.cat(means2d_list, dim=0).unsqueeze(0)

        actual_n = means2d.shape[1]
        radii = torch.rand(1, actual_n, 2) * 10 + 5
        depths = torch.rand(1, actual_n) * 15 + 1

        return {
            "means2d": means2d,
            "radii": radii,
            "depths": depths,
            "tile_size": 16,
            "tile_width": 16,
            "tile_height": 16,
            "image_width": 256,
            "image_height": 256,
            "description": "Clustered case: 4 clusters of Gaussians",
        }

    def generate_test_suite(self, num_tests: int = 5):
        """Generate complete test suite."""
        test_cases = [self.generate_simple_case()]

        if num_tests > 1:
            test_cases.append(self.generate_edge_case_zeros())
        if num_tests > 2:
            test_cases.append(self.generate_medium_case())
        if num_tests > 3:
            test_cases.append(self.generate_clustered_case())
        if num_tests > 4:
            test_cases.append(self.generate_large_case())

        for i in range(num_tests - len(test_cases)):
            n_gauss = 50 + i * 50
            img_size = 256 + i * 128
            tile_w = 16 + i * 8
            test_cases.append(
                {
                    "means2d": torch.rand(1, n_gauss, 2) * img_size,
                    "radii": torch.rand(1, n_gauss, 2) * 20 + 5,
                    "depths": torch.rand(1, n_gauss) * 20 + 1,
                    "tile_size": 16,
                    "tile_width": tile_w,
                    "tile_height": tile_w,
                    "image_width": img_size,
                    "image_height": img_size,
                    "description": f"Additional test {i + 1}: {n_gauss} Gaussians",
                }
            )

        return test_cases[:num_tests]
