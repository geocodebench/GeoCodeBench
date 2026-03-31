"""
Test data generator for get_rays_intersection.
"""

from __future__ import annotations

import torch


class TestDataGenerator:
    """Generate test data for get_rays_intersection."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_rays(self, num_rays, device=None):
        """Generate random rays for testing."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        origins = torch.randn(num_rays, 3, device=device)
        directions = torch.randn(num_rays, 3, device=device)
        directions = directions / torch.norm(directions, dim=1, keepdim=True)
        rays = torch.cat([origins, directions], dim=1)
        return rays

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        ray_group_a = self.generate_rays(3)
        ray_group_b = self.generate_rays(3)
        test_cases.append(
            {
                "ray_group_a": ray_group_a,
                "ray_group_b": ray_group_b,
                "description": "Basic: 3 rays each group",
            }
        )

        if num_tests > 1:
            ray_group_a = self.generate_rays(5)
            ray_group_b = self.generate_rays(5)
            test_cases.append(
                {
                    "ray_group_a": ray_group_a,
                    "ray_group_b": ray_group_b,
                    "description": "Same sizes: 5 vs 5 rays",
                }
            )

        if num_tests > 2:
            ray_group_a = self.generate_rays(10)
            ray_group_b = self.generate_rays(10)
            test_cases.append(
                {
                    "ray_group_a": ray_group_a,
                    "ray_group_b": ray_group_b,
                    "description": "Larger: 10 vs 10 rays",
                }
            )

        if num_tests > 3:
            ray_group_a = self.generate_rays(1)
            ray_group_b = self.generate_rays(1)
            test_cases.append(
                {
                    "ray_group_a": ray_group_a,
                    "ray_group_b": ray_group_b,
                    "description": "Single rays: 1 vs 1",
                }
            )

        if num_tests > 4:
            ray_group_a = self.generate_rays(20)
            ray_group_b = self.generate_rays(20)
            test_cases.append(
                {
                    "ray_group_a": ray_group_a,
                    "ray_group_b": ray_group_b,
                    "description": "Many rays: 20 vs 20",
                }
            )

        if num_tests > 5:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            origins_a = torch.randn(5, 3, device=device)
            directions_a = torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(5, 1)
            ray_group_a = torch.cat([origins_a, directions_a], dim=1)

            origins_b = torch.randn(5, 3, device=device)
            directions_b = torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(5, 1)
            ray_group_b = torch.cat([origins_b, directions_b], dim=1)

            test_cases.append(
                {
                    "ray_group_a": ray_group_a,
                    "ray_group_b": ray_group_b,
                    "description": "Parallel rays: should handle edge case",
                }
            )

        if num_tests > 6:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            origins_a = torch.randn(4, 3, device=device)
            directions_a = torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(4, 1)
            ray_group_a = torch.cat([origins_a, directions_a], dim=1)

            origins_b = torch.randn(4, 3, device=device)
            directions_b = torch.tensor([[0.0, 1.0, 0.0]], device=device).repeat(4, 1)
            ray_group_b = torch.cat([origins_b, directions_b], dim=1)

            test_cases.append(
                {
                    "ray_group_a": ray_group_a,
                    "ray_group_b": ray_group_b,
                    "description": "Orthogonal rays: X vs Y directions",
                }
            )

        if num_tests > 7:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            origin = torch.randn(1, 3, device=device)
            origins_a = origin.repeat(6, 1)
            directions_a = torch.randn(6, 3, device=device)
            directions_a = directions_a / torch.norm(directions_a, dim=1, keepdim=True)
            ray_group_a = torch.cat([origins_a, directions_a], dim=1)

            origins_b = origin.repeat(6, 1)
            directions_b = torch.randn(6, 3, device=device)
            directions_b = directions_b / torch.norm(directions_b, dim=1, keepdim=True)
            ray_group_b = torch.cat([origins_b, directions_b], dim=1)

            test_cases.append(
                {
                    "ray_group_a": ray_group_a,
                    "ray_group_b": ray_group_b,
                    "description": "Same origin: different directions",
                }
            )

        if num_tests > 8:
            ray_group_a = self.generate_rays(15)
            ray_group_b = self.generate_rays(15)
            test_cases.append(
                {
                    "ray_group_a": ray_group_a,
                    "ray_group_b": ray_group_b,
                    "description": "Medium sizes: 15 vs 15",
                }
            )

        if num_tests > 9:
            ray_group_a = self.generate_rays(30)
            ray_group_b = self.generate_rays(30)
            test_cases.append(
                {
                    "ray_group_a": ray_group_a,
                    "ray_group_b": ray_group_b,
                    "description": "Large test: 30 vs 30 rays",
                }
            )

        for i in range(num_tests - len(test_cases)):
            num_rays = 2 + (i % 10)
            ray_group_a = self.generate_rays(num_rays)
            ray_group_b = self.generate_rays(num_rays)
            test_cases.append(
                {
                    "ray_group_a": ray_group_a,
                    "ray_group_b": ray_group_b,
                    "description": f"Additional test {i+1}: {num_rays} vs {num_rays} rays",
                }
            )

        return test_cases[:num_tests]
