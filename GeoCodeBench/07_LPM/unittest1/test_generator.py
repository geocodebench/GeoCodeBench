"""
Test data generator for finite cone functions.
"""

from __future__ import annotations

import torch


class TestDataGenerator:
    """Generate test data for finite_cone_formulation and points_in_finite_cone."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case
        top_point = torch.tensor([0.0, 0.0, 5.0])
        base_center = torch.tensor([0.0, 0.0, 0.0])
        radius = 2.0
        test_cases.append(
            {
                "top_point": top_point,
                "base_center": base_center,
                "radius": radius,
                "description": "Basic: simple cone with apex at (0,0,5), base at origin, radius 2",
            }
        )

        if num_tests > 1:
            top_point = torch.tensor([1.0, 2.0, 3.0])
            base_center = torch.tensor([0.0, 0.0, 0.0])
            radius = 1.5
            test_cases.append(
                {
                    "top_point": top_point,
                    "base_center": base_center,
                    "radius": radius,
                    "description": "Oriented: apex at (1,2,3), base at origin, radius 1.5",
                }
            )

        if num_tests > 2:
            top_point = torch.tensor([0.0, 0.0, 10.0])
            base_center = torch.tensor([0.0, 0.0, 0.0])
            radius = 8.0
            test_cases.append(
                {
                    "top_point": top_point,
                    "base_center": base_center,
                    "radius": radius,
                    "description": "Large radius: apex at (0,0,10), base at origin, radius 8",
                }
            )

        if num_tests > 3:
            top_point = torch.tensor([0.0, 0.0, 1.0])
            base_center = torch.tensor([0.0, 0.0, 0.0])
            radius = 0.1
            test_cases.append(
                {
                    "top_point": top_point,
                    "base_center": base_center,
                    "radius": radius,
                    "description": "Small radius: apex at (0,0,1), base at origin, radius 0.1",
                }
            )

        if num_tests > 4:
            top_point = torch.tensor([0.0, 0.0, 3.0])
            base_center = torch.tensor([1.0, 1.0, 0.0])
            radius = 2.0
            test_cases.append(
                {
                    "top_point": top_point,
                    "base_center": base_center,
                    "radius": radius,
                    "description": "Off-center: apex at (0,0,3), base at (1,1,0), radius 2",
                }
            )

        if num_tests > 5:
            top_point = torch.tensor([0.0, 0.0, 100.0])
            base_center = torch.tensor([0.0, 0.0, 0.0])
            radius = 1.0
            test_cases.append(
                {
                    "top_point": top_point,
                    "base_center": base_center,
                    "radius": radius,
                    "description": "Tall cone: apex at (0,0,100), base at origin, radius 1",
                }
            )

        if num_tests > 6:
            top_point = torch.tensor([0.0, 0.0, 1.0])
            base_center = torch.tensor([0.0, 0.0, 0.0])
            radius = 5.0
            test_cases.append(
                {
                    "top_point": top_point,
                    "base_center": base_center,
                    "radius": radius,
                    "description": "Wide cone: apex at (0,0,1), base at origin, radius 5",
                }
            )

        if num_tests > 7:
            top_point = torch.randn(3) * 5
            base_center = torch.randn(3) * 2
            radius = torch.rand(1).item() * 3 + 0.5
            test_cases.append(
                {
                    "top_point": top_point,
                    "base_center": base_center,
                    "radius": radius,
                    "description": "Random: random apex, base, and radius",
                }
            )

        if num_tests > 8:
            top_point = torch.tensor([0.0, 0.0, 0.01])
            base_center = torch.tensor([0.0, 0.0, 0.0])
            radius = 1.0
            test_cases.append(
                {
                    "top_point": top_point,
                    "base_center": base_center,
                    "radius": radius,
                    "description": "Edge case: very small height (0.01)",
                }
            )

        if num_tests > 9:
            top_point = torch.tensor([2.0, -1.0, 4.0])
            base_center = torch.tensor([-1.0, 3.0, 0.0])
            radius = 2.5
            test_cases.append(
                {
                    "top_point": top_point,
                    "base_center": base_center,
                    "radius": radius,
                    "description": "Complex: apex at (2,-1,4), base at (-1,3,0), radius 2.5",
                }
            )

        for i in range(num_tests - len(test_cases)):
            top_point = torch.randn(3) * (2 + i)
            base_center = torch.randn(3) * (1 + i)
            radius = torch.rand(1).item() * (1 + i) + 0.5
            test_cases.append(
                {
                    "top_point": top_point,
                    "base_center": base_center,
                    "radius": radius,
                    "description": f"Additional test {i+1}: random configuration",
                }
            )

        return test_cases[:num_tests]

    def generate_points_for_cone_test(self, top_point, base_center, radius, num_points=100):
        """Generate test points for points_in_finite_cone testing."""
        height = torch.norm(top_point - base_center).item()
        max_distance = max(height, radius) * 2

        min_coords = torch.min(top_point, base_center) - max_distance
        max_coords = torch.max(top_point, base_center) + max_distance

        points = torch.rand(num_points, 3) * (max_coords - min_coords) + min_coords

        axis_points = top_point.unsqueeze(0) + torch.linspace(0, 1, 10).unsqueeze(1) * (
            base_center - top_point
        ).unsqueeze(0)
        points = torch.cat([points, axis_points], dim=0)

        far_points = torch.randn(10, 3) * max_distance * 3
        points = torch.cat([points, far_points], dim=0)

        return points
