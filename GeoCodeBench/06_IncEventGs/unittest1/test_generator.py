"""
Test data generator for linear_interpolation function.
"""

from __future__ import annotations

import pypose as pp
import torch


class TestDataGenerator:
    """Generate test data for linear_interpolation function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with single batch
        batch_size = tuple()  # No batch
        interpolations = 5
        ctrl_knots = pp.randn_SE3(2)  # Two control knots
        u = torch.linspace(0, 1, interpolations)
        test_cases.append(
            {
                "ctrl_knots": ctrl_knots,
                "u": u,
                "enable_eps": False,
                "description": f"Basic: no batch, {interpolations} interpolations",
            }
        )

        if num_tests > 1:
            # Test 2: With batch dimension
            batch_size = (3,)
            interpolations = 10
            ctrl_knots = pp.randn_SE3(*batch_size, 2)
            u = torch.linspace(0, 1, interpolations)
            test_cases.append(
                {
                    "ctrl_knots": ctrl_knots,
                    "u": u,
                    "enable_eps": False,
                    "description": f"Batch: batch_size={batch_size}, {interpolations} interpolations",
                }
            )

        if num_tests > 2:
            # Test 3: With enable_eps
            batch_size = (2,)
            interpolations = 8
            ctrl_knots = pp.randn_SE3(*batch_size, 2)
            u = torch.linspace(0, 1, interpolations)
            test_cases.append(
                {
                    "ctrl_knots": ctrl_knots,
                    "u": u,
                    "enable_eps": True,
                    "description": f"Enable eps: batch_size={batch_size}, {interpolations} interpolations, eps=True",
                }
            )

        if num_tests > 3:
            # Test 4: Different u values
            batch_size = (4,)
            interpolations = 6
            ctrl_knots = pp.randn_SE3(*batch_size, 2)
            u = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
            test_cases.append(
                {
                    "ctrl_knots": ctrl_knots,
                    "u": u,
                    "enable_eps": False,
                    "description": f"Custom u: batch_size={batch_size}, custom u values",
                }
            )

        if num_tests > 4:
            # Test 5: Larger batch and interpolations
            batch_size = (5, 2)
            interpolations = 15
            ctrl_knots = pp.randn_SE3(*batch_size, 2)
            u = torch.linspace(0, 1, interpolations)
            test_cases.append(
                {
                    "ctrl_knots": ctrl_knots,
                    "u": u,
                    "enable_eps": False,
                    "description": f"Large: batch_size={batch_size}, {interpolations} interpolations",
                }
            )

        if num_tests > 5:
            # Test 6: Per-batch u values
            batch_size = (3,)
            interpolations = 7
            ctrl_knots = pp.randn_SE3(*batch_size, 2)
            u = torch.rand(*batch_size, interpolations)  # Different u for each batch
            test_cases.append(
                {
                    "ctrl_knots": ctrl_knots,
                    "u": u,
                    "enable_eps": False,
                    "description": f"Per-batch u: batch_size={batch_size}, different u per batch",
                }
            )

        if num_tests > 6:
            # Test 7: Edge case - only 2 interpolations
            batch_size = (2,)
            interpolations = 2
            ctrl_knots = pp.randn_SE3(*batch_size, 2)
            u = torch.tensor([0.0, 1.0])
            test_cases.append(
                {
                    "ctrl_knots": ctrl_knots,
                    "u": u,
                    "enable_eps": False,
                    "description": "Edge case: only 2 interpolations (start and end)",
                }
            )

        if num_tests > 7:
            # Test 8: Many interpolations
            batch_size = (2,)
            interpolations = 50
            ctrl_knots = pp.randn_SE3(*batch_size, 2)
            u = torch.linspace(0, 1, interpolations)
            test_cases.append(
                {
                    "ctrl_knots": ctrl_knots,
                    "u": u,
                    "enable_eps": False,
                    "description": f"Many interpolations: {interpolations} points",
                }
            )

        if num_tests > 8:
            # Test 9: 3D batch
            batch_size = (2, 3, 2)
            interpolations = 10
            ctrl_knots = pp.randn_SE3(*batch_size, 2)
            u = torch.linspace(0, 1, interpolations)
            test_cases.append(
                {
                    "ctrl_knots": ctrl_knots,
                    "u": u,
                    "enable_eps": True,
                    "description": f"3D batch: batch_size={batch_size}, {interpolations} interpolations",
                }
            )

        if num_tests > 9:
            # Test 10: Random u values
            batch_size = (4,)
            interpolations = 12
            ctrl_knots = pp.randn_SE3(*batch_size, 2)
            u = torch.sort(torch.rand(interpolations))[0]  # Random but sorted
            test_cases.append(
                {
                    "ctrl_knots": ctrl_knots,
                    "u": u,
                    "enable_eps": False,
                    "description": "Random u: random sorted u values",
                }
            )

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            batch_size = tuple([2 + (i % 3)] * (1 + i % 2))
            interpolations = 5 + i * 2
            ctrl_knots = pp.randn_SE3(*batch_size, 2)
            u = torch.linspace(0, 1, interpolations)

            test_cases.append(
                {
                    "ctrl_knots": ctrl_knots,
                    "u": u,
                    "enable_eps": i % 2 == 0,
                    "description": f"Additional test {i+1}: batch_size={batch_size}, {interpolations} interpolations",
                }
            )

        return test_cases[:num_tests]
