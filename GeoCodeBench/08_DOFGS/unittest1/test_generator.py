"""
Test data generator for distance_to_gaussian_surface function.
"""

from __future__ import annotations

import torch


class TestDataGenerator:
    """Generate test data for distance_to_gaussian_surface function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case - single Gaussian, single query
        mean = torch.tensor([0.0, 0.0, 0.0])
        svec = torch.tensor([1.0, 1.0, 1.0])  # Unit sphere
        rotmat = torch.eye(3)
        query = torch.tensor([1.0, 0.0, 0.0])
        test_cases.append(
            {
                "mean": mean,
                "svec": svec,
                "rotmat": rotmat,
                "query": query,
                "description": "Basic: single Gaussian, single query point",
            }
        )

        if num_tests > 1:
            # Test 2: Multiple query points
            mean = torch.tensor([0.0, 0.0, 0.0])
            svec = torch.tensor([2.0, 1.0, 1.0])  # Ellipsoid
            rotmat = torch.eye(3)
            query = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            test_cases.append(
                {
                    "mean": mean,
                    "svec": svec,
                    "rotmat": rotmat,
                    "query": query,
                    "description": "Multiple queries: 3 query points",
                }
            )

        if num_tests > 2:
            # Test 3: Rotated Gaussian
            mean = torch.tensor([1.0, 1.0, 1.0])
            svec = torch.tensor([2.0, 1.0, 0.5])  # Different scales
            # Rotation around Z-axis by 45 degrees
            angle = torch.tensor(torch.pi / 4)
            rotmat = torch.tensor(
                [[torch.cos(angle), -torch.sin(angle), 0], [torch.sin(angle), torch.cos(angle), 0], [0, 0, 1]]
            )
            query = torch.tensor([[2.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
            test_cases.append(
                {
                    "mean": mean,
                    "svec": svec,
                    "rotmat": rotmat,
                    "query": query,
                    "description": "Rotated Gaussian: 45° rotation around Z-axis",
                }
            )

        if num_tests > 3:
            # Test 4: Batch of Gaussians
            batch_size = 3
            mean = torch.randn(batch_size, 3) * 2
            svec = torch.abs(torch.randn(batch_size, 3)) + 0.5  # Positive scales
            # Random rotation matrices
            rotmat = torch.randn(batch_size, 3, 3)
            rotmat = torch.linalg.qr(rotmat)[0]  # Orthogonalize
            query = torch.randn(batch_size, 3) * 3
            test_cases.append(
                {
                    "mean": mean,
                    "svec": svec,
                    "rotmat": rotmat,
                    "query": query,
                    "description": f"Batch: {batch_size} Gaussians with random parameters",
                }
            )

        if num_tests > 4:
            # Test 5: Edge case - very small scales
            mean = torch.tensor([0.0, 0.0, 0.0])
            svec = torch.tensor([0.1, 0.1, 0.1])  # Very small ellipsoid
            rotmat = torch.eye(3)
            query = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.5]])
            test_cases.append(
                {
                    "mean": mean,
                    "svec": svec,
                    "rotmat": rotmat,
                    "query": query,
                    "description": "Small scales: very small ellipsoid",
                }
            )

        if num_tests > 5:
            # Test 6: Large batch with many queries
            batch_size = 2
            num_queries = 10
            mean = torch.randn(batch_size, 3) * 0.5
            svec = torch.abs(torch.randn(batch_size, 3)) + 1.0
            rotmat = torch.randn(batch_size, 3, 3)
            rotmat = torch.linalg.qr(rotmat)[0]
            query = torch.randn(batch_size, num_queries, 3) * 2
            test_cases.append(
                {
                    "mean": mean,
                    "svec": svec,
                    "rotmat": rotmat,
                    "query": query,
                    "description": f"Large batch: {batch_size} Gaussians, {num_queries} queries each",
                }
            )

        if num_tests > 6:
            # Test 7: Extreme rotation
            mean = torch.tensor([0.0, 0.0, 0.0])
            svec = torch.tensor([3.0, 1.0, 0.5])
            # 90-degree rotation around X-axis
            rotmat = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
            query = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            test_cases.append(
                {
                    "mean": mean,
                    "svec": svec,
                    "rotmat": rotmat,
                    "query": query,
                    "description": "Extreme rotation: 90° around X-axis",
                }
            )

        if num_tests > 7:
            # Test 8: Asymmetric scales
            mean = torch.tensor([0.0, 0.0, 0.0])
            svec = torch.tensor([5.0, 0.2, 0.1])  # Very asymmetric
            rotmat = torch.eye(3)
            query = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [2.0, 0.1, 0.05]])
            test_cases.append(
                {
                    "mean": mean,
                    "svec": svec,
                    "rotmat": rotmat,
                    "query": query,
                    "description": "Asymmetric scales: very different axis lengths",
                }
            )

        if num_tests > 8:
            # Test 9: Complex rotation and translation
            mean = torch.tensor([2.0, -1.0, 0.5])
            svec = torch.tensor([1.5, 2.0, 0.8])
            # Complex rotation: 30° around X, 45° around Y, 60° around Z
            rx = torch.tensor(torch.pi / 6)
            ry = torch.tensor(torch.pi / 4)
            rz = torch.tensor(torch.pi / 3)

            Rx = torch.tensor([[1, 0, 0], [0, torch.cos(rx), -torch.sin(rx)], [0, torch.sin(rx), torch.cos(rx)]])
            Ry = torch.tensor([[torch.cos(ry), 0, torch.sin(ry)], [0, 1, 0], [-torch.sin(ry), 0, torch.cos(ry)]])
            Rz = torch.tensor([[torch.cos(rz), -torch.sin(rz), 0], [torch.sin(rz), torch.cos(rz), 0], [0, 0, 1]])
            rotmat = Rz @ Ry @ Rx
            query = torch.tensor([[3.0, 0.0, 0.5], [1.0, 1.0, 1.0], [2.5, -0.5, 0.0]])
            test_cases.append(
                {
                    "mean": mean,
                    "svec": svec,
                    "rotmat": rotmat,
                    "query": query,
                    "description": "Complex rotation: multiple axis rotations",
                }
            )

        if num_tests > 9:
            # Test 10: Random comprehensive test
            batch_size = 4
            num_queries = 8
            mean = torch.randn(batch_size, 3) * 3
            svec = torch.abs(torch.randn(batch_size, 3)) + 0.3
            rotmat = torch.randn(batch_size, 3, 3)
            rotmat = torch.linalg.qr(rotmat)[0]
            query = torch.randn(batch_size, num_queries, 3) * 4
            test_cases.append(
                {
                    "mean": mean,
                    "svec": svec,
                    "rotmat": rotmat,
                    "query": query,
                    "description": f"Random comprehensive: {batch_size} batches, {num_queries} queries",
                }
            )

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            batch_size = 1 + (i % 3)
            num_queries = 3 + i * 2
            mean = torch.randn(batch_size, 3) * (1 + i)
            svec = torch.abs(torch.randn(batch_size, 3)) + 0.5
            rotmat = torch.randn(batch_size, 3, 3)
            rotmat = torch.linalg.qr(rotmat)[0]
            query = torch.randn(batch_size, num_queries, 3) * (2 + i)

            test_cases.append(
                {
                    "mean": mean,
                    "svec": svec,
                    "rotmat": rotmat,
                    "query": query,
                    "description": f"Additional test {i+1}: batch_size={batch_size}, {num_queries} queries",
                }
            )

        return test_cases[:num_tests]
