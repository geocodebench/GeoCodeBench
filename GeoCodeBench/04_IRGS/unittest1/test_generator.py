"""
Test Data Generator for rotation_between_z function.
Generates various test cases with different characteristics.
"""

import torch


class TestDataGenerator:
    """Generate test data for rotation_between_z."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Single vector
        test_cases.append({
            "vec": torch.randn(3),
            "description": "Single 3D vector",
        })

        if num_tests > 1:
            # Test 2: Batch of vectors
            test_cases.append({
                "vec": torch.randn(10, 3),
                "description": "Batch of 10 vectors",
            })

        if num_tests > 2:
            # Test 3: 2D grid of vectors
            test_cases.append({
                "vec": torch.randn(8, 8, 3),
                "description": "2D grid (8x8) of vectors",
            })

        if num_tests > 3:
            # Test 4: 3D batch of vectors
            test_cases.append({
                "vec": torch.randn(2, 16, 16, 3),
                "description": "3D batch (2x16x16) of vectors",
            })

        if num_tests > 4:
            # Test 5: Edge case - vectors near z-axis
            vec = torch.zeros(20, 3)
            vec[:, 2] = 0.9 + torch.rand(20) * 0.1
            vec[:, 0] = torch.randn(20) * 0.1
            vec[:, 1] = torch.randn(20) * 0.1
            test_cases.append({
                "vec": vec,
                "description": "Edge case: vectors near z-axis",
            })

        if num_tests > 5:
            # Test 6: Edge case - vectors near -z-axis
            vec = torch.zeros(20, 3)
            vec[:, 2] = -0.9 - torch.rand(20) * 0.1
            vec[:, 0] = torch.randn(20) * 0.1
            vec[:, 1] = torch.randn(20) * 0.1
            test_cases.append({
                "vec": vec,
                "description": "Edge case: vectors near -z-axis",
            })

        if num_tests > 6:
            # Test 7: Edge case - vectors on xy-plane
            vec = torch.randn(20, 3)
            vec[:, 2] = 0.0
            test_cases.append({
                "vec": vec,
                "description": "Edge case: vectors on xy-plane",
            })

        if num_tests > 7:
            # Test 8: Large batch
            test_cases.append({
                "vec": torch.randn(100, 3),
                "description": "Large batch (100 vectors)",
            })

        if num_tests > 8:
            # Test 9: Very large 2D grid
            test_cases.append({
                "vec": torch.randn(32, 32, 3),
                "description": "Large 2D grid (32x32)",
            })

        if num_tests > 9:
            # Test 10: Mixed scales
            vec = torch.randn(50, 3)
            vec = vec * (torch.rand(50, 1) * 10)
            test_cases.append({
                "vec": vec,
                "description": "Mixed magnitude vectors",
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            shape_choice = i % 3
            if shape_choice == 0:
                vec = torch.randn(20 + i * 10, 3)
            elif shape_choice == 1:
                size = 4 + i * 2
                vec = torch.randn(size, size, 3)
            else:
                b, h, w = 2, 8 + i * 4, 8 + i * 4
                vec = torch.randn(b, h, w, 3)

            test_cases.append({
                "vec": vec,
                "description": f"Additional test {i + 1}: shape {tuple(vec.shape)}",
            })

        return test_cases[:num_tests]
