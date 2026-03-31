"""
Test data generator for forward_event function.
"""

import torch


class TestDataGenerator:
    """Generate test data for forward_event function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with even batch size
        batch_size = 10
        rays_o = torch.randn(batch_size, 3)
        rays_d = torch.randn(batch_size, 3)
        rays_d = rays_d / rays_d.norm(dim=1, keepdim=True)  # Normalize directions
        rgb_rendered = torch.rand(batch_size)  # Grayscale values in [0, 1]
        test_cases.append(
            {
                "rays_o": rays_o,
                "rays_d": rays_d,
                "rgb_rendered": rgb_rendered,
                "description": f"Basic: batch_size={batch_size}, grayscale values",
            }
        )

        if num_tests > 1:
            # Test 2: Larger batch size
            batch_size = 20
            rays_o = torch.randn(batch_size, 3)
            rays_d = torch.randn(batch_size, 3)
            rays_d = rays_d / rays_d.norm(dim=1, keepdim=True)
            rgb_rendered = torch.rand(batch_size)
            test_cases.append(
                {
                    "rays_o": rays_o,
                    "rays_d": rays_d,
                    "rgb_rendered": rgb_rendered,
                    "description": f"Larger batch: batch_size={batch_size}",
                }
            )

        if num_tests > 2:
            # Test 3: Odd batch size (tests rounding)
            batch_size = 15
            rays_o = torch.randn(batch_size, 3)
            rays_d = torch.randn(batch_size, 3)
            rays_d = rays_d / rays_d.norm(dim=1, keepdim=True)
            rgb_rendered = torch.rand(batch_size)
            test_cases.append(
                {
                    "rays_o": rays_o,
                    "rays_d": rays_d,
                    "rgb_rendered": rgb_rendered,
                    "description": f"Odd batch: batch_size={batch_size} (tests rounding)",
                }
            )

        if num_tests > 3:
            # Test 4: RGB values (Bs, 3) instead of grayscale
            batch_size = 12
            rays_o = torch.randn(batch_size, 3)
            rays_d = torch.randn(batch_size, 3)
            rays_d = rays_d / rays_d.norm(dim=1, keepdim=True)
            rgb_rendered = torch.rand(batch_size, 3)  # RGB values
            test_cases.append(
                {
                    "rays_o": rays_o,
                    "rays_d": rays_d,
                    "rgb_rendered": rgb_rendered,
                    "description": f"RGB values: batch_size={batch_size}, shape (Bs, 3)",
                }
            )

        if num_tests > 4:
            # Test 5: Small batch
            batch_size = 2
            rays_o = torch.randn(batch_size, 3)
            rays_d = torch.randn(batch_size, 3)
            rays_d = rays_d / rays_d.norm(dim=1, keepdim=True)
            rgb_rendered = torch.rand(batch_size)
            test_cases.append(
                {
                    "rays_o": rays_o,
                    "rays_d": rays_d,
                    "rgb_rendered": rgb_rendered,
                    "description": f"Small batch: batch_size={batch_size}",
                }
            )

        if num_tests > 5:
            # Test 6: Large batch
            batch_size = 100
            rays_o = torch.randn(batch_size, 3)
            rays_d = torch.randn(batch_size, 3)
            rays_d = rays_d / rays_d.norm(dim=1, keepdim=True)
            rgb_rendered = torch.rand(batch_size)
            test_cases.append(
                {
                    "rays_o": rays_o,
                    "rays_d": rays_d,
                    "rgb_rendered": rgb_rendered,
                    "description": f"Large batch: batch_size={batch_size}",
                }
            )

        if num_tests > 6:
            # Test 7: Values close to zero (tests log stability)
            batch_size = 16
            rays_o = torch.randn(batch_size, 3)
            rays_d = torch.randn(batch_size, 3)
            rays_d = rays_d / rays_d.norm(dim=1, keepdim=True)
            rgb_rendered = torch.rand(batch_size) * 0.01  # Very small values
            test_cases.append(
                {
                    "rays_o": rays_o,
                    "rays_d": rays_d,
                    "rgb_rendered": rgb_rendered,
                    "description": "Small values: tests log(x + eps) stability",
                }
            )

        if num_tests > 7:
            # Test 8: Column vector shape (Bs, 1)
            batch_size = 14
            rays_o = torch.randn(batch_size, 3)
            rays_d = torch.randn(batch_size, 3)
            rays_d = rays_d / rays_d.norm(dim=1, keepdim=True)
            rgb_rendered = torch.rand(batch_size, 1)  # Column vector
            test_cases.append(
                {
                    "rays_o": rays_o,
                    "rays_d": rays_d,
                    "rgb_rendered": rgb_rendered,
                    "description": "Column vector: shape (Bs, 1)",
                }
            )

        if num_tests > 8:
            # Test 9: Large values
            batch_size = 18
            rays_o = torch.randn(batch_size, 3)
            rays_d = torch.randn(batch_size, 3)
            rays_d = rays_d / rays_d.norm(dim=1, keepdim=True)
            rgb_rendered = torch.rand(batch_size) * 10.0  # Large values
            test_cases.append(
                {
                    "rays_o": rays_o,
                    "rays_d": rays_d,
                    "rgb_rendered": rgb_rendered,
                    "description": "Large values: rgb in [0, 10]",
                }
            )

        if num_tests > 9:
            # Test 10: Mixed scenario
            batch_size = 50
            rays_o = torch.randn(batch_size, 3)
            rays_d = torch.randn(batch_size, 3)
            rays_d = rays_d / rays_d.norm(dim=1, keepdim=True)
            rgb_rendered = torch.rand(batch_size) * 5.0
            test_cases.append(
                {
                    "rays_o": rays_o,
                    "rays_d": rays_d,
                    "rgb_rendered": rgb_rendered,
                    "description": f"Mixed: batch_size={batch_size}, values in [0, 5]",
                }
            )

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            batch_size = 8 + i * 4
            rays_o = torch.randn(batch_size, 3)
            rays_d = torch.randn(batch_size, 3)
            rays_d = rays_d / rays_d.norm(dim=1, keepdim=True)
            rgb_rendered = torch.rand(batch_size) * (1 + i * 0.5)

            test_cases.append(
                {
                    "rays_o": rays_o,
                    "rays_d": rays_d,
                    "rgb_rendered": rgb_rendered,
                    "description": f"Additional test {i+1}: batch_size={batch_size}",
                }
            )

        return test_cases[:num_tests]
