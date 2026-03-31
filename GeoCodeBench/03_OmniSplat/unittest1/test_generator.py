"""
Test Data Generator for coords_grid, yin_to_3d, yang90_from_3d functions.
Generates various test cases with different characteristics.
"""

import torch


class TestDataGenerator:
    """Generate test data for the three functions."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic small test
        test_cases.append(
            {
                "b": 1,
                "h": 8,
                "w": 8,
                "description": "Basic small test (b=1, h=8, w=8)",
                "n_points": 20,
            }
        )

        if num_tests > 1:
            # Test 2: Medium size
            test_cases.append(
                {
                    "b": 2,
                    "h": 32,
                    "w": 32,
                    "description": "Medium size (b=2, h=32, w=32)",
                    "n_points": 100,
                }
            )

        if num_tests > 2:
            # Test 3: Large size
            test_cases.append(
                {
                    "b": 1,
                    "h": 128,
                    "w": 128,
                    "description": "Large size (b=1, h=128, w=128)",
                    "n_points": 500,
                }
            )

        if num_tests > 3:
            # Test 4: Batch processing
            test_cases.append(
                {
                    "b": 4,
                    "h": 64,
                    "w": 64,
                    "description": "Batch processing (b=4, h=64, w=64)",
                    "n_points": 200,
                }
            )

        if num_tests > 4:
            # Test 5: Non-square dimensions
            test_cases.append(
                {
                    "b": 2,
                    "h": 48,
                    "w": 96,
                    "description": "Non-square dimensions (b=2, h=48, w=96)",
                    "n_points": 150,
                }
            )

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            h_val = 16 * (i + 6)
            w_val = 16 * (i + 6)
            test_cases.append(
                {
                    "b": 1 + (i % 3),
                    "h": h_val,
                    "w": w_val,
                    "description": f"Additional test {i+1} (b={1 + (i % 3)}, h={h_val}, w={w_val})",
                    "n_points": 50 + i * 50,
                }
            )

        return test_cases[:num_tests]
