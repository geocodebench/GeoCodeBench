"""
Test Data Generator for solve_shift_and_scale_two_focal() function.
"""

import numpy as np


class TestDataGenerator:
    """Generate test data for solve_shift_and_scale_two_focal() function."""

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        for i in range(num_tests):
            # Setup instance (with positive depths)
            while True:
                x1 = np.c_[np.random.randn(4, 2), np.ones((4,))]
                f1_gt = 1000.0 + 2000.0 * np.random.rand(1)
                f2_gt = 1000.0 + 2000.0 * np.random.rand(1)

                d1_gt = 1.0 + 5 * np.random.rand(4)
                X = x1 * d1_gt[:, None]
                R = np.linalg.qr(np.random.randn(3, 3))[0]
                R = R * np.linalg.det(R)
                t = np.random.randn(3)
                X2 = X @ R.T + t
                d2_gt = X2[:, 2]
                x2 = X2 / d2_gt[:, None]

                # Add shared focal length
                x1[:, 0:2] *= f1_gt
                x2[:, 0:2] *= f2_gt
                x2[:, 0:2] += np.random.randn(4, 2) * 1 + 0.5
                if np.all(d2_gt > 0):
                    break

            # Shift and scale gt depths
            a1_gt = np.random.rand(1)
            b1_gt = np.random.randn(1)
            a2_gt = np.random.rand(1)
            b2_gt = np.random.randn(1)

            # d1_gt = a1 * d1 + b1
            d1 = (d1_gt - b1_gt) / a1_gt
            d2 = (d2_gt - b2_gt) / a2_gt

            test_cases.append({
                'x1': x1.copy(),
                'x2': x2.copy(),
                'd1': d1.copy(),
                'd2': d2.copy(),
                'description': f'Test {i+1}: Random configuration',
                'ground_truth': {
                    'a1_gt': a1_gt,
                    'b1_gt': b1_gt,
                    'a2_gt': a2_gt,
                    'b2_gt': b2_gt,
                    'f1_gt': f1_gt,
                    'f2_gt': f2_gt,
                }
            })

        return test_cases
