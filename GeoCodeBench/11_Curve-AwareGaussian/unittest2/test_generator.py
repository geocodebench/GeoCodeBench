"""
Test Data Generator for bezier_curve_length function.
"""

import numpy as np


class TestDataGenerator:
    """Generate test data for bezier_curve_length function."""

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic linear curve (degree 1, should be exact)
        control_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        num_samples = 10
        test_cases.append({
            'control_points': control_points,
            'num_samples': num_samples,
            'description': f'Basic linear curve (degree 1), samples={num_samples}',
        })

        if num_tests > 1:
            # Test 2: Quadratic Bezier curve
            control_points = np.array([
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 0.0, 0.0]
            ], dtype=np.float64)
            num_samples = 20
            test_cases.append({
                'control_points': control_points,
                'num_samples': num_samples,
                'description': f'Quadratic Bezier (degree 2), samples={num_samples}',
            })

        if num_tests > 2:
            # Test 3: Cubic Bezier curve
            control_points = np.array([
                [0.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.5, 1.0, 0.0],
                [2.0, 0.0, 0.0]
            ], dtype=np.float64)
            num_samples = 30
            test_cases.append({
                'control_points': control_points,
                'num_samples': num_samples,
                'description': f'Cubic Bezier (degree 3), samples={num_samples}',
            })

        if num_tests > 3:
            # Test 4: 3D Bezier curve with random control points
            control_points = np.random.rand(5, 3).astype(np.float64) * 10.0
            num_samples = 25
            test_cases.append({
                'control_points': control_points,
                'num_samples': num_samples,
                'description': f'Random 4th-degree Bezier (degree 4), samples={num_samples}',
            })

        if num_tests > 4:
            # Test 5: High-degree Bezier curve
            control_points = np.random.rand(8, 3).astype(np.float64) * 5.0
            num_samples = 40
            test_cases.append({
                'control_points': control_points,
                'num_samples': num_samples,
                'description': f'High-degree Bezier (degree 7), samples={num_samples}',
            })

        if num_tests > 5:
            # Test 6: Small num_samples (edge case)
            control_points = np.array([
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 1.0],
                [2.0, 2.0, 0.0],
                [3.0, 0.0, 0.0]
            ], dtype=np.float64)
            num_samples = 5
            test_cases.append({
                'control_points': control_points,
                'num_samples': num_samples,
                'description': f'Small num_samples (edge case), samples={num_samples}',
            })

        if num_tests > 6:
            # Test 7: Large num_samples (high accuracy)
            control_points = np.array([
                [0.0, 0.0, 0.0],
                [0.5, 1.5, 0.5],
                [1.5, 1.5, 0.5],
                [2.0, 0.0, 0.0]
            ], dtype=np.float64)
            num_samples = 100
            test_cases.append({
                'control_points': control_points,
                'num_samples': num_samples,
                'description': f'Large num_samples (high accuracy), samples={num_samples}',
            })

        if num_tests > 7:
            # Test 8: Curve with zero length in one direction
            control_points = np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0]
            ], dtype=np.float64)
            num_samples = 20
            test_cases.append({
                'control_points': control_points,
                'num_samples': num_samples,
                'description': f'Curve in 1D (no Y/Z component), samples={num_samples}',
            })

        if num_tests > 8:
            # Test 9: Complex 3D curve
            control_points = np.array([
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 1.0, -1.0],
                [3.0, 0.0, 0.0],
                [4.0, -1.0, 1.0]
            ], dtype=np.float64)
            num_samples = 35
            test_cases.append({
                'control_points': control_points,
                'num_samples': num_samples,
                'description': f'Complex 3D curve (degree 4), samples={num_samples}',
            })

        if num_tests > 9:
            # Test 10: Degenerate curve (all points collinear)
            control_points = np.array([
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]
            ], dtype=np.float64)
            num_samples = 20
            test_cases.append({
                'control_points': control_points,
                'num_samples': num_samples,
                'description': f'Degenerate curve (collinear points), samples={num_samples}',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            degree = 3 + (i % 4)
            control_points = np.random.rand(degree + 1, 3).astype(np.float64) * 5.0
            num_samples = 20 + i * 5

            test_cases.append({
                'control_points': control_points,
                'num_samples': num_samples,
                'description': f'Additional test {i+1}: degree {degree}, samples={num_samples}',
            })

        return test_cases[:num_tests]
