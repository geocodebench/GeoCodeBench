"""
Test Data Generator for project_simplex() function.
Generates various test cases with different configurations.
"""

import numpy as np


class TestDataGenerator:
    """Generate test data for project_simplex() function."""

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small batch, z=1
        num_samples = 3
        feature_dim = 5
        V = np.random.randn(num_samples, feature_dim)
        z = 1
        test_cases.append({
            'V': V,
            'z': z,
            'description': f'Basic: num_samples={num_samples}, feature_dim={feature_dim}, z={z}',
        })

        if num_tests > 1:
            # Test 2: Single sample, z=1
            num_samples = 1
            feature_dim = 4
            V = np.random.randn(num_samples, feature_dim)
            z = 1
            test_cases.append({
                'V': V,
                'z': z,
                'description': f'Single sample: num_samples={num_samples}, feature_dim={feature_dim}, z={z}',
            })

        if num_tests > 2:
            # Test 3: Larger batch, z=2.5
            num_samples = 5
            feature_dim = 6
            V = np.random.randn(num_samples, feature_dim)
            z = 2.5
            test_cases.append({
                'V': V,
                'z': z,
                'description': f'Larger batch: num_samples={num_samples}, feature_dim={feature_dim}, z={z}',
            })

        if num_tests > 3:
            # Test 4: Medium batch, z as array
            num_samples = 4
            feature_dim = 7
            V = np.random.randn(num_samples, feature_dim)
            z = np.array([1.0, 2.0, 3.0, 1.5])
            test_cases.append({
                'V': V,
                'z': z,
                'description': f'Array z: num_samples={num_samples}, feature_dim={feature_dim}, z=array',
            })

        if num_tests > 4:
            # Test 5: Large feature dimension
            num_samples = 3
            feature_dim = 20
            V = np.random.randn(num_samples, feature_dim)
            z = 1
            test_cases.append({
                'V': V,
                'z': z,
                'description': f'Large feature_dim: num_samples={num_samples}, feature_dim={feature_dim}, z={z}',
            })

        if num_tests > 5:
            # Test 6: Large batch
            num_samples = 10
            feature_dim = 8
            V = np.random.randn(num_samples, feature_dim)
            z = 1
            test_cases.append({
                'V': V,
                'z': z,
                'description': f'Large batch: num_samples={num_samples}, feature_dim={feature_dim}, z={z}',
            })

        if num_tests > 6:
            # Test 7: Edge case - all zeros input
            num_samples = 3
            feature_dim = 5
            V = np.zeros((num_samples, feature_dim))
            z = 1
            test_cases.append({
                'V': V,
                'z': z,
                'description': f'Edge case: zero input, num_samples={num_samples}, feature_dim={feature_dim}, z={z}',
            })

        if num_tests > 7:
            # Test 8: All positive input
            num_samples = 4
            feature_dim = 6
            V = np.abs(np.random.randn(num_samples, feature_dim))
            z = 1
            test_cases.append({
                'V': V,
                'z': z,
                'description': f'All positive: num_samples={num_samples}, feature_dim={feature_dim}, z={z}',
            })

        if num_tests > 8:
            # Test 9: Very large feature dimension
            num_samples = 2
            feature_dim = 50
            V = np.random.randn(num_samples, feature_dim)
            z = 1
            test_cases.append({
                'V': V,
                'z': z,
                'description': f'Very large feature_dim: num_samples={num_samples}, feature_dim={feature_dim}, z={z}',
            })

        if num_tests > 9:
            # Test 10: z as array with different values
            num_samples = 5
            feature_dim = 5
            V = np.random.randn(num_samples, feature_dim)
            z = np.random.uniform(0.5, 3.0, num_samples)
            test_cases.append({
                'V': V,
                'z': z,
                'description': f'Random z array: num_samples={num_samples}, feature_dim={feature_dim}, z=random array',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            num_samples = 2 + (i % 8)
            feature_dim = 3 + (i % 10)
            V = np.random.randn(num_samples, feature_dim)
            if i % 3 == 0:
                z = np.random.uniform(0.5, 3.0, num_samples)
            else:
                z = 1.0 + (i % 5) * 0.5

            test_cases.append({
                'V': V,
                'z': z,
                'description': f'Additional test {i+1}: num_samples={num_samples}, feature_dim={feature_dim}, z={z}',
            })

        return test_cases[:num_tests]
