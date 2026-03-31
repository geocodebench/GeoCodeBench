"""
Test Data Generator for shift_direct() function.
Generates various test cases with different configurations.
"""

import numpy as np


class TestDataGenerator:
    """Generate test data for shift_direct() function."""

    def __init__(self, seed=42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small batch
        n_rays = 4
        n_samples = 8
        n_bins = 100
        num_rgb_channels = 3
        dists = self.rng.rand(n_rays, n_samples) * (n_bins - 1)
        direct_rgbs = self.rng.rand(n_rays, n_samples, num_rgb_channels)
        weights = self.rng.rand(n_rays, n_samples)
        weights = weights / weights.sum(axis=1, keepdims=True)  # Normalize
        impulse_response = None
        test_cases.append({
            'dists': dists,
            'direct_rgbs': direct_rgbs,
            'weights': weights,
            'n_bins': n_bins,
            'num_rgb_channels': num_rgb_channels,
            'impulse_response': impulse_response,
            'description': f'Basic: n_rays={n_rays}, n_samples={n_samples}, n_bins={n_bins}',
        })

        if num_tests > 1:
            # Test 2: Single ray, more samples
            n_rays = 1
            n_samples = 16
            n_bins = 200
            num_rgb_channels = 3
            dists = self.rng.rand(n_rays, n_samples) * (n_bins - 1)
            direct_rgbs = self.rng.rand(n_rays, n_samples, num_rgb_channels)
            weights = self.rng.rand(n_rays, n_samples)
            weights = weights / weights.sum(axis=1, keepdims=True)
            impulse_response = None
            test_cases.append({
                'dists': dists,
                'direct_rgbs': direct_rgbs,
                'weights': weights,
                'n_bins': n_bins,
                'num_rgb_channels': num_rgb_channels,
                'impulse_response': impulse_response,
                'description': f'Single ray: n_rays={n_rays}, n_samples={n_samples}, n_bins={n_bins}',
            })

        if num_tests > 2:
            # Test 3: Larger batch, more bins
            n_rays = 8
            n_samples = 12
            n_bins = 500
            num_rgb_channels = 3
            dists = self.rng.rand(n_rays, n_samples) * (n_bins - 1)
            direct_rgbs = self.rng.rand(n_rays, n_samples, num_rgb_channels)
            weights = self.rng.rand(n_rays, n_samples)
            weights = weights / weights.sum(axis=1, keepdims=True)
            impulse_response = None
            test_cases.append({
                'dists': dists,
                'direct_rgbs': direct_rgbs,
                'weights': weights,
                'n_bins': n_bins,
                'num_rgb_channels': num_rgb_channels,
                'impulse_response': impulse_response,
                'description': f'Larger batch: n_rays={n_rays}, n_samples={n_samples}, n_bins={n_bins}',
            })

        if num_tests > 3:
            # Test 4: Edge case - dists at boundaries
            n_rays = 3
            n_samples = 6
            n_bins = 100
            num_rgb_channels = 3
            dists = self.rng.rand(n_rays, n_samples) * (n_bins - 1)
            # Set some values to exact integers
            dists[0, 0] = 0.0
            dists[0, 1] = float(n_bins - 1)
            dists[1, 0] = 50.0
            direct_rgbs = self.rng.rand(n_rays, n_samples, num_rgb_channels)
            weights = self.rng.rand(n_rays, n_samples)
            weights = weights / weights.sum(axis=1, keepdims=True)
            impulse_response = None
            test_cases.append({
                'dists': dists,
                'direct_rgbs': direct_rgbs,
                'weights': weights,
                'n_bins': n_bins,
                'num_rgb_channels': num_rgb_channels,
                'impulse_response': impulse_response,
                'description': f'Edge case: boundary dists, n_rays={n_rays}, n_bins={n_bins}',
            })

        if num_tests > 4:
            # Test 5: Very large n_bins (like real use case)
            n_rays = 5
            n_samples = 10
            n_bins = 700
            num_rgb_channels = 3
            dists = self.rng.rand(n_rays, n_samples) * (n_bins - 1)
            direct_rgbs = self.rng.rand(n_rays, n_samples, num_rgb_channels)
            weights = self.rng.rand(n_rays, n_samples)
            weights = weights / weights.sum(axis=1, keepdims=True)
            impulse_response = None
            test_cases.append({
                'dists': dists,
                'direct_rgbs': direct_rgbs,
                'weights': weights,
                'n_bins': n_bins,
                'num_rgb_channels': num_rgb_channels,
                'impulse_response': impulse_response,
                'description': f'Large n_bins: n_rays={n_rays}, n_samples={n_samples}, n_bins={n_bins}',
            })

        if num_tests > 5:
            # Test 6: Small n_bins
            n_rays = 6
            n_samples = 5
            n_bins = 20
            num_rgb_channels = 3
            dists = self.rng.rand(n_rays, n_samples) * (n_bins - 1)
            direct_rgbs = self.rng.rand(n_rays, n_samples, num_rgb_channels)
            weights = self.rng.rand(n_rays, n_samples)
            weights = weights / weights.sum(axis=1, keepdims=True)
            impulse_response = None
            test_cases.append({
                'dists': dists,
                'direct_rgbs': direct_rgbs,
                'weights': weights,
                'n_bins': n_bins,
                'num_rgb_channels': num_rgb_channels,
                'impulse_response': impulse_response,
                'description': f'Small n_bins: n_rays={n_rays}, n_samples={n_samples}, n_bins={n_bins}',
            })

        if num_tests > 6:
            # Test 7: Edge case - all zeros weights
            n_rays = 3
            n_samples = 4
            n_bins = 50
            num_rgb_channels = 3
            dists = self.rng.rand(n_rays, n_samples) * (n_bins - 1)
            direct_rgbs = self.rng.rand(n_rays, n_samples, num_rgb_channels)
            weights = np.zeros((n_rays, n_samples))
            weights[:, 0] = 1.0  # Only first sample has weight
            impulse_response = None
            test_cases.append({
                'dists': dists,
                'direct_rgbs': direct_rgbs,
                'weights': weights,
                'n_bins': n_bins,
                'num_rgb_channels': num_rgb_channels,
                'impulse_response': impulse_response,
                'description': f'Sparse weights: n_rays={n_rays}, n_samples={n_samples}, n_bins={n_bins}',
            })

        if num_tests > 7:
            # Test 8: Many samples
            n_rays = 4
            n_samples = 32
            n_bins = 150
            num_rgb_channels = 3
            dists = self.rng.rand(n_rays, n_samples) * (n_bins - 1)
            direct_rgbs = self.rng.rand(n_rays, n_samples, num_rgb_channels)
            weights = self.rng.rand(n_rays, n_samples)
            weights = weights / weights.sum(axis=1, keepdims=True)
            impulse_response = None
            test_cases.append({
                'dists': dists,
                'direct_rgbs': direct_rgbs,
                'weights': weights,
                'n_bins': n_bins,
                'num_rgb_channels': num_rgb_channels,
                'impulse_response': impulse_response,
                'description': f'Many samples: n_rays={n_rays}, n_samples={n_samples}, n_bins={n_bins}',
            })

        if num_tests > 8:
            # Test 9: Very large batch
            n_rays = 32
            n_samples = 8
            n_bins = 200
            num_rgb_channels = 3
            dists = self.rng.rand(n_rays, n_samples) * (n_bins - 1)
            direct_rgbs = self.rng.rand(n_rays, n_samples, num_rgb_channels)
            weights = self.rng.rand(n_rays, n_samples)
            weights = weights / weights.sum(axis=1, keepdims=True)
            impulse_response = None
            test_cases.append({
                'dists': dists,
                'direct_rgbs': direct_rgbs,
                'weights': weights,
                'n_bins': n_bins,
                'num_rgb_channels': num_rgb_channels,
                'impulse_response': impulse_response,
                'description': f'Very large batch: n_rays={n_rays}, n_samples={n_samples}, n_bins={n_bins}',
            })

        if num_tests > 9:
            # Test 10: Normalized RGB values
            n_rays = 7
            n_samples = 10
            n_bins = 100
            num_rgb_channels = 3
            dists = self.rng.rand(n_rays, n_samples) * (n_bins - 1)
            direct_rgbs = self.rng.rand(n_rays, n_samples, num_rgb_channels)
            direct_rgbs = direct_rgbs / (direct_rgbs.max() + 1e-8)  # Normalize to [0, 1]
            weights = self.rng.rand(n_rays, n_samples)
            weights = weights / weights.sum(axis=1, keepdims=True)
            impulse_response = None
            test_cases.append({
                'dists': dists,
                'direct_rgbs': direct_rgbs,
                'weights': weights,
                'n_bins': n_bins,
                'num_rgb_channels': num_rgb_channels,
                'impulse_response': impulse_response,
                'description': f'Normalized RGB: n_rays={n_rays}, n_samples={n_samples}, n_bins={n_bins}',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            n_rays = 3 + (i % 10)
            n_samples = 5 + (i % 15)
            n_bins = 50 + (i % 20) * 30
            num_rgb_channels = 3
            dists = self.rng.rand(n_rays, n_samples) * (n_bins - 1)
            direct_rgbs = self.rng.rand(n_rays, n_samples, num_rgb_channels)
            weights = self.rng.rand(n_rays, n_samples)
            weights = weights / weights.sum(axis=1, keepdims=True)
            impulse_response = None

            test_cases.append({
                'dists': dists,
                'direct_rgbs': direct_rgbs,
                'weights': weights,
                'n_bins': n_bins,
                'num_rgb_channels': num_rgb_channels,
                'impulse_response': impulse_response,
                'description': f'Additional test {i+1}: n_rays={n_rays}, n_samples={n_samples}, n_bins={n_bins}',
            })

        return test_cases[:num_tests]
