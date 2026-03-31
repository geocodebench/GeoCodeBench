"""
Test data generator for ddexp_so3() unit tests.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


class TestDataGenerator:
    """Generate test data for ddexp_so3() function."""

    def __init__(self, seed=42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []
        rng = self.rng

        # Test 1: Basic case with single rotation vector
        x = jnp.asarray(rng.normal(size=(3,)) * 0.5)
        z = jnp.asarray(rng.normal(size=(3,)) * 0.3)
        eps = 1e-8
        test_cases.append({
            'x': x,
            'z': z,
            'eps': eps,
            'description': f'Basic: single rotation vector, normal angles',
        })

        if num_tests > 1:
            # Test 2: Small angle case (near zero)
            x = jnp.asarray(rng.normal(size=(3,)) * 1e-6)
            z = jnp.asarray(rng.normal(size=(3,)) * 0.1)
            eps = 1e-8
            test_cases.append({
                'x': x,
                'z': z,
                'eps': eps,
                'description': f'Small angle: near-zero rotation vector',
            })

        if num_tests > 2:
            # Test 3: Batch of rotation vectors
            x = jnp.asarray(rng.normal(size=(5, 3)) * 0.8)
            z = jnp.asarray(rng.normal(size=(5, 3)) * 0.4)
            eps = 1e-8
            test_cases.append({
                'x': x,
                'z': z,
                'eps': eps,
                'description': f'Batch: 5 rotation vectors',
            })

        if num_tests > 3:
            # Test 4: Large angle case
            x = jnp.asarray(rng.normal(size=(3,)) * 2.0)
            z = jnp.asarray(rng.normal(size=(3,)) * 0.5)
            eps = 1e-8
            test_cases.append({
                'x': x,
                'z': z,
                'eps': eps,
                'description': f'Large angle: large rotation vector',
            })

        if num_tests > 4:
            # Test 5: Orthogonal vectors
            x = jnp.array([1.0, 0.0, 0.0])
            z = jnp.array([0.0, 1.0, 0.0])
            eps = 1e-8
            test_cases.append({
                'x': x,
                'z': z,
                'eps': eps,
                'description': f'Orthogonal: x and z are orthogonal',
            })

        if num_tests > 5:
            # Test 6: Parallel vectors
            x = jnp.array([1.0, 0.0, 0.0]) * 0.5
            z = jnp.array([1.0, 0.0, 0.0]) * 0.3
            eps = 1e-8
            test_cases.append({
                'x': x,
                'z': z,
                'eps': eps,
                'description': f'Parallel: x and z are parallel',
            })

        if num_tests > 6:
            # Test 7: Multi-dimensional batch
            x = jnp.asarray(rng.normal(size=(3, 4, 3)) * 0.6)
            z = jnp.asarray(rng.normal(size=(3, 4, 3)) * 0.3)
            eps = 1e-8
            test_cases.append({
                'x': x,
                'z': z,
                'eps': eps,
                'description': f'Multi-dim batch: (3, 4, 3) shape',
            })

        if num_tests > 7:
            # Test 8: Zero vector case
            x = jnp.array([0.0, 0.0, 0.0])
            z = jnp.asarray(rng.normal(size=(3,)) * 0.2)
            eps = 1e-8
            test_cases.append({
                'x': x,
                'z': z,
                'eps': eps,
                'description': f'Zero vector: x is zero',
            })

        if num_tests > 8:
            # Test 9: Very small eps threshold
            x = jnp.asarray(rng.normal(size=(3,)) * 1e-7)
            z = jnp.asarray(rng.normal(size=(3,)) * 0.1)
            eps = 1e-10
            test_cases.append({
                'x': x,
                'z': z,
                'eps': eps,
                'description': f'Very small eps: eps=1e-10',
            })

        if num_tests > 9:
            # Test 10: Large batch
            x = jnp.asarray(rng.normal(size=(10, 3)) * 0.7)
            z = jnp.asarray(rng.normal(size=(10, 3)) * 0.35)
            eps = 1e-8
            test_cases.append({
                'x': x,
                'z': z,
                'eps': eps,
                'description': f'Large batch: 10 rotation vectors',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            batch_size = 2 + (i % 8)
            x = jnp.asarray(rng.normal(size=(batch_size, 3)) * (0.3 + (i % 5) * 0.2))
            z = jnp.asarray(rng.normal(size=(batch_size, 3)) * (0.2 + (i % 3) * 0.1))
            eps = 1e-8

            test_cases.append({
                'x': x,
                'z': z,
                'eps': eps,
                'description': f'Additional test {i+1}: batch_size={batch_size}',
            })

        return test_cases[:num_tests]
