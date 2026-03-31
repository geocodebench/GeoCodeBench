"""
Test data generator for compute_angular_velocity_from_coeffs() unit tests.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


class TestDataGenerator:
    """Generate test data for compute_angular_velocity_from_coeffs() function."""

    def __init__(self, seed=42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []
        rng = self.rng

        # Test 1: Basic case - small rotation vector
        phi = jnp.asarray(rng.normal(size=(3,)) * 0.1)
        phi_dot = jnp.asarray(rng.normal(size=(3,)) * 0.1)
        test_cases.append({
            'phi': phi,
            'phi_dot': phi_dot,
            'description': 'Basic: small rotation vector (3,)',
        })

        if num_tests > 1:
            # Test 2: Zero rotation vector (edge case)
            phi = jnp.zeros(3)
            phi_dot = jnp.asarray(rng.normal(size=(3,)) * 0.1)
            test_cases.append({
                'phi': phi,
                'phi_dot': phi_dot,
                'description': 'Edge case: zero rotation vector',
            })

        if num_tests > 2:
            # Test 3: Large rotation vector
            phi = jnp.asarray(rng.normal(size=(3,)) * 2.0)
            phi_dot = jnp.asarray(rng.normal(size=(3,)) * 0.5)
            test_cases.append({
                'phi': phi,
                'phi_dot': phi_dot,
                'description': 'Large rotation vector (3,)',
            })

        if num_tests > 3:
            # Test 4: Batch case - multiple rotation vectors
            phi = jnp.asarray(rng.normal(size=(5, 3)) * 0.5)
            phi_dot = jnp.asarray(rng.normal(size=(5, 3)) * 0.2)
            test_cases.append({
                'phi': phi,
                'phi_dot': phi_dot,
                'description': 'Batch case: shape (5, 3)',
            })

        if num_tests > 4:
            # Test 5: Very small rotation vector (near zero)
            phi = jnp.asarray(rng.normal(size=(3,)) * 1e-6)
            phi_dot = jnp.asarray(rng.normal(size=(3,)) * 0.1)
            test_cases.append({
                'phi': phi,
                'phi_dot': phi_dot,
                'description': 'Very small rotation vector (near zero)',
            })

        if num_tests > 5:
            # Test 6: Larger batch
            phi = jnp.asarray(rng.normal(size=(10, 3)) * 0.3)
            phi_dot = jnp.asarray(rng.normal(size=(10, 3)) * 0.15)
            test_cases.append({
                'phi': phi,
                'phi_dot': phi_dot,
                'description': 'Larger batch: shape (10, 3)',
            })

        if num_tests > 6:
            # Test 7: High-dimensional batch
            phi = jnp.asarray(rng.normal(size=(3, 4, 3)) * 0.4)
            phi_dot = jnp.asarray(rng.normal(size=(3, 4, 3)) * 0.2)
            test_cases.append({
                'phi': phi,
                'phi_dot': phi_dot,
                'description': 'High-dimensional batch: shape (3, 4, 3)',
            })

        if num_tests > 7:
            # Test 8: Rotation vector with large norm
            phi = jnp.asarray(rng.normal(size=(3,)))
            phi = phi / jnp.linalg.norm(phi) * 3.0  # Normalize to norm 3.0
            phi_dot = jnp.asarray(rng.normal(size=(3,)) * 0.3)
            test_cases.append({
                'phi': phi,
                'phi_dot': phi_dot,
                'description': 'Rotation vector with large norm (3.0)',
            })

        if num_tests > 8:
            # Test 9: Zero derivative
            phi = jnp.asarray(rng.normal(size=(3,)) * 0.5)
            phi_dot = jnp.zeros(3)
            test_cases.append({
                'phi': phi,
                'phi_dot': phi_dot,
                'description': 'Zero derivative (phi_dot = 0)',
            })

        if num_tests > 9:
            # Test 10: Unit rotation vector
            phi = jnp.asarray(rng.normal(size=(3,)))
            phi = phi / jnp.linalg.norm(phi)  # Normalize to unit vector
            phi_dot = jnp.asarray(rng.normal(size=(3,)) * 0.2)
            test_cases.append({
                'phi': phi,
                'phi_dot': phi_dot,
                'description': 'Unit rotation vector (norm = 1.0)',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            batch_size = 2 + (i % 5)
            phi = jnp.asarray(rng.normal(size=(batch_size, 3)) * (0.1 + (i % 3) * 0.2))
            phi_dot = jnp.asarray(rng.normal(size=(batch_size, 3)) * (0.05 + (i % 2) * 0.1))

            test_cases.append({
                'phi': phi,
                'phi_dot': phi_dot,
                'description': f'Additional test {i+1}: shape {phi.shape}',
            })

        return test_cases[:num_tests]
