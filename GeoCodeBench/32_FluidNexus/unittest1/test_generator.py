"""
Test Data Generator for GaussianModel.project_gas_constraints() function.
Generates various test cases with different configurations.
"""

from __future__ import annotations

import torch


class TestDataGenerator:
    """Generate test data for GaussianModel.project_gas_constraints() function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []
        device = "cpu"  # Use CPU for testing

        # Test 1: Basic case with small number of particles
        N = 10
        estimate_xyz = torch.randn(N, 3, device=device) * 0.01
        xyz = estimate_xyz.clone()
        velocity = torch.randn(N, 3, device=device) * 0.001
        force = torch.zeros(N, 3, device=device)
        imass = torch.ones(N, 1, device=device)
        counts = torch.zeros(N, 1, device=device)

        test_cases.append({
            'estimate_xyz': estimate_xyz,
            'xyz': xyz,
            'velocity': velocity,
            'force': force,
            'imass': imass,
            'counts': counts,
            'H': 0.00625,
            'p0': 1.0,
            'k': 1.0,
            'KNN_K': 32,
            'description': f'Basic: N={N} particles',
        })

        if num_tests > 1:
            # Test 2: Medium number of particles
            N = 20
            estimate_xyz = torch.randn(N, 3, device=device) * 0.01
            xyz = estimate_xyz.clone()
            velocity = torch.randn(N, 3, device=device) * 0.001
            force = torch.zeros(N, 3, device=device)
            imass = torch.ones(N, 1, device=device)
            counts = torch.zeros(N, 1, device=device)

            test_cases.append({
                'estimate_xyz': estimate_xyz,
                'xyz': xyz,
                'velocity': velocity,
                'force': force,
                'imass': imass,
                'counts': counts,
                'H': 0.00625,
                'p0': 1.0,
                'k': 1.0,
                'KNN_K': 32,
                'description': f'Medium: N={N} particles',
            })

        if num_tests > 2:
            # Test 3: Larger number of particles
            N = 30
            estimate_xyz = torch.randn(N, 3, device=device) * 0.01
            xyz = estimate_xyz.clone()
            velocity = torch.randn(N, 3, device=device) * 0.001
            force = torch.zeros(N, 3, device=device)
            imass = torch.ones(N, 1, device=device)
            counts = torch.zeros(N, 1, device=device)

            test_cases.append({
                'estimate_xyz': estimate_xyz,
                'xyz': xyz,
                'velocity': velocity,
                'force': force,
                'imass': imass,
                'counts': counts,
                'H': 0.00625,
                'p0': 1.0,
                'k': 1.0,
                'KNN_K': 32,
                'description': f'Larger: N={N} particles',
            })

        if num_tests > 3:
            # Test 4: Different H value
            N = 15
            estimate_xyz = torch.randn(N, 3, device=device) * 0.01
            xyz = estimate_xyz.clone()
            velocity = torch.randn(N, 3, device=device) * 0.001
            force = torch.zeros(N, 3, device=device)
            imass = torch.ones(N, 1, device=device)
            counts = torch.zeros(N, 1, device=device)

            test_cases.append({
                'estimate_xyz': estimate_xyz,
                'xyz': xyz,
                'velocity': velocity,
                'force': force,
                'imass': imass,
                'counts': counts,
                'H': 0.01,
                'p0': 1.0,
                'k': 1.0,
                'KNN_K': 32,
                'description': f'Different H: N={N}, H=0.01',
            })

        if num_tests > 4:
            # Test 5: Different p0 and k values
            N = 12
            estimate_xyz = torch.randn(N, 3, device=device) * 0.01
            xyz = estimate_xyz.clone()
            velocity = torch.randn(N, 3, device=device) * 0.001
            force = torch.zeros(N, 3, device=device)
            imass = torch.ones(N, 1, device=device)
            counts = torch.zeros(N, 1, device=device)

            test_cases.append({
                'estimate_xyz': estimate_xyz,
                'xyz': xyz,
                'velocity': velocity,
                'force': force,
                'imass': imass,
                'counts': counts,
                'H': 0.00625,
                'p0': 2.0,
                'k': 0.5,
                'KNN_K': 32,
                'description': f'Different params: N={N}, p0=2.0, k=0.5',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            N = 8 + (i % 15)
            estimate_xyz = torch.randn(N, 3, device=device) * 0.01
            xyz = estimate_xyz.clone()
            velocity = torch.randn(N, 3, device=device) * 0.001
            force = torch.zeros(N, 3, device=device)
            imass = torch.ones(N, 1, device=device)
            counts = torch.zeros(N, 1, device=device)

            test_cases.append({
                'estimate_xyz': estimate_xyz,
                'xyz': xyz,
                'velocity': velocity,
                'force': force,
                'imass': imass,
                'counts': counts,
                'H': 0.00625,
                'p0': 1.0,
                'k': 1.0,
                'KNN_K': 32,
                'description': f'Additional test {i+1}: N={N}',
            })

        return test_cases[:num_tests]
