"""
Test Data Generator for GaussianModel.re_simulation_get_visual_xyz_delta() function.
Generates various test cases with different configurations.
"""

from __future__ import annotations

import torch


class TestDataGenerator:
    """Generate test data for GaussianModel.re_simulation_get_visual_xyz_delta() function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Default constants
        H = 0.00625
        KNN_K = 32
        secs = 0.01

        # Test 1: Basic case with small number of particles
        N_xyz = 10
        V_visual = 8
        xyz = torch.randn(N_xyz, 3) * 0.01
        visual_xyz = torch.randn(V_visual, 3) * 0.01
        velocity = torch.randn(N_xyz, 3) * 0.1
        test_cases.append({
            'xyz': xyz,
            'visual_xyz': visual_xyz,
            'velocity': velocity,
            'H': H,
            'KNN_K': KNN_K,
            'secs': secs,
            'description': f'Basic: N_xyz={N_xyz}, V_visual={V_visual}',
        })

        if num_tests > 1:
            # Test 2: Single visual particle
            N_xyz = 5
            V_visual = 1
            xyz = torch.randn(N_xyz, 3) * 0.01
            visual_xyz = torch.randn(V_visual, 3) * 0.01
            velocity = torch.randn(N_xyz, 3) * 0.1
            test_cases.append({
                'xyz': xyz,
                'visual_xyz': visual_xyz,
                'velocity': velocity,
                'H': H,
                'KNN_K': KNN_K,
                'secs': secs,
                'description': f'Single visual: N_xyz={N_xyz}, V_visual={V_visual}',
            })

        if num_tests > 2:
            # Test 3: Larger number of particles
            N_xyz = 20
            V_visual = 15
            xyz = torch.randn(N_xyz, 3) * 0.01
            visual_xyz = torch.randn(V_visual, 3) * 0.01
            velocity = torch.randn(N_xyz, 3) * 0.1
            test_cases.append({
                'xyz': xyz,
                'visual_xyz': visual_xyz,
                'velocity': velocity,
                'H': H,
                'KNN_K': KNN_K,
                'secs': secs,
                'description': f'Larger: N_xyz={N_xyz}, V_visual={V_visual}',
            })

        if num_tests > 3:
            # Test 4: More visual particles than xyz
            N_xyz = 8
            V_visual = 12
            xyz = torch.randn(N_xyz, 3) * 0.01
            visual_xyz = torch.randn(V_visual, 3) * 0.01
            velocity = torch.randn(N_xyz, 3) * 0.1
            test_cases.append({
                'xyz': xyz,
                'visual_xyz': visual_xyz,
                'velocity': velocity,
                'H': H,
                'KNN_K': KNN_K,
                'secs': secs,
                'description': f'More visual: N_xyz={N_xyz}, V_visual={V_visual}',
            })

        if num_tests > 4:
            # Test 5: Particles close together
            N_xyz = 10
            V_visual = 8
            xyz = torch.randn(N_xyz, 3) * 0.002  # Very close
            visual_xyz = torch.randn(V_visual, 3) * 0.002
            velocity = torch.randn(N_xyz, 3) * 0.05
            test_cases.append({
                'xyz': xyz,
                'visual_xyz': visual_xyz,
                'velocity': velocity,
                'H': H,
                'KNN_K': KNN_K,
                'secs': secs,
                'description': f'Close particles: N_xyz={N_xyz}, V_visual={V_visual}',
            })

        if num_tests > 5:
            # Test 6: Particles far apart
            N_xyz = 10
            V_visual = 8
            xyz = torch.randn(N_xyz, 3) * 0.05  # Far apart
            visual_xyz = torch.randn(V_visual, 3) * 0.05
            velocity = torch.randn(N_xyz, 3) * 0.2
            test_cases.append({
                'xyz': xyz,
                'visual_xyz': visual_xyz,
                'velocity': velocity,
                'H': H,
                'KNN_K': KNN_K,
                'secs': secs,
                'description': f'Far particles: N_xyz={N_xyz}, V_visual={V_visual}',
            })

        if num_tests > 6:
            # Test 7: Edge case - zero velocity
            N_xyz = 8
            V_visual = 6
            xyz = torch.randn(N_xyz, 3) * 0.01
            visual_xyz = torch.randn(V_visual, 3) * 0.01
            velocity = torch.zeros(N_xyz, 3)
            test_cases.append({
                'xyz': xyz,
                'visual_xyz': visual_xyz,
                'velocity': velocity,
                'H': H,
                'KNN_K': KNN_K,
                'secs': secs,
                'description': f'Zero velocity: N_xyz={N_xyz}, V_visual={V_visual}',
            })

        if num_tests > 7:
            # Test 8: Different time step
            N_xyz = 10
            V_visual = 8
            xyz = torch.randn(N_xyz, 3) * 0.01
            visual_xyz = torch.randn(V_visual, 3) * 0.01
            velocity = torch.randn(N_xyz, 3) * 0.1
            test_cases.append({
                'xyz': xyz,
                'visual_xyz': visual_xyz,
                'velocity': velocity,
                'H': H,
                'KNN_K': KNN_K,
                'secs': 0.02,  # Different time step
                'description': f'Different secs: N_xyz={N_xyz}, V_visual={V_visual}, secs=0.02',
            })

        if num_tests > 8:
            # Test 9: Very large number of particles
            N_xyz = 50
            V_visual = 40
            xyz = torch.randn(N_xyz, 3) * 0.01
            visual_xyz = torch.randn(V_visual, 3) * 0.01
            velocity = torch.randn(N_xyz, 3) * 0.1
            test_cases.append({
                'xyz': xyz,
                'visual_xyz': visual_xyz,
                'velocity': velocity,
                'H': H,
                'KNN_K': KNN_K,
                'secs': secs,
                'description': f'Large scale: N_xyz={N_xyz}, V_visual={V_visual}',
            })

        if num_tests > 9:
            # Test 10: Different H value
            N_xyz = 10
            V_visual = 8
            xyz = torch.randn(N_xyz, 3) * 0.01
            visual_xyz = torch.randn(V_visual, 3) * 0.01
            velocity = torch.randn(N_xyz, 3) * 0.1
            test_cases.append({
                'xyz': xyz,
                'visual_xyz': visual_xyz,
                'velocity': velocity,
                'H': 0.01,  # Larger H
                'KNN_K': KNN_K,
                'secs': secs,
                'description': f'Larger H: N_xyz={N_xyz}, V_visual={V_visual}, H=0.01',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            N_xyz = 5 + (i % 15)
            V_visual = 3 + (i % 12)
            xyz = torch.randn(N_xyz, 3) * (0.005 + (i % 5) * 0.005)
            visual_xyz = torch.randn(V_visual, 3) * (0.005 + (i % 5) * 0.005)
            velocity = torch.randn(N_xyz, 3) * (0.05 + (i % 3) * 0.05)

            test_cases.append({
                'xyz': xyz,
                'visual_xyz': visual_xyz,
                'velocity': velocity,
                'H': H,
                'KNN_K': KNN_K,
                'secs': secs,
                'description': f'Additional test {i+1}: N_xyz={N_xyz}, V_visual={V_visual}',
            })

        return test_cases[:num_tests]
