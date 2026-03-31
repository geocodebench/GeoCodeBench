"""
Test Data Generator for update_pose function.
Generates test cases with different configurations.
"""

import numpy as np
import torch

from reference_implementation import MockCamera


class TestDataGenerator:
    """Generate test data for update_pose function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small deltas
        R = torch.eye(3, dtype=torch.float32)
        T = torch.zeros(3, dtype=torch.float32)
        cam_trans_delta = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float32, requires_grad=True)
        cam_rot_delta = torch.tensor([0.001, 0.002, 0.003], dtype=torch.float32, requires_grad=True)
        camera = MockCamera(R, T, cam_trans_delta, cam_rot_delta)
        test_cases.append({
            'camera': camera,
            'converged_threshold': 1e-4,
            'description': 'Basic: small translation and rotation deltas',
        })

        if num_tests > 1:
            # Test 2: Larger deltas
            R = torch.eye(3, dtype=torch.float32)
            T = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
            cam_trans_delta = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, requires_grad=True)
            cam_rot_delta = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, requires_grad=True)
            camera = MockCamera(R, T, cam_trans_delta, cam_rot_delta)
            test_cases.append({
                'camera': camera,
                'converged_threshold': 1e-3,
                'description': 'Larger deltas: moderate translation and rotation',
            })

        if num_tests > 2:
            # Test 3: Random initial pose
            R = self._generate_random_rotation()
            T = torch.randn(3, dtype=torch.float32) * 2.0
            cam_trans_delta = torch.randn(3, dtype=torch.float32, requires_grad=True) * 0.1
            cam_rot_delta = torch.randn(3, dtype=torch.float32, requires_grad=True) * 0.1
            camera = MockCamera(R, T, cam_trans_delta, cam_rot_delta)
            test_cases.append({
                'camera': camera,
                'converged_threshold': 1e-4,
                'description': 'Random pose: random initial R, T with random deltas',
            })

        if num_tests > 3:
            # Test 4: Very small deltas (should converge)
            R = torch.eye(3, dtype=torch.float32)
            T = torch.zeros(3, dtype=torch.float32)
            cam_trans_delta = torch.tensor([1e-5, 2e-5, 3e-5], dtype=torch.float32, requires_grad=True)
            cam_rot_delta = torch.tensor([1e-6, 2e-6, 3e-6], dtype=torch.float32, requires_grad=True)
            camera = MockCamera(R, T, cam_trans_delta, cam_rot_delta)
            test_cases.append({
                'camera': camera,
                'converged_threshold': 1e-4,
                'description': 'Very small deltas: should converge',
            })

        if num_tests > 4:
            # Test 5: Large rotation
            R = self._generate_random_rotation()
            T = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32)
            cam_trans_delta = torch.tensor([0.05, 0.1, 0.15], dtype=torch.float32, requires_grad=True)
            cam_rot_delta = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32, requires_grad=True)
            camera = MockCamera(R, T, cam_trans_delta, cam_rot_delta)
            test_cases.append({
                'camera': camera,
                'converged_threshold': 1e-3,
                'description': 'Large rotation: significant rotation delta',
            })

        if num_tests > 5:
            # Test 6: Zero deltas (should converge immediately)
            R = torch.eye(3, dtype=torch.float32)
            T = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            cam_trans_delta = torch.zeros(3, dtype=torch.float32, requires_grad=True)
            cam_rot_delta = torch.zeros(3, dtype=torch.float32, requires_grad=True)
            camera = MockCamera(R, T, cam_trans_delta, cam_rot_delta)
            test_cases.append({
                'camera': camera,
                'converged_threshold': 1e-4,
                'description': 'Zero deltas: should converge immediately',
            })

        if num_tests > 6:
            # Test 7: Different threshold
            R = torch.eye(3, dtype=torch.float32)
            T = torch.zeros(3, dtype=torch.float32)
            cam_trans_delta = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32, requires_grad=True)
            cam_rot_delta = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32, requires_grad=True)
            camera = MockCamera(R, T, cam_trans_delta, cam_rot_delta)
            test_cases.append({
                'camera': camera,
                'converged_threshold': 1e-2,
                'description': 'Different threshold: more lenient convergence',
            })

        if num_tests > 7:
            # Test 8: Complex initial pose
            R = self._generate_random_rotation()
            T = torch.tensor([-1.0, 2.0, -0.5], dtype=torch.float32)
            cam_trans_delta = torch.tensor([0.02, -0.01, 0.03], dtype=torch.float32, requires_grad=True)
            cam_rot_delta = torch.tensor([-0.01, 0.02, -0.01], dtype=torch.float32, requires_grad=True)
            camera = MockCamera(R, T, cam_trans_delta, cam_rot_delta)
            test_cases.append({
                'camera': camera,
                'converged_threshold': 1e-4,
                'description': 'Complex pose: random R, negative T components',
            })

        if num_tests > 8:
            # Test 9: Edge case - very large deltas
            R = torch.eye(3, dtype=torch.float32)
            T = torch.zeros(3, dtype=torch.float32)
            cam_trans_delta = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)
            cam_rot_delta = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, requires_grad=True)
            camera = MockCamera(R, T, cam_trans_delta, cam_rot_delta)
            test_cases.append({
                'camera': camera,
                'converged_threshold': 1e-1,
                'description': 'Large deltas: significant pose changes',
            })

        if num_tests > 9:
            # Test 10: Mixed positive/negative deltas
            R = torch.eye(3, dtype=torch.float32)
            T = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
            cam_trans_delta = torch.tensor([0.1, -0.05, 0.2], dtype=torch.float32, requires_grad=True)
            cam_rot_delta = torch.tensor([-0.1, 0.15, -0.05], dtype=torch.float32, requires_grad=True)
            camera = MockCamera(R, T, cam_trans_delta, cam_rot_delta)
            test_cases.append({
                'camera': camera,
                'converged_threshold': 1e-3,
                'description': 'Mixed deltas: positive and negative components',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            R = self._generate_random_rotation() if i % 2 == 0 else torch.eye(3, dtype=torch.float32)
            T = torch.randn(3, dtype=torch.float32) * (0.5 + i * 0.1)
            cam_trans_delta = torch.randn(3, dtype=torch.float32, requires_grad=True) * (0.01 + i * 0.01)
            cam_rot_delta = torch.randn(3, dtype=torch.float32, requires_grad=True) * (0.01 + i * 0.01)
            camera = MockCamera(R, T, cam_trans_delta, cam_rot_delta)

            test_cases.append({
                'camera': camera,
                'converged_threshold': 1e-4,
                'description': f'Additional test {i+1}: random configuration',
            })

        return test_cases[:num_tests]

    def _generate_random_rotation(self):
        """Generate a random 3x3 rotation matrix."""
        # Generate random axis-angle representation
        axis = torch.randn(3, dtype=torch.float32)
        axis = axis / torch.norm(axis)
        angle = torch.rand(1, dtype=torch.float32) * np.pi / 4  # Small angle for numerical stability

        # Rodrigues' rotation formula
        K = torch.tensor([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]], dtype=torch.float32)

        R = torch.eye(3, dtype=torch.float32) + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.mm(K, K)
        return R
