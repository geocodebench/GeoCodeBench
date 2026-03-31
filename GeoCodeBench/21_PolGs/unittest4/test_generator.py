"""
Test Data Generator for camera ray functions.
"""

import numpy as np
import torch


class MockEnvMap:
    """Mock environment map for testing."""

    def __init__(self, seed=42):
        torch.manual_seed(seed)
        # Simple MLP to simulate environment map
        self.linear1 = torch.nn.Linear(3, 16)
        self.linear2 = torch.nn.Linear(16, 3)

    def __call__(self, rays_d):
        """
        Args:
            rays_d: Ray directions, shape (N, 3)
        Returns:
            colors: RGB colors, shape (N, 3)
        """
        x = torch.relu(self.linear1(rays_d))
        x = self.linear2(x)
        return x


class TestDataGenerator:
    """Generate test data for camera ray functions."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small resolution
        H, W = 32, 48
        K = np.array([[320.0, 0.0, 24.0],
                      [0.0, 320.0, 16.0],
                      [0.0, 0.0, 1.0]])
        R = torch.tensor([[1.0, 0.0, 0.0],
                         [0.0, 0.9, -0.436],
                         [0.0, 0.436, 0.9]], dtype=torch.float32)
        T = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float32)
        normal_map = torch.randn(H, W, 3)
        normal_map = normal_map / torch.norm(normal_map, dim=-1, keepdim=True)
        env_map = MockEnvMap(seed=42)

        test_cases.append({
            'HWK': (H, W, K),
            'R': R,
            'T': T,
            'normal_map': normal_map,
            'env_map': env_map,
            'rayd': torch.randn(H, W, 3),
            'normal': torch.randn(H, W, 3),
            'rays_d_cubemap': torch.randn(H, W, 3),
            'description': f'Basic: H={H}, W={W}, simple camera pose',
        })

        if num_tests > 1:
            # Test 2: Different resolution
            H, W = 64, 64
            K = np.array([[400.0, 0.0, 32.0],
                          [0.0, 400.0, 32.0],
                          [0.0, 0.0, 1.0]])
            R = torch.tensor([[0.866, 0.0, 0.5],
                             [0.0, 1.0, 0.0],
                             [-0.5, 0.0, 0.866]], dtype=torch.float32)
            T = torch.tensor([1.0, 0.5, 1.5], dtype=torch.float32)
            normal_map = torch.randn(H, W, 3)
            normal_map = normal_map / torch.norm(normal_map, dim=-1, keepdim=True)
            env_map = MockEnvMap(seed=43)

            test_cases.append({
                'HWK': (H, W, K),
                'R': R,
                'T': T,
                'normal_map': normal_map,
                'env_map': env_map,
                'rayd': torch.randn(H, W, 3),
                'normal': torch.randn(H, W, 3),
                'rays_d_cubemap': torch.randn(H, W, 3),
                'description': f'Different resolution: H={H}, W={W}',
            })

        if num_tests > 2:
            # Test 3: Wide angle
            H, W = 48, 96
            K = np.array([[200.0, 0.0, 48.0],
                          [0.0, 200.0, 24.0],
                          [0.0, 0.0, 1.0]])
            R = torch.eye(3, dtype=torch.float32)
            T = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
            normal_map = torch.randn(H, W, 3)
            normal_map = normal_map / torch.norm(normal_map, dim=-1, keepdim=True)
            env_map = MockEnvMap(seed=44)

            test_cases.append({
                'HWK': (H, W, K),
                'R': R,
                'T': T,
                'normal_map': normal_map,
                'env_map': env_map,
                'rayd': torch.randn(H, W, 3),
                'normal': torch.randn(H, W, 3),
                'rays_d_cubemap': torch.randn(H, W, 3),
                'description': f'Wide angle: H={H}, W={W}, wide FOV',
            })

        if num_tests > 3:
            # Test 4: Rotated camera
            H, W = 40, 60
            K = np.array([[350.0, 0.0, 30.0],
                          [0.0, 350.0, 20.0],
                          [0.0, 0.0, 1.0]])
            angle = np.pi / 4
            R = torch.tensor([[np.cos(angle), 0, np.sin(angle)],
                             [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]], dtype=torch.float32)
            T = torch.tensor([2.0, 1.0, 3.0], dtype=torch.float32)
            normal_map = torch.randn(H, W, 3)
            normal_map = normal_map / torch.norm(normal_map, dim=-1, keepdim=True)
            env_map = MockEnvMap(seed=45)

            test_cases.append({
                'HWK': (H, W, K),
                'R': R,
                'T': T,
                'normal_map': normal_map,
                'env_map': env_map,
                'rayd': torch.randn(H, W, 3),
                'normal': torch.randn(H, W, 3),
                'rays_d_cubemap': torch.randn(H, W, 3),
                'description': f'Rotated camera: 45 degree rotation',
            })

        if num_tests > 4:
            # Test 5: Larger resolution
            H, W = 80, 120
            K = np.array([[500.0, 0.0, 60.0],
                          [0.0, 500.0, 40.0],
                          [0.0, 0.0, 1.0]])
            R = torch.tensor([[0.707, -0.707, 0.0],
                             [0.707, 0.707, 0.0],
                             [0.0, 0.0, 1.0]], dtype=torch.float32)
            T = torch.tensor([-1.0, -1.0, 5.0], dtype=torch.float32)
            normal_map = torch.randn(H, W, 3)
            normal_map = normal_map / torch.norm(normal_map, dim=-1, keepdim=True)
            env_map = MockEnvMap(seed=46)

            test_cases.append({
                'HWK': (H, W, K),
                'R': R,
                'T': T,
                'normal_map': normal_map,
                'env_map': env_map,
                'rayd': torch.randn(H, W, 3),
                'normal': torch.randn(H, W, 3),
                'rays_d_cubemap': torch.randn(H, W, 3),
                'description': f'Larger resolution: H={H}, W={W}',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            H = 32 + i * 8
            W = 48 + i * 12
            K = np.array([[300.0 + i * 50, 0.0, W / 2.0],
                          [0.0, 300.0 + i * 50, H / 2.0],
                          [0.0, 0.0, 1.0]])
            angle = (i + 1) * np.pi / 8
            R = torch.tensor([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0],
                             [0, 0, 1]], dtype=torch.float32)
            T = torch.tensor([float(i), float(i) * 0.5, float(i) + 2.0], dtype=torch.float32)
            normal_map = torch.randn(H, W, 3)
            normal_map = normal_map / torch.norm(normal_map, dim=-1, keepdim=True)
            env_map = MockEnvMap(seed=47 + i)

            test_cases.append({
                'HWK': (H, W, K),
                'R': R,
                'T': T,
                'normal_map': normal_map,
                'env_map': env_map,
                'rayd': torch.randn(H, W, 3),
                'normal': torch.randn(H, W, 3),
                'rays_d_cubemap': torch.randn(H, W, 3),
                'description': f'Additional test {i+1}: H={H}, W={W}',
            })

        return test_cases[:num_tests]
