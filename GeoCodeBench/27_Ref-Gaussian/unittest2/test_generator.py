"""
Test Data Generator for get_full_color_volume_indirect function.
Generates test cases with different configurations.
"""

import torch
import numpy as np

from reference_implementation import MockEnvmap, MockPC, init_fg_lut


class TestDataGenerator:
    """Generate test data for get_full_color_volume_indirect function."""

    def __init__(self, seed=42, device='cpu'):
        self.seed = seed
        self.device = device
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize FG_LUT
        init_fg_lut(device=device)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small number of points
        N = 100  # number of points
        H, W = 64, 64
        test_cases.append(self._create_test_case(N, H, W, use_ray_tracer=True,
                                                  description=f"Basic: N={N}, H={H}, W={W}, with ray tracer"))

        if num_tests > 1:
            # Test 2: Without ray tracer
            N = 150
            H, W = 64, 64
            test_cases.append(self._create_test_case(N, H, W, use_ray_tracer=False,
                                                      description=f"No ray tracer: N={N}, H={H}, W={W}"))

        if num_tests > 2:
            # Test 3: Larger image resolution
            N = 200
            H, W = 128, 128
            test_cases.append(self._create_test_case(N, H, W, use_ray_tracer=True,
                                                      description=f"Larger resolution: N={N}, H={H}, W={W}"))

        if num_tests > 3:
            # Test 4: More points
            N = 500
            H, W = 64, 64
            test_cases.append(self._create_test_case(N, H, W, use_ray_tracer=True,
                                                      description=f"More points: N={N}"))

        if num_tests > 4:
            # Test 5: Different alpha values (some zeros)
            N = 100
            H, W = 64, 64
            test_case = self._create_test_case(N, H, W, use_ray_tracer=True,
                                                description=f"Mixed alpha: N={N}, some alpha=0")
            # Set some alpha to zero
            mask = torch.rand(N) > 0.7
            test_case['render_alpha'][mask] = 0.0
            test_cases.append(test_case)

        if num_tests > 5:
            # Test 6: High roughness
            N = 150
            H, W = 64, 64
            test_case = self._create_test_case(N, H, W, use_ray_tracer=False,
                                                description=f"High roughness: N={N}")
            test_case['roughness'] = torch.rand(N, 1, device=self.device) * 0.5 + 0.5  # [0.5, 1.0]
            test_cases.append(test_case)

        if num_tests > 6:
            # Test 7: Low roughness
            N = 150
            H, W = 64, 64
            test_case = self._create_test_case(N, H, W, use_ray_tracer=True,
                                                description=f"Low roughness: N={N}")
            test_case['roughness'] = torch.rand(N, 1, device=self.device) * 0.3  # [0.0, 0.3]
            test_cases.append(test_case)

        if num_tests > 7:
            # Test 8: High reflection strength
            N = 200
            H, W = 64, 64
            test_case = self._create_test_case(N, H, W, use_ray_tracer=False,
                                                description=f"High refl_strength: N={N}")
            test_case['refl_strength'] = torch.rand(N, 1, device=self.device) * 0.5 + 0.5  # [0.5, 1.0]
            test_cases.append(test_case)

        if num_tests > 8:
            # Test 9: Small image
            N = 50
            H, W = 32, 32
            test_cases.append(self._create_test_case(N, H, W, use_ray_tracer=True,
                                                      description=f"Small image: N={N}, H={H}, W={W}"))

        if num_tests > 9:
            # Test 10: Large number of points
            N = 1000
            H, W = 64, 64
            test_cases.append(self._create_test_case(N, H, W, use_ray_tracer=False,
                                                      description=f"Large N: N={N}"))

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            N = 100 + i * 50
            H, W = 64, 64
            test_cases.append(self._create_test_case(N, H, W, use_ray_tracer=(i % 2 == 0),
                                                      description=f"Additional test {i+1}: N={N}"))

        return test_cases[:num_tests]

    def _create_test_case(self, N, H, W, use_ray_tracer=True, description=""):
        """Create a single test case."""
        # Camera intrinsics
        focal = 500.0
        K = np.array([
            [focal, 0, W / 2],
            [0, focal, H / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        K_tensor = torch.from_numpy(K).to(self.device)

        # Camera extrinsics (random rotation and translation)
        axis = torch.randn(3, device=self.device)
        axis = axis / torch.norm(axis)
        angle = torch.rand(1, device=self.device) * np.pi
        K_mat = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], device=self.device)
        R = torch.eye(3, device=self.device) + torch.sin(angle) * K_mat + (1 - torch.cos(angle)) * (K_mat @ K_mat)
        T = torch.randn(3, device=self.device) * 2.0

        # Generate random 3D points
        xyz = torch.randn(N, 3, device=self.device) * 5.0

        # Generate random normals (normalized)
        normal_map = torch.randn(N, 3, device=self.device)
        normal_map = normal_map / torch.norm(normal_map, dim=-1, keepdim=True)

        # Generate random albedo
        albedo = torch.rand(N, 3, device=self.device) * 0.8 + 0.1

        # Generate random alpha
        render_alpha = torch.rand(N, 1, device=self.device) * 0.8 + 0.2

        # Generate random roughness
        roughness = torch.rand(N, 1, device=self.device) * 0.8 + 0.1

        # Generate random reflection strength
        refl_strength = torch.rand(N, 1, device=self.device) * 0.8 + 0.1

        # Generate random indirect light
        indirect_light = torch.rand(N, 3, device=self.device) * 0.5

        # Create mock objects
        envmap = MockEnvmap(device=self.device)
        pc = MockPC(use_ray_tracer=use_ray_tracer, device=self.device)

        HWK = (H, W, K)

        return {
            'envmap': envmap,
            'xyz': xyz,
            'albedo': albedo,
            'HWK': HWK,
            'R': R,
            'T': T,
            'normal_map': normal_map,
            'render_alpha': render_alpha,
            'scaling_modifier': 1.0,
            'refl_strength': refl_strength,
            'roughness': roughness,
            'pc': pc,
            'indirect_light': indirect_light,
            'description': description,
        }
