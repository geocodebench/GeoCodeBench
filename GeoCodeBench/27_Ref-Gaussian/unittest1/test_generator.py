"""
Test Data Generator for get_specular_color_surfel function.
Generates test cases with different configurations.
"""

import torch
import numpy as np

from helper_functions import MockEnvmap, MockPC, create_mock_FG_LUT


class TestDataGenerator:
    """Generate test data for get_specular_color_surfel function."""

    def __init__(self, seed=42, device='cpu'):
        self.seed = seed
        self.device = device
        torch.manual_seed(seed)
        np.random.seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case without ray tracer, no indirect light
        H, W = 8, 16
        K = np.array([[100.0, 0.0, W/2], [0.0, 100.0, H/2], [0.0, 0.0, 1.0]])
        R = torch.eye(3, device=self.device)
        T = torch.zeros(3, device=self.device)
        envmap = MockEnvmap(H=16, W=32, device=self.device)
        albedo = torch.rand(H, W, 3, device=self.device) * 0.5 + 0.3
        normal_map = torch.randn(H, W, 3, device=self.device)
        normal_map = normal_map / torch.norm(normal_map, dim=-1, keepdim=True)
        render_alpha = torch.rand(H, W, 1, device=self.device)
        refl_strength = torch.rand(H, W, 1, device=self.device) * 0.5
        roughness = torch.rand(H, W, 1, device=self.device) * 0.3 + 0.1
        pc = MockPC(has_ray_tracer=False, device=self.device)

        test_cases.append({
            'envmap': envmap,
            'albedo': albedo,
            'HWK': (H, W, K),
            'R': R,
            'T': T,
            'normal_map': normal_map,
            'render_alpha': render_alpha,
            'refl_strength': refl_strength,
            'roughness': roughness,
            'pc': pc,
            'surf_depth': None,
            'indirect_light': None,
            'description': f'Basic: H={H}, W={W}, no ray tracer, no indirect light',
        })

        if num_tests > 1:
            # Test 2: With ray tracer and indirect light
            H, W = 12, 24
            K = np.array([[150.0, 0.0, W/2], [0.0, 150.0, H/2], [0.0, 0.0, 1.0]])
            R = torch.eye(3, device=self.device)
            T = torch.tensor([0.1, 0.2, 0.3], device=self.device)
            envmap = MockEnvmap(H=16, W=32, device=self.device)
            albedo = torch.rand(H, W, 3, device=self.device) * 0.5 + 0.3
            normal_map = torch.randn(H, W, 3, device=self.device)
            normal_map = normal_map / torch.norm(normal_map, dim=-1, keepdim=True)
            render_alpha = torch.rand(H, W, 1, device=self.device)
            refl_strength = torch.rand(H, W, 1, device=self.device) * 0.5
            roughness = torch.rand(H, W, 1, device=self.device) * 0.3 + 0.1
            pc = MockPC(has_ray_tracer=True, device=self.device)
            surf_depth = torch.rand(1, H, W, device=self.device) * 5 + 1
            indirect_light = torch.rand(H, W, 3, device=self.device) * 0.3

            test_cases.append({
                'envmap': envmap,
                'albedo': albedo,
                'HWK': (H, W, K),
                'R': R,
                'T': T,
                'normal_map': normal_map,
                'render_alpha': render_alpha,
                'refl_strength': refl_strength,
                'roughness': roughness,
                'pc': pc,
                'surf_depth': surf_depth,
                'indirect_light': indirect_light,
                'description': f'With ray tracer: H={H}, W={W}, with indirect light',
            })

        if num_tests > 2:
            # Test 3: Different image size
            H, W = 16, 32
            K = np.array([[200.0, 0.0, W/2], [0.0, 200.0, H/2], [0.0, 0.0, 1.0]])
            R = torch.eye(3, device=self.device)
            T = torch.zeros(3, device=self.device)
            envmap = MockEnvmap(H=16, W=32, device=self.device)
            albedo = torch.rand(H, W, 3, device=self.device) * 0.5 + 0.3
            normal_map = torch.randn(H, W, 3, device=self.device)
            normal_map = normal_map / torch.norm(normal_map, dim=-1, keepdim=True)
            render_alpha = torch.rand(H, W, 1, device=self.device)
            refl_strength = torch.rand(H, W, 1, device=self.device) * 0.5
            roughness = torch.rand(H, W, 1, device=self.device) * 0.3 + 0.1
            pc = MockPC(has_ray_tracer=False, device=self.device)

            test_cases.append({
                'envmap': envmap,
                'albedo': albedo,
                'HWK': (H, W, K),
                'R': R,
                'T': T,
                'normal_map': normal_map,
                'render_alpha': render_alpha,
                'refl_strength': refl_strength,
                'roughness': roughness,
                'pc': pc,
                'surf_depth': None,
                'indirect_light': None,
                'description': f'Larger size: H={H}, W={W}',
            })

        if num_tests > 3:
            # Test 4: With rotated camera
            H, W = 10, 20
            K = np.array([[120.0, 0.0, W/2], [0.0, 120.0, H/2], [0.0, 0.0, 1.0]])
            angle = np.pi / 6
            R = torch.tensor([[np.cos(angle), 0, np.sin(angle)],
                             [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]], device=self.device, dtype=torch.float32)
            T = torch.tensor([0.5, 0.0, 0.5], device=self.device)
            envmap = MockEnvmap(H=16, W=32, device=self.device)
            albedo = torch.rand(H, W, 3, device=self.device) * 0.5 + 0.3
            normal_map = torch.randn(H, W, 3, device=self.device)
            normal_map = normal_map / torch.norm(normal_map, dim=-1, keepdim=True)
            render_alpha = torch.rand(H, W, 1, device=self.device)
            refl_strength = torch.rand(H, W, 1, device=self.device) * 0.5
            roughness = torch.rand(H, W, 1, device=self.device) * 0.3 + 0.1
            pc = MockPC(has_ray_tracer=False, device=self.device)

            test_cases.append({
                'envmap': envmap,
                'albedo': albedo,
                'HWK': (H, W, K),
                'R': R,
                'T': T,
                'normal_map': normal_map,
                'render_alpha': render_alpha,
                'refl_strength': refl_strength,
                'roughness': roughness,
                'pc': pc,
                'surf_depth': None,
                'indirect_light': None,
                'description': f'Rotated camera: H={H}, W={W}, rotated 30° around Y',
            })

        if num_tests > 4:
            # Test 5: Edge case - high roughness
            H, W = 8, 16
            K = np.array([[100.0, 0.0, W/2], [0.0, 100.0, H/2], [0.0, 0.0, 1.0]])
            R = torch.eye(3, device=self.device)
            T = torch.zeros(3, device=self.device)
            envmap = MockEnvmap(H=16, W=32, device=self.device)
            albedo = torch.rand(H, W, 3, device=self.device) * 0.5 + 0.3
            normal_map = torch.randn(H, W, 3, device=self.device)
            normal_map = normal_map / torch.norm(normal_map, dim=-1, keepdim=True)
            render_alpha = torch.rand(H, W, 1, device=self.device)
            refl_strength = torch.rand(H, W, 1, device=self.device) * 0.5
            roughness = torch.rand(H, W, 1, device=self.device) * 0.5 + 0.5  # High roughness
            pc = MockPC(has_ray_tracer=False, device=self.device)

            test_cases.append({
                'envmap': envmap,
                'albedo': albedo,
                'HWK': (H, W, K),
                'R': R,
                'T': T,
                'normal_map': normal_map,
                'render_alpha': render_alpha,
                'refl_strength': refl_strength,
                'roughness': roughness,
                'pc': pc,
                'surf_depth': None,
                'indirect_light': None,
                'description': f'High roughness: roughness=[0.5, 1.0]',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            H = 8 + i * 2
            W = 16 + i * 4
            K = np.array([[100.0 + i * 20, 0.0, W/2], [0.0, 100.0 + i * 20, H/2], [0.0, 0.0, 1.0]])
            R = torch.eye(3, device=self.device)
            T = torch.zeros(3, device=self.device)
            envmap = MockEnvmap(H=16, W=32, device=self.device)
            albedo = torch.rand(H, W, 3, device=self.device) * 0.5 + 0.3
            normal_map = torch.randn(H, W, 3, device=self.device)
            normal_map = normal_map / torch.norm(normal_map, dim=-1, keepdim=True)
            render_alpha = torch.rand(H, W, 1, device=self.device)
            refl_strength = torch.rand(H, W, 1, device=self.device) * 0.5
            roughness = torch.rand(H, W, 1, device=self.device) * 0.3 + 0.1
            has_tracer = (i % 2 == 0)
            pc = MockPC(has_ray_tracer=has_tracer, device=self.device)
            surf_depth = torch.rand(1, H, W, device=self.device) * 5 + 1 if has_tracer else None
            indirect_light = torch.rand(H, W, 3, device=self.device) * 0.3 if has_tracer else None

            test_cases.append({
                'envmap': envmap,
                'albedo': albedo,
                'HWK': (H, W, K),
                'R': R,
                'T': T,
                'normal_map': normal_map,
                'render_alpha': render_alpha,
                'refl_strength': refl_strength,
                'roughness': roughness,
                'pc': pc,
                'surf_depth': surf_depth,
                'indirect_light': indirect_light,
                'description': f'Additional test {i+1}: H={H}, W={W}, tracer={has_tracer}',
            })

        return test_cases[:num_tests]
