"""
Test Data Generator for get_candidate_poses function.
"""

from __future__ import annotations

import torch
import numpy as np


class MockScene:
    """Mock scene object for testing."""
    def __init__(self, num_imgs):
        self.imgs = [torch.randn(3, 512, 704) for _ in range(num_imgs)]


class MockVcOpts:
    """Mock view crafter options."""
    def __init__(self):
        self.center_scale = 0.5
        self.elevation = 0.0
        self.d_r = 0.0  # Default radius offset for candidate poses


class MockSelf:
    """Mock self object with necessary attributes."""
    def __init__(self, num_views=10, device='cpu'):
        self.device = device
        self.scene = MockScene(num_views)
        self.pcd = [torch.randn(1000, 3, device=device) for _ in range(num_views)]
        self.d_images = [
            {'img_ori': torch.randn(1, 3, 512, 704, device=device)}
            for _ in range(num_views)
        ]
        self.c2ws = torch.eye(4, device=device).unsqueeze(0).repeat(num_views, 1, 1)
        self.principal_points = torch.tensor([[256, 352]], device=device).repeat(num_views, 1)
        self.focals = torch.tensor([[500, 500]], device=device).repeat(num_views, 1)
        self.depth = [torch.ones(512, 704, device=device) * 2.0 for _ in range(num_views)]
        self.d_H = 512
        self.d_W = 704
        self.vc_opts = MockVcOpts()


class TestDataGenerator:
    """Generate test data for get_candidate_poses function."""

    def __init__(self, seed=42, device='cpu'):
        self.seed = seed
        self.device = device
        torch.manual_seed(seed)
        np.random.seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with 2x2 grid
        mock_self = MockSelf(num_views=10, device=self.device)
        d_phi = [-10.0, 0.0]
        d_theta = [-10.0, 0.0]
        fovx = 1.0
        fovy = 1.0
        test_cases.append({
            'mock_self': mock_self,
            'd_phi': d_phi,
            'd_theta': d_theta,
            'fovx': fovx,
            'fovy': fovy,
            'which_train_view': 5,
            'pc_render_single_view': True,
            'ignore_0_0': False,
            'description': f'Basic: 2x2 grid, d_phi={d_phi}, d_theta={d_theta}',
        })

        if num_tests > 1:
            # Test 2: 3x3 grid
            mock_self = MockSelf(num_views=10, device=self.device)
            d_phi = [-15.0, 0.0, 15.0]
            d_theta = [-15.0, 0.0, 15.0]
            test_cases.append({
                'mock_self': mock_self,
                'd_phi': d_phi,
                'd_theta': d_theta,
                'fovx': 1.2,
                'fovy': 1.2,
                'which_train_view': 5,
                'pc_render_single_view': True,
                'ignore_0_0': False,
                'description': f'3x3 grid: d_phi={d_phi}, d_theta={d_theta}',
            })

        if num_tests > 2:
            # Test 3: With ignore_0_0 enabled
            mock_self = MockSelf(num_views=10, device=self.device)
            d_phi = [-10.0, 0.0, 10.0]
            d_theta = [-10.0, 0.0, 10.0]
            test_cases.append({
                'mock_self': mock_self,
                'd_phi': d_phi,
                'd_theta': d_theta,
                'fovx': 1.0,
                'fovy': 1.0,
                'which_train_view': 3,
                'pc_render_single_view': True,
                'ignore_0_0': True,
                'description': f'3x3 grid with ignore_0_0: d_phi={d_phi}, d_theta={d_theta}',
            })

        if num_tests > 3:
            # Test 4: Asymmetric grid
            mock_self = MockSelf(num_views=10, device=self.device)
            d_phi = [-20.0, -10.0, 0.0, 10.0]
            d_theta = [-15.0, 0.0]
            test_cases.append({
                'mock_self': mock_self,
                'd_phi': d_phi,
                'd_theta': d_theta,
                'fovx': 1.5,
                'fovy': 1.0,
                'which_train_view': 7,
                'pc_render_single_view': True,
                'ignore_0_0': False,
                'description': f'Asymmetric: 4x2 grid, d_phi={d_phi}, d_theta={d_theta}',
            })

        if num_tests > 4:
            # Test 5: Different view index
            mock_self = MockSelf(num_views=10, device=self.device)
            d_phi = [-5.0, 0.0, 5.0]
            d_theta = [-5.0, 0.0, 5.0]
            test_cases.append({
                'mock_self': mock_self,
                'd_phi': d_phi,
                'd_theta': d_theta,
                'fovx': 0.8,
                'fovy': 0.8,
                'which_train_view': 2,
                'pc_render_single_view': True,
                'ignore_0_0': False,
                'description': f'Different view (idx=2): 3x3 grid',
            })

        if num_tests > 5:
            # Test 6: Multi-view point cloud
            mock_self = MockSelf(num_views=10, device=self.device)
            d_phi = [-10.0, 10.0]
            d_theta = [-10.0, 10.0]
            test_cases.append({
                'mock_self': mock_self,
                'd_phi': d_phi,
                'd_theta': d_theta,
                'fovx': 1.0,
                'fovy': 1.0,
                'which_train_view': 5,
                'pc_render_single_view': False,
                'ignore_0_0': False,
                'description': f'Multi-view PC: 2x2 grid, pc_render_single_view=False',
            })

        if num_tests > 6:
            # Test 7: Single angle each
            mock_self = MockSelf(num_views=10, device=self.device)
            d_phi = [0.0]
            d_theta = [0.0]
            test_cases.append({
                'mock_self': mock_self,
                'd_phi': d_phi,
                'd_theta': d_theta,
                'fovx': 1.0,
                'fovy': 1.0,
                'which_train_view': 5,
                'pc_render_single_view': True,
                'ignore_0_0': False,
                'description': f'Single pose: d_phi=[0], d_theta=[0]',
            })

        if num_tests > 7:
            # Test 8: Large grid
            mock_self = MockSelf(num_views=10, device=self.device)
            d_phi = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
            d_theta = [-20.0, -10.0, 0.0, 10.0, 20.0]
            test_cases.append({
                'mock_self': mock_self,
                'd_phi': d_phi,
                'd_theta': d_theta,
                'fovx': 2.0,
                'fovy': 1.5,
                'which_train_view': 5,
                'pc_render_single_view': True,
                'ignore_0_0': False,
                'description': f'Large grid: 7x5 grid (35 poses)',
            })

        if num_tests > 8:
            # Test 9: Negative angles only
            mock_self = MockSelf(num_views=10, device=self.device)
            d_phi = [-25.0, -15.0, -5.0]
            d_theta = [-25.0, -15.0, -5.0]
            test_cases.append({
                'mock_self': mock_self,
                'd_phi': d_phi,
                'd_theta': d_theta,
                'fovx': 1.0,
                'fovy': 1.0,
                'which_train_view': 4,
                'pc_render_single_view': True,
                'ignore_0_0': False,
                'description': f'Negative angles: 3x3 grid with negative values',
            })

        if num_tests > 9:
            # Test 10: Positive angles only
            mock_self = MockSelf(num_views=10, device=self.device)
            d_phi = [5.0, 15.0, 25.0]
            d_theta = [5.0, 15.0, 25.0]
            test_cases.append({
                'mock_self': mock_self,
                'd_phi': d_phi,
                'd_theta': d_theta,
                'fovx': 1.0,
                'fovy': 1.0,
                'which_train_view': 6,
                'pc_render_single_view': True,
                'ignore_0_0': False,
                'description': f'Positive angles: 3x3 grid with positive values',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            mock_self = MockSelf(num_views=10, device=self.device)
            n_phi = 2 + (i % 3)
            n_theta = 2 + ((i + 1) % 3)
            d_phi = list(np.linspace(-20, 20, n_phi))
            d_theta = list(np.linspace(-15, 15, n_theta))

            test_cases.append({
                'mock_self': mock_self,
                'd_phi': d_phi,
                'd_theta': d_theta,
                'fovx': 1.0 + i * 0.1,
                'fovy': 1.0 + i * 0.1,
                'which_train_view': 5,
                'pc_render_single_view': True,
                'ignore_0_0': False,
                'description': f'Additional test {i+1}: {n_phi}x{n_theta} grid',
            })

        return test_cases[:num_tests]
