"""
Test Data Generator for compute_hom function.
Generates test cases with different configurations.
"""

import torch
import numpy as np

from reference_implementation import MockView


class TestDataGenerator:
    """Generate test data for compute_hom function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Small image, patch_size=3
        H, W = 32, 32
        depth = torch.rand(1, H, W) * 5.0 + 1.0  # depth in [1, 6]
        normal = torch.randn(3, H, W)
        normal = torch.nn.functional.normalize(normal, p=2, dim=0)
        points = torch.randn(H * W, 3) * 2.0  # 3D points
        view_ref = MockView(H, W)
        view_src = MockView(H, W)
        test_cases.append({
            'depth': depth,
            'normal': normal,
            'points': points,
            'view_ref': view_ref,
            'view_src': view_src,
            'patch_size': 3,
            'patch_offset': None,
            'description': f'Small: H={H}, W={W}, patch_size=3',
        })

        if num_tests > 1:
            # Test 2: Medium image, patch_size=3
            H, W = 48, 64
            depth = torch.rand(1, H, W) * 5.0 + 1.0
            normal = torch.randn(3, H, W)
            normal = torch.nn.functional.normalize(normal, p=2, dim=0)
            points = torch.randn(H * W, 3) * 2.0
            view_ref = MockView(H, W)
            view_src = MockView(H, W)
            test_cases.append({
                'depth': depth,
                'normal': normal,
                'points': points,
                'view_ref': view_ref,
                'view_src': view_src,
                'patch_size': 3,
                'patch_offset': None,
                'description': f'Medium: H={H}, W={W}, patch_size=3',
            })

        if num_tests > 2:
            # Test 3: Different patch_size
            H, W = 32, 32
            depth = torch.rand(1, H, W) * 5.0 + 1.0
            normal = torch.randn(3, H, W)
            normal = torch.nn.functional.normalize(normal, p=2, dim=0)
            points = torch.randn(H * W, 3) * 2.0
            view_ref = MockView(H, W)
            view_src = MockView(H, W)
            test_cases.append({
                'depth': depth,
                'normal': normal,
                'points': points,
                'view_ref': view_ref,
                'view_src': view_src,
                'patch_size': 2,
                'patch_offset': None,
                'description': f'Patch size 2: H={H}, W={W}, patch_size=2',
            })

        if num_tests > 3:
            # Test 4: Large depth values
            H, W = 40, 40
            depth = torch.rand(1, H, W) * 10.0 + 5.0  # depth in [5, 15]
            normal = torch.randn(3, H, W)
            normal = torch.nn.functional.normalize(normal, p=2, dim=0)
            points = torch.randn(H * W, 3) * 5.0
            view_ref = MockView(H, W)
            view_src = MockView(H, W)
            test_cases.append({
                'depth': depth,
                'normal': normal,
                'points': points,
                'view_ref': view_ref,
                'view_src': view_src,
                'patch_size': 3,
                'patch_offset': None,
                'description': f'Large depth: H={H}, W={W}, depth in [5,15]',
            })

        if num_tests > 4:
            # Test 5: Different image aspect ratio
            H, W = 30, 50
            depth = torch.rand(1, H, W) * 5.0 + 1.0
            normal = torch.randn(3, H, W)
            normal = torch.nn.functional.normalize(normal, p=2, dim=0)
            points = torch.randn(H * W, 3) * 2.0
            view_ref = MockView(H, W)
            view_src = MockView(H, W)
            test_cases.append({
                'depth': depth,
                'normal': normal,
                'points': points,
                'view_ref': view_ref,
                'view_src': view_src,
                'patch_size': 3,
                'patch_offset': None,
                'description': f'Aspect ratio: H={H}, W={W}',
            })

        if num_tests > 5:
            # Test 6: patch_size=1
            H, W = 32, 32
            depth = torch.rand(1, H, W) * 5.0 + 1.0
            normal = torch.randn(3, H, W)
            normal = torch.nn.functional.normalize(normal, p=2, dim=0)
            points = torch.randn(H * W, 3) * 2.0
            view_ref = MockView(H, W)
            view_src = MockView(H, W)
            test_cases.append({
                'depth': depth,
                'normal': normal,
                'points': points,
                'view_ref': view_ref,
                'view_src': view_src,
                'patch_size': 1,
                'patch_offset': None,
                'description': f'Patch size 1: H={H}, W={W}, patch_size=1',
            })

        if num_tests > 6:
            # Test 7: Uniform normal
            H, W = 32, 32
            depth = torch.rand(1, H, W) * 5.0 + 1.0
            normal = torch.zeros(3, H, W)
            normal[2, :, :] = 1.0  # All normals pointing in z direction
            points = torch.randn(H * W, 3) * 2.0
            view_ref = MockView(H, W)
            view_src = MockView(H, W)
            test_cases.append({
                'depth': depth,
                'normal': normal,
                'points': points,
                'view_ref': view_ref,
                'view_src': view_src,
                'patch_size': 3,
                'patch_offset': None,
                'description': f'Uniform normal: H={H}, W={W}, normal=(0,0,1)',
            })

        if num_tests > 7:
            # Test 8: Small depth values
            H, W = 36, 36
            depth = torch.rand(1, H, W) * 0.5 + 0.5  # depth in [0.5, 1.0]
            normal = torch.randn(3, H, W)
            normal = torch.nn.functional.normalize(normal, p=2, dim=0)
            points = torch.randn(H * W, 3) * 0.5
            view_ref = MockView(H, W)
            view_src = MockView(H, W)
            test_cases.append({
                'depth': depth,
                'normal': normal,
                'points': points,
                'view_ref': view_ref,
                'view_src': view_src,
                'patch_size': 3,
                'patch_offset': None,
                'description': f'Small depth: H={H}, W={W}, depth in [0.5,1.0]',
            })

        if num_tests > 8:
            # Test 9: Larger image
            H, W = 64, 64
            depth = torch.rand(1, H, W) * 5.0 + 1.0
            normal = torch.randn(3, H, W)
            normal = torch.nn.functional.normalize(normal, p=2, dim=0)
            points = torch.randn(H * W, 3) * 2.0
            view_ref = MockView(H, W)
            view_src = MockView(H, W)
            test_cases.append({
                'depth': depth,
                'normal': normal,
                'points': points,
                'view_ref': view_ref,
                'view_src': view_src,
                'patch_size': 3,
                'patch_offset': None,
                'description': f'Large: H={H}, W={W}, patch_size=3',
            })

        if num_tests > 9:
            # Test 10: patch_size=4
            H, W = 40, 40
            depth = torch.rand(1, H, W) * 5.0 + 1.0
            normal = torch.randn(3, H, W)
            normal = torch.nn.functional.normalize(normal, p=2, dim=0)
            points = torch.randn(H * W, 3) * 2.0
            view_ref = MockView(H, W)
            view_src = MockView(H, W)
            test_cases.append({
                'depth': depth,
                'normal': normal,
                'points': points,
                'view_ref': view_ref,
                'view_src': view_src,
                'patch_size': 4,
                'patch_offset': None,
                'description': f'Patch size 4: H={H}, W={W}, patch_size=4',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            H = 32 + i * 4
            W = 32 + i * 4
            depth = torch.rand(1, H, W) * 5.0 + 1.0
            normal = torch.randn(3, H, W)
            normal = torch.nn.functional.normalize(normal, p=2, dim=0)
            points = torch.randn(H * W, 3) * 2.0
            view_ref = MockView(H, W)
            view_src = MockView(H, W)

            test_cases.append({
                'depth': depth,
                'normal': normal,
                'points': points,
                'view_ref': view_ref,
                'view_src': view_src,
                'patch_size': 3,
                'patch_offset': None,
                'description': f'Additional test {i+1}: H={H}, W={W}',
            })

        return test_cases[:num_tests]
