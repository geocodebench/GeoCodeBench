"""
Test Data Generator for BA_with_scale_shift() function.
Generates test cases with different configurations.
"""

from __future__ import annotations

import torch

# Import MockPoses for generating test data
from reference_implementation import MockPoses


class TestDataGenerator:
    """Generate test data for BA_with_scale_shift() function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []
        device = 'cpu'  # No CUDA

        # Test 1: Basic case with small dimensions
        B, P, ht, wd = 1, 4, 32, 32
        N = 2
        target = torch.randn(B, N, ht, wd, 2, device=device)
        weight = torch.ones(B, N, ht, wd, device=device)
        eta = torch.ones(B, P, ht, wd, device=device) * 0.01
        poses = MockPoses(B, P, device=device)
        disps = torch.rand(B, P, ht, wd, device=device) * 0.1 + 0.01
        intrinsics = torch.randn(B, N, 3, 3, device=device)
        intrinsics[:, :, 2, 2] = 1.0  # Make valid intrinsics
        ii = torch.randint(0, P, (N,), device=device)
        jj = torch.randint(0, P, (N,), device=device)
        mono_disps = torch.rand(B, P, ht, wd, device=device) * 0.1 + 0.01
        scales = torch.ones(B, P, device=device) * 1.0
        shifts = torch.zeros(B, P, device=device)
        valid_depth_mask = torch.ones(B, P, ht, wd, device=device, dtype=torch.bool)
        ignore_frames = 0

        test_cases.append({
            'target': target,
            'weight': weight,
            'eta': eta,
            'poses': poses,
            'disps': disps,
            'intrinsics': intrinsics,
            'ii': ii,
            'jj': jj,
            'mono_disps': mono_disps,
            'scales': scales,
            'shifts': shifts,
            'valid_depth_mask': valid_depth_mask,
            'ignore_frames': ignore_frames,
            'description': f'Basic: B={B}, P={P}, ht={ht}, wd={wd}, N={N}',
        })

        if num_tests > 1:
            # Test 2: Larger dimensions
            B, P, ht, wd = 1, 6, 64, 64
            N = 3
            target = torch.randn(B, N, ht, wd, 2, device=device)
            weight = torch.ones(B, N, ht, wd, device=device)
            eta = torch.ones(B, P, ht, wd, device=device) * 0.01
            poses = MockPoses(B, P, device=device)
            disps = torch.rand(B, P, ht, wd, device=device) * 0.1 + 0.01
            intrinsics = torch.randn(B, N, 3, 3, device=device)
            intrinsics[:, :, 2, 2] = 1.0
            ii = torch.randint(0, P, (N,), device=device)
            jj = torch.randint(0, P, (N,), device=device)
            mono_disps = torch.rand(B, P, ht, wd, device=device) * 0.1 + 0.01
            scales = torch.ones(B, P, device=device) * 1.0
            shifts = torch.zeros(B, P, device=device)
            valid_depth_mask = torch.ones(B, P, ht, wd, device=device, dtype=torch.bool)

            test_cases.append({
                'target': target,
                'weight': weight,
                'eta': eta,
                'poses': poses,
                'disps': disps,
                'intrinsics': intrinsics,
                'ii': ii,
                'jj': jj,
                'mono_disps': mono_disps,
                'scales': scales,
                'shifts': shifts,
                'valid_depth_mask': valid_depth_mask,
                'ignore_frames': ignore_frames,
                'description': f'Larger: B={B}, P={P}, ht={ht}, wd={wd}, N={N}',
            })

        if num_tests > 2:
            # Test 3: Different alpha
            B, P, ht, wd = 1, 5, 48, 48
            N = 2
            target = torch.randn(B, N, ht, wd, 2, device=device)
            weight = torch.ones(B, N, ht, wd, device=device)
            eta = torch.ones(B, P, ht, wd, device=device) * 0.01
            poses = MockPoses(B, P, device=device)
            disps = torch.rand(B, P, ht, wd, device=device) * 0.1 + 0.01
            intrinsics = torch.randn(B, N, 3, 3, device=device)
            intrinsics[:, :, 2, 2] = 1.0
            ii = torch.randint(0, P, (N,), device=device)
            jj = torch.randint(0, P, (N,), device=device)
            mono_disps = torch.rand(B, P, ht, wd, device=device) * 0.1 + 0.01
            scales = torch.ones(B, P, device=device) * 1.0
            shifts = torch.zeros(B, P, device=device)
            valid_depth_mask = torch.ones(B, P, ht, wd, device=device, dtype=torch.bool)

            test_cases.append({
                'target': target,
                'weight': weight,
                'eta': eta,
                'poses': poses,
                'disps': disps,
                'intrinsics': intrinsics,
                'ii': ii,
                'jj': jj,
                'mono_disps': mono_disps,
                'scales': scales,
                'shifts': shifts,
                'valid_depth_mask': valid_depth_mask,
                'ignore_frames': ignore_frames,
                'alpha': 2.0,
                'description': f'Different alpha: B={B}, P={P}, alpha=2.0',
            })

        if num_tests > 3:
            # Test 4: With ignore_frames
            B, P, ht, wd = 1, 6, 32, 32
            N = 2
            target = torch.randn(B, N, ht, wd, 2, device=device)
            weight = torch.ones(B, N, ht, wd, device=device)
            eta = torch.ones(B, P, ht, wd, device=device) * 0.01
            poses = MockPoses(B, P, device=device)
            disps = torch.rand(B, P, ht, wd, device=device) * 0.1 + 0.01
            intrinsics = torch.randn(B, N, 3, 3, device=device)
            intrinsics[:, :, 2, 2] = 1.0
            ii = torch.randint(0, P, (N,), device=device)
            jj = torch.randint(0, P, (N,), device=device)
            mono_disps = torch.rand(B, P, ht, wd, device=device) * 0.1 + 0.01
            scales = torch.ones(B, P, device=device) * 1.0
            shifts = torch.zeros(B, P, device=device)
            valid_depth_mask = torch.ones(B, P, ht, wd, device=device, dtype=torch.bool)

            test_cases.append({
                'target': target,
                'weight': weight,
                'eta': eta,
                'poses': poses,
                'disps': disps,
                'intrinsics': intrinsics,
                'ii': ii,
                'jj': jj,
                'mono_disps': mono_disps,
                'scales': scales,
                'shifts': shifts,
                'valid_depth_mask': valid_depth_mask,
                'ignore_frames': 2,
                'description': f'With ignore_frames: B={B}, P={P}, ignore_frames=2',
            })

        if num_tests > 4:
            # Test 5: Different scales and shifts
            B, P, ht, wd = 1, 4, 40, 40
            N = 2
            target = torch.randn(B, N, ht, wd, 2, device=device)
            weight = torch.ones(B, N, ht, wd, device=device)
            eta = torch.ones(B, P, ht, wd, device=device) * 0.01
            poses = MockPoses(B, P, device=device)
            disps = torch.rand(B, P, ht, wd, device=device) * 0.1 + 0.01
            intrinsics = torch.randn(B, N, 3, 3, device=device)
            intrinsics[:, :, 2, 2] = 1.0
            ii = torch.randint(0, P, (N,), device=device)
            jj = torch.randint(0, P, (N,), device=device)
            mono_disps = torch.rand(B, P, ht, wd, device=device) * 0.1 + 0.01
            scales = torch.ones(B, P, device=device) * 1.5
            shifts = torch.ones(B, P, device=device) * 0.1
            valid_depth_mask = torch.ones(B, P, ht, wd, device=device, dtype=torch.bool)

            test_cases.append({
                'target': target,
                'weight': weight,
                'eta': eta,
                'poses': poses,
                'disps': disps,
                'intrinsics': intrinsics,
                'ii': ii,
                'jj': jj,
                'mono_disps': mono_disps,
                'scales': scales,
                'shifts': shifts,
                'valid_depth_mask': valid_depth_mask,
                'ignore_frames': ignore_frames,
                'description': f'Different scales/shifts: B={B}, P={P}',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            B, P, ht, wd = 1, 3 + (i % 5), 32 + (i % 16) * 2, 32 + (i % 16) * 2
            N = 2 + (i % 3)
            target = torch.randn(B, N, ht, wd, 2, device=device)
            weight = torch.ones(B, N, ht, wd, device=device)
            eta = torch.ones(B, P, ht, wd, device=device) * 0.01
            poses = MockPoses(B, P, device=device)
            disps = torch.rand(B, P, ht, wd, device=device) * 0.1 + 0.01
            intrinsics = torch.randn(B, N, 3, 3, device=device)
            intrinsics[:, :, 2, 2] = 1.0
            ii = torch.randint(0, P, (N,), device=device)
            jj = torch.randint(0, P, (N,), device=device)
            mono_disps = torch.rand(B, P, ht, wd, device=device) * 0.1 + 0.01
            scales = torch.ones(B, P, device=device) * (1.0 + i * 0.1)
            shifts = torch.zeros(B, P, device=device)
            valid_depth_mask = torch.ones(B, P, ht, wd, device=device, dtype=torch.bool)

            test_cases.append({
                'target': target,
                'weight': weight,
                'eta': eta,
                'poses': poses,
                'disps': disps,
                'intrinsics': intrinsics,
                'ii': ii,
                'jj': jj,
                'mono_disps': mono_disps,
                'scales': scales,
                'shifts': shifts,
                'valid_depth_mask': valid_depth_mask,
                'ignore_frames': ignore_frames,
                'description': f'Additional test {i+1}: B={B}, P={P}, ht={ht}, wd={wd}',
            })

        return test_cases[:num_tests]
