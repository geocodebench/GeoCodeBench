"""
Test Data Generator for linear_match function.
"""

import torch


class TestDataGenerator:
    """Generate test data for linear_match function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small image
        H, W = 64, 64
        patch_size = 8
        d0 = torch.randn(1, H, W) * 5.0 + 10.0  # Depth around 10
        d1 = torch.randn(1, H, W) * 5.0 + 10.0
        mask = torch.ones(1, H, W)
        test_cases.append({
            'd0': d0,
            'd1': d1,
            'mask': mask,
            'patch_size': patch_size,
            'description': f'Basic: H={H}, W={W}, patch_size={patch_size}',
        })

        if num_tests > 1:
            # Test 2: Different patch size
            H, W = 80, 80
            patch_size = 10
            d0 = torch.randn(1, H, W) * 3.0 + 8.0
            d1 = torch.randn(1, H, W) * 3.0 + 8.0
            mask = torch.ones(1, H, W)
            test_cases.append({
                'd0': d0,
                'd1': d1,
                'mask': mask,
                'patch_size': patch_size,
                'description': f'Different patch: H={H}, W={W}, patch_size={patch_size}',
            })

        if num_tests > 2:
            # Test 3: With partial mask
            H, W = 96, 96
            patch_size = 16
            d0 = torch.randn(1, H, W) * 4.0 + 12.0
            d1 = torch.randn(1, H, W) * 4.0 + 12.0
            mask = torch.rand(1, H, W) > 0.3  # 70% valid pixels
            mask = mask.float()
            test_cases.append({
                'd0': d0,
                'd1': d1,
                'mask': mask,
                'patch_size': patch_size,
                'description': f'Partial mask: H={H}, W={W}, patch_size={patch_size}, mask~70%',
            })

        if num_tests > 3:
            # Test 4: Non-square image
            H, W = 64, 128
            patch_size = 8
            d0 = torch.randn(1, H, W) * 6.0 + 15.0
            d1 = torch.randn(1, H, W) * 6.0 + 15.0
            mask = torch.ones(1, H, W)
            test_cases.append({
                'd0': d0,
                'd1': d1,
                'mask': mask,
                'patch_size': patch_size,
                'description': f'Non-square: H={H}, W={W}, patch_size={patch_size}',
            })

        if num_tests > 4:
            # Test 5: Larger image
            H, W = 128, 128
            patch_size = 16
            d0 = torch.randn(1, H, W) * 5.0 + 20.0
            d1 = torch.randn(1, H, W) * 5.0 + 20.0
            mask = torch.ones(1, H, W)
            test_cases.append({
                'd0': d0,
                'd1': d1,
                'mask': mask,
                'patch_size': patch_size,
                'description': f'Large: H={H}, W={W}, patch_size={patch_size}',
            })

        if num_tests > 5:
            # Test 6: Small patch size
            H, W = 100, 100
            patch_size = 5
            d0 = torch.randn(1, H, W) * 4.0 + 10.0
            d1 = torch.randn(1, H, W) * 4.0 + 10.0
            mask = torch.ones(1, H, W)
            test_cases.append({
                'd0': d0,
                'd1': d1,
                'mask': mask,
                'patch_size': patch_size,
                'description': f'Small patch: H={H}, W={W}, patch_size={patch_size}',
            })

        if num_tests > 6:
            # Test 7: Sparse mask
            H, W = 80, 80
            patch_size = 10
            d0 = torch.randn(1, H, W) * 3.0 + 8.0
            d1 = torch.randn(1, H, W) * 3.0 + 8.0
            mask = torch.rand(1, H, W) > 0.5  # 50% valid pixels
            mask = mask.float()
            test_cases.append({
                'd0': d0,
                'd1': d1,
                'mask': mask,
                'patch_size': patch_size,
                'description': f'Sparse mask: H={H}, W={W}, patch_size={patch_size}, mask~50%',
            })

        if num_tests > 7:
            # Test 8: Non-divisible dimensions
            H, W = 75, 90
            patch_size = 10
            d0 = torch.randn(1, H, W) * 5.0 + 12.0
            d1 = torch.randn(1, H, W) * 5.0 + 12.0
            mask = torch.ones(1, H, W)
            test_cases.append({
                'd0': d0,
                'd1': d1,
                'mask': mask,
                'patch_size': patch_size,
                'description': f'Non-divisible: H={H}, W={W}, patch_size={patch_size}',
            })

        if num_tests > 8:
            # Test 9: Different depth ranges
            H, W = 96, 96
            patch_size = 12
            d0 = torch.randn(1, H, W) * 2.0 + 5.0
            d1 = torch.randn(1, H, W) * 10.0 + 50.0
            mask = torch.ones(1, H, W)
            test_cases.append({
                'd0': d0,
                'd1': d1,
                'mask': mask,
                'patch_size': patch_size,
                'description': f'Different ranges: H={H}, W={W}, patch_size={patch_size}',
            })

        if num_tests > 9:
            # Test 10: Complex mask pattern
            H, W = 120, 120
            patch_size = 15
            d0 = torch.randn(1, H, W) * 4.0 + 10.0
            d1 = torch.randn(1, H, W) * 4.0 + 10.0
            # Create a checkerboard-like mask pattern
            mask = torch.zeros(1, H, W)
            mask[:, ::2, ::2] = 1.0
            mask[:, 1::2, 1::2] = 1.0
            test_cases.append({
                'd0': d0,
                'd1': d1,
                'mask': mask,
                'patch_size': patch_size,
                'description': f'Complex mask: H={H}, W={W}, patch_size={patch_size}, checkerboard',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            H = 64 + i * 16
            W = 64 + i * 16
            patch_size = 8 + (i % 3) * 4
            d0 = torch.randn(1, H, W) * 5.0 + 10.0
            d1 = torch.randn(1, H, W) * 5.0 + 10.0
            mask = torch.ones(1, H, W)

            test_cases.append({
                'd0': d0,
                'd1': d1,
                'mask': mask,
                'patch_size': patch_size,
                'description': f'Additional test {i+1}: H={H}, W={W}, patch_size={patch_size}',
            })

        return test_cases[:num_tests]
