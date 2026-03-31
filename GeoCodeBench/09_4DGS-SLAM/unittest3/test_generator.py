"""
Test Data Generator for image_gradient function.
Generates test cases with different configurations.
"""

import torch


class TestDataGenerator:
    """Generate test data for image_gradient function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with single channel grayscale image
        image = torch.randn(1, 64, 64)
        test_cases.append({
            'image': image,
            'description': f'Basic: 1 channel, 64x64 grayscale image',
        })

        if num_tests > 1:
            # Test 2: RGB image (3 channels)
            image = torch.randn(3, 128, 128)
            test_cases.append({
                'image': image,
                'description': f'RGB: 3 channels, 128x128 color image',
            })

        if num_tests > 2:
            # Test 3: Small image
            image = torch.randn(1, 16, 16)
            test_cases.append({
                'image': image,
                'description': f'Small: 1 channel, 16x16 image',
            })

        if num_tests > 3:
            # Test 4: Large image
            image = torch.randn(3, 256, 256)
            test_cases.append({
                'image': image,
                'description': f'Large: 3 channels, 256x256 image',
            })

        if num_tests > 4:
            # Test 5: Multi-channel image (more than 3 channels)
            image = torch.randn(5, 64, 64)
            test_cases.append({
                'image': image,
                'description': f'Multi-channel: 5 channels, 64x64 image',
            })

        if num_tests > 5:
            # Test 6: Very small image
            image = torch.randn(1, 8, 8)
            test_cases.append({
                'image': image,
                'description': f'Very small: 1 channel, 8x8 image',
            })

        if num_tests > 6:
            # Test 7: Rectangular image
            image = torch.randn(3, 64, 128)
            test_cases.append({
                'image': image,
                'description': f'Rectangular: 3 channels, 64x128 image',
            })

        if num_tests > 7:
            # Test 8: High resolution image
            image = torch.randn(1, 512, 512)
            test_cases.append({
                'image': image,
                'description': f'High-res: 1 channel, 512x512 image',
            })

        if num_tests > 8:
            # Test 9: Many channels
            image = torch.randn(10, 32, 32)
            test_cases.append({
                'image': image,
                'description': f'Many channels: 10 channels, 32x32 image',
            })

        if num_tests > 9:
            # Test 10: Edge case - very small rectangular
            image = torch.randn(2, 4, 8)
            test_cases.append({
                'image': image,
                'description': f'Edge case: 2 channels, 4x8 image',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            channels = 1 + (i % 5)  # 1-5 channels
            height = 16 + (i % 8) * 8  # 16, 24, 32, ..., 64
            width = 16 + (i % 8) * 8  # 16, 24, 32, ..., 64
            image = torch.randn(channels, height, width)

            test_cases.append({
                'image': image,
                'description': f'Additional test {i+1}: {channels} channels, {height}x{width} image',
            })

        return test_cases[:num_tests]
