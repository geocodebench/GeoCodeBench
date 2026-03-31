"""
Test data generator for init_reso_scheduler function.
"""

import torch


class TestDataGenerator:
    """Generate test data for init_reso_scheduler function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small images
        images = [torch.randn(3, 64, 64) for _ in range(2)]
        test_cases.append({
            'original_images': images,
            'description': f'Basic: 2 images, size 64x64',
        })

        if num_tests > 1:
            # Test 2: Different image size
            images = [torch.randn(3, 128, 128) for _ in range(3)]
            test_cases.append({
                'original_images': images,
                'description': f'Medium: 3 images, size 128x128',
            })

        if num_tests > 2:
            # Test 3: Single image
            images = [torch.randn(3, 96, 96)]
            test_cases.append({
                'original_images': images,
                'description': f'Single: 1 image, size 96x96',
            })

        if num_tests > 3:
            # Test 4: More images
            images = [torch.randn(3, 80, 80) for _ in range(5)]
            test_cases.append({
                'original_images': images,
                'description': f'Multiple: 5 images, size 80x80',
            })

        if num_tests > 4:
            # Test 5: Larger images
            images = [torch.randn(3, 160, 160) for _ in range(2)]
            test_cases.append({
                'original_images': images,
                'description': f'Large: 2 images, size 160x160',
            })

        if num_tests > 5:
            # Test 6: Very small images
            images = [torch.randn(3, 32, 32) for _ in range(4)]
            test_cases.append({
                'original_images': images,
                'description': f'Small: 4 images, size 32x32',
            })

        if num_tests > 6:
            # Test 7: Rectangular images
            images = [torch.randn(3, 64, 96) for _ in range(3)]
            test_cases.append({
                'original_images': images,
                'description': f'Rectangular: 3 images, size 64x96',
            })

        if num_tests > 7:
            # Test 8: Many images
            images = [torch.randn(3, 72, 72) for _ in range(8)]
            test_cases.append({
                'original_images': images,
                'description': f'Many: 8 images, size 72x72',
            })

        if num_tests > 8:
            # Test 9: Different aspect ratio
            images = [torch.randn(3, 48, 128) for _ in range(2)]
            test_cases.append({
                'original_images': images,
                'description': f'Wide: 2 images, size 48x128',
            })

        if num_tests > 9:
            # Test 10: Grayscale-like (1 channel)
            images = [torch.randn(1, 64, 64) for _ in range(3)]
            test_cases.append({
                'original_images': images,
                'description': f'Grayscale: 3 images, 1 channel, size 64x64',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            num_images = 2 + (i % 4)
            size = 64 + i * 8
            images = [torch.randn(3, size, size) for _ in range(num_images)]

            test_cases.append({
                'original_images': images,
                'description': f'Additional test {i+1}: {num_images} images, size {size}x{size}',
            })

        return test_cases[:num_tests]
