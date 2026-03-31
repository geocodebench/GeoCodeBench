"""
Test Data Generator for high_frequency_strength() and patchify_and_get_fdomain().
Generates test cases with different configurations.
"""

import numpy as np


class TestDataGenerator:
    """Generate test data for high_frequency_strength() and patchify_and_get_fdomain() functions."""

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case for high_frequency_strength - small patch
        patch1 = np.random.rand(16, 16).astype(np.float64)
        test_cases.append({
            'function': 'high_frequency_strength',
            'args': {'patch': patch1},
            'description': f'high_frequency_strength: patch shape {patch1.shape}',
        })

        if num_tests > 1:
            # Test 2: Basic case for patchify_and_get_fdomain - small image
            image1 = np.random.rand(32, 32, 3).astype(np.float64)
            patch_size1 = (8, 8)
            test_cases.append({
                'function': 'patchify_and_get_fdomain',
                'args': {'image': image1, 'patch_size': patch_size1},
                'description': f'patchify_and_get_fdomain: image shape {image1.shape}, patch_size {patch_size1}',
            })

        if num_tests > 2:
            # Test 3: high_frequency_strength - larger patch
            patch2 = np.random.rand(32, 32).astype(np.float64)
            test_cases.append({
                'function': 'high_frequency_strength',
                'args': {'patch': patch2},
                'description': f'high_frequency_strength: patch shape {patch2.shape}',
            })

        if num_tests > 3:
            # Test 4: patchify_and_get_fdomain - grayscale image
            image2 = np.random.rand(64, 64).astype(np.float64)
            patch_size2 = (16, 16)
            test_cases.append({
                'function': 'patchify_and_get_fdomain',
                'args': {'image': image2, 'patch_size': patch_size2},
                'description': f'patchify_and_get_fdomain: grayscale image shape {image2.shape}, patch_size {patch_size2}',
            })

        if num_tests > 4:
            # Test 5: high_frequency_strength - square patch with specific pattern
            patch3 = np.random.rand(24, 24).astype(np.float64)
            test_cases.append({
                'function': 'high_frequency_strength',
                'args': {'patch': patch3},
                'description': f'high_frequency_strength: patch shape {patch3.shape}',
            })

        if num_tests > 5:
            # Test 6: patchify_and_get_fdomain - larger image
            image3 = np.random.rand(80, 80, 3).astype(np.float64)
            patch_size3 = (20, 20)
            test_cases.append({
                'function': 'patchify_and_get_fdomain',
                'args': {'image': image3, 'patch_size': patch_size3},
                'description': f'patchify_and_get_fdomain: image shape {image3.shape}, patch_size {patch_size3}',
            })

        if num_tests > 6:
            # Test 7: high_frequency_strength - edge case with zeros
            patch4 = np.zeros((16, 16), dtype=np.float64)
            test_cases.append({
                'function': 'high_frequency_strength',
                'args': {'patch': patch4},
                'description': f'high_frequency_strength: zero patch shape {patch4.shape}',
            })

        if num_tests > 7:
            # Test 8: patchify_and_get_fdomain - RGB image with non-divisible size
            image4 = np.random.rand(50, 50, 3).astype(np.float64)
            patch_size4 = (12, 12)
            test_cases.append({
                'function': 'patchify_and_get_fdomain',
                'args': {'image': image4, 'patch_size': patch_size4},
                'description': f'patchify_and_get_fdomain: image shape {image4.shape}, patch_size {patch_size4}',
            })

        if num_tests > 8:
            # Test 9: high_frequency_strength - very small patch
            patch5 = np.random.rand(8, 8).astype(np.float64)
            test_cases.append({
                'function': 'high_frequency_strength',
                'args': {'patch': patch5},
                'description': f'high_frequency_strength: small patch shape {patch5.shape}',
            })

        if num_tests > 9:
            # Test 10: patchify_and_get_fdomain - large image
            image5 = np.random.rand(100, 100, 3).astype(np.float64)
            patch_size5 = (25, 25)
            test_cases.append({
                'function': 'patchify_and_get_fdomain',
                'args': {'image': image5, 'patch_size': patch_size5},
                'description': f'patchify_and_get_fdomain: large image shape {image5.shape}, patch_size {patch_size5}',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            if i % 2 == 0:
                # high_frequency_strength test
                size = 16 + (i % 5) * 4
                patch = np.random.rand(size, size).astype(np.float64)
                test_cases.append({
                    'function': 'high_frequency_strength',
                    'args': {'patch': patch},
                    'description': f'high_frequency_strength: additional test {i+1}, patch shape {patch.shape}',
                })
            else:
                # patchify_and_get_fdomain test
                img_size = 40 + (i % 5) * 10
                patch_s = 8 + (i % 3) * 4
                image = np.random.rand(img_size, img_size, 3).astype(np.float64)
                patch_size = (patch_s, patch_s)
                test_cases.append({
                    'function': 'patchify_and_get_fdomain',
                    'args': {'image': image, 'patch_size': patch_size},
                    'description': f'patchify_and_get_fdomain: additional test {i+1}, image shape {image.shape}, patch_size {patch_size}',
                })

        return test_cases[:num_tests]
