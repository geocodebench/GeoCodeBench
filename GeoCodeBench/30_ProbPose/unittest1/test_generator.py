"""
Test Data Generator for generate_probmaps() function.
Generates test cases with different heatmap sizes, keypoints, and sigma.
"""

import numpy as np


class TestDataGenerator:
    """Generate test data for generate_probmaps() function."""

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small heatmap, single instance, few keypoints
        N, K, D = 1, 5, 2
        W, H = 32, 32
        keypoints = np.random.rand(N, K, D) * min(W, H)
        keypoints_visible = np.ones((N, K), dtype=np.float32)
        sigma = 0.55
        test_cases.append({
            'heatmap_size': (W, H),
            'keypoints': keypoints.astype(np.float32),
            'keypoints_visible': keypoints_visible,
            'sigma': sigma,
            'description': f'Basic: N={N}, K={K}, size={W}x{H}, sigma={sigma}',
        })

        if num_tests > 1:
            # Test 2: Multiple instances
            N, K, D = 2, 5, 2
            W, H = 48, 48
            keypoints = np.random.rand(N, K, D) * min(W, H)
            keypoints_visible = np.ones((N, K), dtype=np.float32)
            sigma = 0.55
            test_cases.append({
                'heatmap_size': (W, H),
                'keypoints': keypoints.astype(np.float32),
                'keypoints_visible': keypoints_visible,
                'sigma': sigma,
                'description': f'Multiple instances: N={N}, K={K}, size={W}x{H}',
            })

        if num_tests > 2:
            # Test 3: Larger heatmap with more keypoints
            N, K, D = 1, 10, 2
            W, H = 64, 64
            keypoints = np.random.rand(N, K, D) * min(W, H)
            keypoints_visible = np.ones((N, K), dtype=np.float32)
            sigma = 0.55
            test_cases.append({
                'heatmap_size': (W, H),
                'keypoints': keypoints.astype(np.float32),
                'keypoints_visible': keypoints_visible,
                'sigma': sigma,
                'description': f'More keypoints: N={N}, K={K}, size={W}x{H}',
            })

        if num_tests > 3:
            # Test 4: Some invisible keypoints
            N, K, D = 1, 8, 2
            W, H = 48, 48
            keypoints = np.random.rand(N, K, D) * min(W, H)
            keypoints_visible = np.array([[1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]], dtype=np.float32)
            sigma = 0.55
            test_cases.append({
                'heatmap_size': (W, H),
                'keypoints': keypoints.astype(np.float32),
                'keypoints_visible': keypoints_visible,
                'sigma': sigma,
                'description': f'Partial visibility: N={N}, K={K}, some invisible',
            })

        if num_tests > 4:
            # Test 5: Different sigma value
            N, K, D = 1, 6, 2
            W, H = 56, 56
            keypoints = np.random.rand(N, K, D) * min(W, H)
            keypoints_visible = np.ones((N, K), dtype=np.float32)
            sigma = 0.75
            test_cases.append({
                'heatmap_size': (W, H),
                'keypoints': keypoints.astype(np.float32),
                'keypoints_visible': keypoints_visible,
                'sigma': sigma,
                'description': f'Different sigma: N={N}, K={K}, sigma={sigma}',
            })

        if num_tests > 5:
            # Test 6: Non-square heatmap
            N, K, D = 1, 7, 2
            W, H = 80, 60
            keypoints = np.random.rand(N, K, D)
            keypoints[:, :, 0] *= W
            keypoints[:, :, 1] *= H
            keypoints_visible = np.ones((N, K), dtype=np.float32)
            sigma = 0.55
            test_cases.append({
                'heatmap_size': (W, H),
                'keypoints': keypoints.astype(np.float32),
                'keypoints_visible': keypoints_visible,
                'sigma': sigma,
                'description': f'Non-square: N={N}, K={K}, size={W}x{H}',
            })

        if num_tests > 6:
            # Test 7: Edge case - keypoint at corner
            N, K, D = 1, 4, 2
            W, H = 40, 40
            keypoints = np.array([[[0.0, 0.0], [W-1, 0.0], [0.0, H-1], [W-1, H-1]]], dtype=np.float32)
            keypoints_visible = np.ones((N, K), dtype=np.float32)
            sigma = 0.55
            test_cases.append({
                'heatmap_size': (W, H),
                'keypoints': keypoints,
                'keypoints_visible': keypoints_visible,
                'sigma': sigma,
                'description': f'Corner keypoints: N={N}, K={K}',
            })

        if num_tests > 7:
            # Test 8: Larger batch with mixed visibility
            N, K, D = 3, 6, 2
            W, H = 48, 48
            keypoints = np.random.rand(N, K, D) * min(W, H)
            keypoints_visible = np.random.choice([0.0, 1.0], size=(N, K)).astype(np.float32)
            sigma = 0.55
            test_cases.append({
                'heatmap_size': (W, H),
                'keypoints': keypoints.astype(np.float32),
                'keypoints_visible': keypoints_visible,
                'sigma': sigma,
                'description': f'Large batch: N={N}, K={K}, mixed visibility',
            })

        if num_tests > 8:
            # Test 9: Very small sigma
            N, K, D = 1, 5, 2
            W, H = 32, 32
            keypoints = np.random.rand(N, K, D) * min(W, H)
            keypoints_visible = np.ones((N, K), dtype=np.float32)
            sigma = 0.25
            test_cases.append({
                'heatmap_size': (W, H),
                'keypoints': keypoints.astype(np.float32),
                'keypoints_visible': keypoints_visible,
                'sigma': sigma,
                'description': f'Small sigma: N={N}, K={K}, sigma={sigma}',
            })

        if num_tests > 9:
            # Test 10: Keypoints near center
            N, K, D = 1, 5, 2
            W, H = 64, 64
            keypoints = np.random.rand(N, K, D) * 10 + (min(W, H) / 2 - 5)
            keypoints_visible = np.ones((N, K), dtype=np.float32)
            sigma = 0.55
            test_cases.append({
                'heatmap_size': (W, H),
                'keypoints': keypoints.astype(np.float32),
                'keypoints_visible': keypoints_visible,
                'sigma': sigma,
                'description': f'Center keypoints: N={N}, K={K}',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            N = 1 + (i % 3)
            K = 5 + (i % 8)
            D = 2
            W = 32 + (i % 4) * 16
            H = 32 + ((i + 1) % 4) * 16
            keypoints = np.random.rand(N, K, D)
            keypoints[:, :, 0] *= W
            keypoints[:, :, 1] *= H
            keypoints_visible = np.ones((N, K), dtype=np.float32)
            if i % 3 == 0:
                keypoints_visible[:, ::2] = 0.0
            sigma = 0.4 + (i % 5) * 0.1

            test_cases.append({
                'heatmap_size': (W, H),
                'keypoints': keypoints.astype(np.float32),
                'keypoints_visible': keypoints_visible,
                'sigma': sigma,
                'description': f'Additional test {i+1}: N={N}, K={K}, size={W}x{H}',
            })

        return test_cases[:num_tests]
