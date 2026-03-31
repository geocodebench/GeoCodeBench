"""
Test Data Generator for _get_subpixel_maximums() function.
Generates test cases with heatmaps and integer locations.
"""

import numpy as np


class TestDataGenerator:
    """Generate test data for _get_subpixel_maximums() function."""

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)

    def generate_gaussian_heatmap(self, H, W, center_x, center_y, sigma=2.0):
        """Generate a Gaussian heatmap centered at (center_x, center_y)."""
        y, x = np.ogrid[:H, :W]
        heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        return heatmap.astype(np.float32)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small heatmap, single location
        N, H, W = 1, 32, 32
        heatmaps = np.zeros((N, H, W), dtype=np.float32)
        locs = np.zeros((N, 2), dtype=np.float32)
        center_x, center_y = 15.5, 16.2  # Sub-pixel center
        heatmaps[0] = self.generate_gaussian_heatmap(H, W, center_x, center_y, sigma=2.0)
        locs[0] = [15, 16]  # Integer location near peak
        test_cases.append({
            'heatmaps': heatmaps,
            'locs': locs,
            'description': f'Basic: N={N}, H={H}, W={W}, single location',
        })

        if num_tests > 1:
            # Test 2: Multiple locations
            N, H, W = 3, 48, 48
            heatmaps = np.zeros((N, H, W), dtype=np.float32)
            locs = np.zeros((N, 2), dtype=np.float32)
            centers = [(20.3, 22.7), (30.1, 15.8), (10.9, 35.2)]
            for i, (cx, cy) in enumerate(centers):
                heatmaps[i] = self.generate_gaussian_heatmap(H, W, cx, cy, sigma=2.5)
                locs[i] = [int(cx), int(cy)]
            test_cases.append({
                'heatmaps': heatmaps,
                'locs': locs,
                'description': f'Multiple locations: N={N}, H={H}, W={W}',
            })

        if num_tests > 2:
            # Test 3: Larger heatmap with more locations
            N, H, W = 5, 64, 64
            heatmaps = np.zeros((N, H, W), dtype=np.float32)
            locs = np.zeros((N, 2), dtype=np.float32)
            for i in range(N):
                cx = np.random.uniform(5, W - 5)
                cy = np.random.uniform(5, H - 5)
                heatmaps[i] = self.generate_gaussian_heatmap(H, W, cx, cy, sigma=3.0)
                locs[i] = [int(cx), int(cy)]
            test_cases.append({
                'heatmaps': heatmaps,
                'locs': locs,
                'description': f'Larger heatmap: N={N}, H={H}, W={W}',
            })

        if num_tests > 3:
            # Test 4: Some locations near boundaries (should be handled)
            N, H, W = 4, 32, 32
            heatmaps = np.zeros((N, H, W), dtype=np.float32)
            locs = np.zeros((N, 2), dtype=np.float32)
            # Mix of boundary and interior locations
            test_locs = [(1, 1), (1, 15), (15, 1), (15, 15)]
            for i, (x, y) in enumerate(test_locs):
                cx, cy = x + 0.3, y + 0.4
                heatmaps[i] = self.generate_gaussian_heatmap(H, W, cx, cy, sigma=2.0)
                locs[i] = [x, y]
            test_cases.append({
                'heatmaps': heatmaps,
                'locs': locs,
                'description': f'Boundary cases: N={N}, some near boundaries',
            })

        if num_tests > 4:
            # Test 5: Very small heatmap
            N, H, W = 2, 16, 16
            heatmaps = np.zeros((N, H, W), dtype=np.float32)
            locs = np.zeros((N, 2), dtype=np.float32)
            centers = [(8.2, 7.5), (10.8, 9.3)]
            for i, (cx, cy) in enumerate(centers):
                heatmaps[i] = self.generate_gaussian_heatmap(H, W, cx, cy, sigma=1.5)
                locs[i] = [int(cx), int(cy)]
            test_cases.append({
                'heatmaps': heatmaps,
                'locs': locs,
                'description': f'Small heatmap: N={N}, H={H}, W={W}',
            })

        if num_tests > 5:
            # Test 6: Large heatmap
            N, H, W = 3, 128, 128
            heatmaps = np.zeros((N, H, W), dtype=np.float32)
            locs = np.zeros((N, 2), dtype=np.float32)
            for i in range(N):
                cx = np.random.uniform(10, W - 10)
                cy = np.random.uniform(10, H - 10)
                heatmaps[i] = self.generate_gaussian_heatmap(H, W, cx, cy, sigma=4.0)
                locs[i] = [int(cx), int(cy)]
            test_cases.append({
                'heatmaps': heatmaps,
                'locs': locs,
                'description': f'Large heatmap: N={N}, H={H}, W={W}',
            })

        if num_tests > 6:
            # Test 7: Flat heatmap (low gradient)
            N, H, W = 2, 32, 32
            heatmaps = np.ones((N, H, W), dtype=np.float32) * 0.5
            locs = np.array([[15, 15], [20, 20]], dtype=np.float32)
            test_cases.append({
                'heatmaps': heatmaps,
                'locs': locs,
                'description': f'Flat heatmap: N={N}, low gradient',
            })

        if num_tests > 7:
            # Test 8: Sharp peaks (high gradient)
            N, H, W = 2, 32, 32
            heatmaps = np.zeros((N, H, W), dtype=np.float32)
            locs = np.zeros((N, 2), dtype=np.float32)
            centers = [(15.1, 16.2), (20.8, 18.5)]
            for i, (cx, cy) in enumerate(centers):
                heatmaps[i] = self.generate_gaussian_heatmap(H, W, cx, cy, sigma=0.8)
                locs[i] = [int(cx), int(cy)]
            test_cases.append({
                'heatmaps': heatmaps,
                'locs': locs,
                'description': f'Sharp peaks: N={N}, high gradient',
            })

        if num_tests > 8:
            # Test 9: Multiple peaks in same heatmap
            N, H, W = 1, 48, 48
            heatmaps = np.zeros((N, H, W), dtype=np.float32)
            locs = np.zeros((N, 2), dtype=np.float32)
            # Create heatmap with multiple peaks, but test only one location
            heatmaps[0] = self.generate_gaussian_heatmap(H, W, 20.3, 22.7, sigma=2.0)
            heatmaps[0] += 0.5 * self.generate_gaussian_heatmap(H, W, 30.1, 15.8, sigma=2.0)
            locs[0] = [20, 22]
            test_cases.append({
                'heatmaps': heatmaps,
                'locs': locs,
                'description': f'Multiple peaks: N={N}, complex heatmap',
            })

        if num_tests > 9:
            # Test 10: Random realistic case
            N, H, W = 4, 64, 64
            heatmaps = np.zeros((N, H, W), dtype=np.float32)
            locs = np.zeros((N, 2), dtype=np.float32)
            for i in range(N):
                cx = np.random.uniform(5, W - 5)
                cy = np.random.uniform(5, H - 5)
                sigma = np.random.uniform(1.5, 3.5)
                heatmaps[i] = self.generate_gaussian_heatmap(H, W, cx, cy, sigma=sigma)
                locs[i] = [int(cx), int(cy)]
            test_cases.append({
                'heatmaps': heatmaps,
                'locs': locs,
                'description': f'Random realistic: N={N}, varied parameters',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            N = np.random.randint(1, 6)
            H = np.random.randint(16, 65)
            W = np.random.randint(16, 65)
            heatmaps = np.zeros((N, H, W), dtype=np.float32)
            locs = np.zeros((N, 2), dtype=np.float32)
            for j in range(N):
                cx = np.random.uniform(5, W - 5)
                cy = np.random.uniform(5, H - 5)
                sigma = np.random.uniform(1.5, 3.0)
                heatmaps[j] = self.generate_gaussian_heatmap(H, W, cx, cy, sigma=sigma)
                locs[j] = [int(cx), int(cy)]

            test_cases.append({
                'heatmaps': heatmaps,
                'locs': locs,
                'description': f'Additional test {i+1}: N={N}, H={H}, W={W}',
            })

        return test_cases[:num_tests]
