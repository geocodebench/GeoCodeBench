"""
Test Data Generator for depth_to_point_cloud function.
"""

import numpy as np


class TestDataGenerator:
    """Generate test data for depth_to_point_cloud function."""

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Small image with simple mask
        H, W = 10, 10
        depth_map = np.random.uniform(1.0, 10.0, (H, W))
        intrinsic_matrix = np.array([
            [500.0, 0.0, 5.0],
            [0.0, 500.0, 5.0],
            [0.0, 0.0, 1.0]
        ])
        c2w = np.eye(4)  # Identity transformation
        mask = np.ones((H, W), dtype=np.float32)
        rgb_map = np.random.uniform(0, 255, (H, W, 3)).astype(np.uint8)

        test_cases.append({
            'depth_map': depth_map,
            'intrinsic_matrix': intrinsic_matrix,
            'c2w': c2w,
            'mask': mask,
            'rgb_map': rgb_map,
            'description': f'Small image: {H}x{W}, full mask, identity transform',
        })

        if num_tests > 1:
            # Test 2: Medium image with partial mask
            H, W = 20, 30
            depth_map = np.random.uniform(0.5, 15.0, (H, W))
            intrinsic_matrix = np.array([
                [800.0, 0.0, 15.0],
                [0.0, 800.0, 10.0],
                [0.0, 0.0, 1.0]
            ])
            # Random rotation and translation
            theta = np.pi / 6
            c2w = np.array([
                [np.cos(theta), -np.sin(theta), 0, 1.0],
                [np.sin(theta), np.cos(theta), 0, 2.0],
                [0, 0, 1, 0.5],
                [0, 0, 0, 1]
            ])
            mask = np.random.choice([0, 1], size=(H, W), p=[0.3, 0.7])
            rgb_map = np.random.uniform(0, 255, (H, W, 3)).astype(np.uint8)

            test_cases.append({
                'depth_map': depth_map,
                'intrinsic_matrix': intrinsic_matrix,
                'c2w': c2w,
                'mask': mask,
                'rgb_map': rgb_map,
                'description': f'Medium image: {H}x{W}, partial mask (70%), rotation transform',
            })

        if num_tests > 2:
            # Test 3: Different intrinsics
            H, W = 15, 25
            depth_map = np.random.uniform(2.0, 8.0, (H, W))
            intrinsic_matrix = np.array([
                [600.0, 0.0, 12.5],
                [0.0, 650.0, 7.5],
                [0.0, 0.0, 1.0]
            ])
            # 3D rotation
            angle = np.pi / 4
            c2w = np.array([
                [1, 0, 0, 3.0],
                [0, np.cos(angle), -np.sin(angle), 1.5],
                [0, np.sin(angle), np.cos(angle), 2.0],
                [0, 0, 0, 1]
            ])
            mask = np.random.choice([0, 1], size=(H, W), p=[0.5, 0.5])
            rgb_map = np.random.uniform(0, 255, (H, W, 3)).astype(np.uint8)

            test_cases.append({
                'depth_map': depth_map,
                'intrinsic_matrix': intrinsic_matrix,
                'c2w': c2w,
                'mask': mask,
                'rgb_map': rgb_map,
                'description': f'Different intrinsics: {H}x{W}, 50% mask, 3D rotation',
            })

        if num_tests > 3:
            # Test 4: Sparse mask
            H, W = 25, 20
            depth_map = np.random.uniform(1.0, 20.0, (H, W))
            intrinsic_matrix = np.array([
                [750.0, 0.0, 10.0],
                [0.0, 750.0, 12.5],
                [0.0, 0.0, 1.0]
            ])
            # Complex transformation
            theta_x, theta_y, theta_z = np.pi / 8, np.pi / 6, np.pi / 12
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(theta_x), -np.sin(theta_x)],
                          [0, np.sin(theta_x), np.cos(theta_x)]])
            Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                          [0, 1, 0],
                          [-np.sin(theta_y), 0, np.cos(theta_y)]])
            Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                          [np.sin(theta_z), np.cos(theta_z), 0],
                          [0, 0, 1]])
            R = Rz @ Ry @ Rx
            t = np.array([[2.5], [1.5], [3.0]])
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3:4] = t

            mask = np.random.choice([0, 1], size=(H, W), p=[0.8, 0.2])
            rgb_map = np.random.uniform(0, 255, (H, W, 3)).astype(np.uint8)

            test_cases.append({
                'depth_map': depth_map,
                'intrinsic_matrix': intrinsic_matrix,
                'c2w': c2w,
                'mask': mask,
                'rgb_map': rgb_map,
                'description': f'Sparse mask: {H}x{W}, 20% mask, complex transform',
            })

        if num_tests > 4:
            # Test 5: Large image with centered mask
            H, W = 40, 60
            depth_map = np.random.uniform(0.1, 50.0, (H, W))
            intrinsic_matrix = np.array([
                [1000.0, 0.0, 30.0],
                [0.0, 1000.0, 20.0],
                [0.0, 0.0, 1.0]
            ])
            c2w = np.array([
                [0.8, -0.6, 0, 5.0],
                [0.6, 0.8, 0, 3.0],
                [0, 0, 1, 1.5],
                [0, 0, 0, 1]
            ])
            # Create a centered circular mask
            mask = np.zeros((H, W), dtype=np.float32)
            center_h, center_w = H // 2, W // 2
            for i in range(H):
                for j in range(W):
                    if (i - center_h)**2 + (j - center_w)**2 < (min(H, W) // 3)**2:
                        mask[i, j] = 1
            rgb_map = np.random.uniform(0, 255, (H, W, 3)).astype(np.uint8)

            test_cases.append({
                'depth_map': depth_map,
                'intrinsic_matrix': intrinsic_matrix,
                'c2w': c2w,
                'mask': mask,
                'rgb_map': rgb_map,
                'description': f'Large image: {H}x{W}, circular mask, rotation',
            })

        if num_tests > 5:
            # Test 6: Edge case - very few valid pixels
            H, W = 30, 30
            depth_map = np.random.uniform(1.0, 5.0, (H, W))
            intrinsic_matrix = np.array([
                [550.0, 0.0, 15.0],
                [0.0, 550.0, 15.0],
                [0.0, 0.0, 1.0]
            ])
            c2w = np.eye(4)
            c2w[:3, 3] = [1, 2, 3]
            mask = np.zeros((H, W), dtype=np.float32)
            mask[10:15, 10:15] = 1  # Small valid region
            rgb_map = np.random.uniform(0, 255, (H, W, 3)).astype(np.uint8)

            test_cases.append({
                'depth_map': depth_map,
                'intrinsic_matrix': intrinsic_matrix,
                'c2w': c2w,
                'mask': mask,
                'rgb_map': rgb_map,
                'description': f'Few pixels: {H}x{W}, small valid region, translation',
            })

        if num_tests > 6:
            # Test 7: Non-square image
            H, W = 50, 30
            depth_map = np.random.uniform(2.0, 12.0, (H, W))
            intrinsic_matrix = np.array([
                [900.0, 0.0, 15.0],
                [0.0, 850.0, 25.0],
                [0.0, 0.0, 1.0]
            ])
            angle = np.pi / 3
            c2w = np.array([
                [np.cos(angle), 0, np.sin(angle), 2.0],
                [0, 1, 0, 1.0],
                [-np.sin(angle), 0, np.cos(angle), 4.0],
                [0, 0, 0, 1]
            ])
            mask = np.random.choice([0, 1], size=(H, W), p=[0.4, 0.6])
            rgb_map = np.random.uniform(0, 255, (H, W, 3)).astype(np.uint8)

            test_cases.append({
                'depth_map': depth_map,
                'intrinsic_matrix': intrinsic_matrix,
                'c2w': c2w,
                'mask': mask,
                'rgb_map': rgb_map,
                'description': f'Non-square: {H}x{W}, 60% mask, Y-axis rotation',
            })

        if num_tests > 7:
            # Test 8: Different principal point
            H, W = 35, 35
            depth_map = np.random.uniform(0.5, 25.0, (H, W))
            intrinsic_matrix = np.array([
                [700.0, 0.0, 25.0],  # Off-center principal point
                [0.0, 700.0, 10.0],
                [0.0, 0.0, 1.0]
            ])
            c2w = np.array([
                [0.5, -0.866, 0, 4.0],
                [0.866, 0.5, 0, 2.0],
                [0, 0, 1, 0.5],
                [0, 0, 0, 1]
            ])
            mask = np.random.choice([0, 1], size=(H, W), p=[0.35, 0.65])
            rgb_map = np.random.uniform(0, 255, (H, W, 3)).astype(np.uint8)

            test_cases.append({
                'depth_map': depth_map,
                'intrinsic_matrix': intrinsic_matrix,
                'c2w': c2w,
                'mask': mask,
                'rgb_map': rgb_map,
                'description': f'Off-center principal point: {H}x{W}, 65% mask',
            })

        if num_tests > 8:
            # Test 9: Large depth variation
            H, W = 28, 42
            depth_map = np.random.uniform(0.1, 100.0, (H, W))  # Large range
            intrinsic_matrix = np.array([
                [650.0, 0.0, 21.0],
                [0.0, 680.0, 14.0],
                [0.0, 0.0, 1.0]
            ])
            theta = np.pi / 5
            c2w = np.array([
                [np.cos(theta), -np.sin(theta), 0, 6.0],
                [np.sin(theta), np.cos(theta), 0, 4.0],
                [0, 0, 1, 2.5],
                [0, 0, 0, 1]
            ])
            mask = np.random.choice([0, 1], size=(H, W), p=[0.25, 0.75])
            rgb_map = np.random.uniform(0, 255, (H, W, 3)).astype(np.uint8)

            test_cases.append({
                'depth_map': depth_map,
                'intrinsic_matrix': intrinsic_matrix,
                'c2w': c2w,
                'mask': mask,
                'rgb_map': rgb_map,
                'description': f'Large depth range: {H}x{W}, depth 0.1-100, 75% mask',
            })

        if num_tests > 9:
            # Test 10: Mixed - realistic scenario
            H, W = 48, 64
            depth_map = np.random.uniform(1.0, 30.0, (H, W))
            intrinsic_matrix = np.array([
                [1200.0, 0.0, 32.0],
                [0.0, 1200.0, 24.0],
                [0.0, 0.0, 1.0]
            ])
            # Complex rotation matrix
            alpha, beta, gamma = np.pi / 10, np.pi / 8, np.pi / 12
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(alpha), -np.sin(alpha)],
                          [0, np.sin(alpha), np.cos(alpha)]])
            Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                          [0, 1, 0],
                          [-np.sin(beta), 0, np.cos(beta)]])
            Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                          [np.sin(gamma), np.cos(gamma), 0],
                          [0, 0, 1]])
            R = Rz @ Ry @ Rx
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = [3.5, 2.5, 5.0]
            mask = np.random.choice([0, 1], size=(H, W), p=[0.45, 0.55])
            rgb_map = np.random.uniform(0, 255, (H, W, 3)).astype(np.uint8)

            test_cases.append({
                'depth_map': depth_map,
                'intrinsic_matrix': intrinsic_matrix,
                'c2w': c2w,
                'mask': mask,
                'rgb_map': rgb_map,
                'description': f'Realistic: {H}x{W}, 55% mask, full 3D rotation+translation',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            H, W = 20 + i * 5, 25 + i * 5
            depth_map = np.random.uniform(1.0, 15.0, (H, W))
            intrinsic_matrix = np.array([
                [500.0 + i * 50, 0.0, W / 2],
                [0.0, 500.0 + i * 50, H / 2],
                [0.0, 0.0, 1.0]
            ])
            angle = (i + 1) * np.pi / 10
            c2w = np.array([
                [np.cos(angle), -np.sin(angle), 0, float(i)],
                [np.sin(angle), np.cos(angle), 0, float(i + 1)],
                [0, 0, 1, 1.0],
                [0, 0, 0, 1]
            ])
            mask = np.random.choice([0, 1], size=(H, W), p=[0.3 + i * 0.05, 0.7 - i * 0.05])
            rgb_map = np.random.uniform(0, 255, (H, W, 3)).astype(np.uint8)

            test_cases.append({
                'depth_map': depth_map,
                'intrinsic_matrix': intrinsic_matrix,
                'c2w': c2w,
                'mask': mask,
                'rgb_map': rgb_map,
                'description': f'Additional test {i+1}: {H}x{W}',
            })

        return test_cases[:num_tests]
