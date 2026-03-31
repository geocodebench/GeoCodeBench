"""
Test Data Generator for compute_epipolar_distance function.
Generates test cases with different configurations.
"""

import numpy as np


class TestDataGenerator:
    """Generate test data for compute_epipolar_distance function."""

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with 5 point pairs
        num_points = 5
        T_21 = self._generate_random_transform()
        K = self._generate_camera_matrix()
        p_1, p_2 = self._generate_corresponding_points(num_points, T_21, K)
        test_cases.append({
            'T_21': T_21,
            'K': K,
            'p_1': p_1,
            'p_2': p_2,
            'description': f'Basic: {num_points} point pairs',
        })

        if num_tests > 1:
            # Test 2: More points
            num_points = 10
            T_21 = self._generate_random_transform()
            K = self._generate_camera_matrix()
            p_1, p_2 = self._generate_corresponding_points(num_points, T_21, K)
            test_cases.append({
                'T_21': T_21,
                'K': K,
                'p_1': p_1,
                'p_2': p_2,
                'description': f'More points: {num_points} point pairs',
            })

        if num_tests > 2:
            # Test 3: Identity transformation
            num_points = 8
            T_21 = np.eye(4)
            K = self._generate_camera_matrix()
            p_1, p_2 = self._generate_corresponding_points(num_points, T_21, K)
            test_cases.append({
                'T_21': T_21,
                'K': K,
                'p_1': p_1,
                'p_2': p_2,
                'description': f'Identity transform: {num_points} point pairs',
            })

        if num_tests > 3:
            # Test 4: Large rotation
            num_points = 6
            T_21 = self._generate_large_rotation_transform()
            K = self._generate_camera_matrix()
            p_1, p_2 = self._generate_corresponding_points(num_points, T_21, K)
            test_cases.append({
                'T_21': T_21,
                'K': K,
                'p_1': p_1,
                'p_2': p_2,
                'description': f'Large rotation: {num_points} point pairs',
            })

        if num_tests > 4:
            # Test 5: Many points
            num_points = 20
            T_21 = self._generate_random_transform()
            K = self._generate_camera_matrix()
            p_1, p_2 = self._generate_corresponding_points(num_points, T_21, K)
            test_cases.append({
                'T_21': T_21,
                'K': K,
                'p_1': p_1,
                'p_2': p_2,
                'description': f'Many points: {num_points} point pairs',
            })

        if num_tests > 5:
            # Test 6: Pure translation
            num_points = 7
            T_21 = self._generate_translation_only_transform()
            K = self._generate_camera_matrix()
            p_1, p_2 = self._generate_corresponding_points(num_points, T_21, K)
            test_cases.append({
                'T_21': T_21,
                'K': K,
                'p_1': p_1,
                'p_2': p_2,
                'description': f'Pure translation: {num_points} point pairs',
            })

        if num_tests > 6:
            # Test 7: Different camera parameters
            num_points = 9
            T_21 = self._generate_random_transform()
            K = self._generate_different_camera_matrix()
            p_1, p_2 = self._generate_corresponding_points(num_points, T_21, K)
            test_cases.append({
                'T_21': T_21,
                'K': K,
                'p_1': p_1,
                'p_2': p_2,
                'description': f'Different camera: {num_points} point pairs',
            })

        if num_tests > 7:
            # Test 8: Edge case - single point
            num_points = 1
            T_21 = self._generate_random_transform()
            K = self._generate_camera_matrix()
            p_1, p_2 = self._generate_corresponding_points(num_points, T_21, K)
            test_cases.append({
                'T_21': T_21,
                'K': K,
                'p_1': p_1,
                'p_2': p_2,
                'description': f'Single point: {num_points} point pair',
            })

        if num_tests > 8:
            # Test 9: Extreme camera parameters
            num_points = 12
            T_21 = self._generate_random_transform()
            K = self._generate_extreme_camera_matrix()
            p_1, p_2 = self._generate_corresponding_points(num_points, T_21, K)
            test_cases.append({
                'T_21': T_21,
                'K': K,
                'p_1': p_1,
                'p_2': p_2,
                'description': f'Extreme camera: {num_points} point pairs',
            })

        if num_tests > 9:
            # Test 10: Random configuration
            num_points = 15
            T_21 = self._generate_random_transform()
            K = self._generate_camera_matrix()
            p_1, p_2 = self._generate_corresponding_points(num_points, T_21, K)
            test_cases.append({
                'T_21': T_21,
                'K': K,
                'p_1': p_1,
                'p_2': p_2,
                'description': f'Random config: {num_points} point pairs',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            num_points = 5 + i * 2
            T_21 = self._generate_random_transform()
            K = self._generate_camera_matrix()
            p_1, p_2 = self._generate_corresponding_points(num_points, T_21, K)

            test_cases.append({
                'T_21': T_21,
                'K': K,
                'p_1': p_1,
                'p_2': p_2,
                'description': f'Additional test {i+1}: {num_points} point pairs',
            })

        return test_cases[:num_tests]

    def _generate_random_transform(self):
        """Generate a random 4x4 transformation matrix."""
        # Random rotation matrix
        angle = np.random.uniform(0, 2 * np.pi)
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)

        # Rodrigues' rotation formula
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

        # Random translation
        t = np.random.randn(3) * 0.5

        # Combine into 4x4 matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    def _generate_large_rotation_transform(self):
        """Generate a transformation with large rotation."""
        # Large rotation around z-axis
        angle = np.pi / 2  # 90 degrees
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])

        t = np.random.randn(3) * 0.3

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    def _generate_translation_only_transform(self):
        """Generate a pure translation transformation."""
        T = np.eye(4)
        T[:3, 3] = np.random.randn(3) * 0.5
        return T

    def _generate_camera_matrix(self):
        """Generate a standard camera intrinsic matrix."""
        fx = fy = 500.0
        cx = cy = 320.0
        return np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])

    def _generate_different_camera_matrix(self):
        """Generate a different camera intrinsic matrix."""
        fx = 800.0
        fy = 600.0
        cx = 400.0
        cy = 300.0
        return np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])

    def _generate_extreme_camera_matrix(self):
        """Generate extreme camera parameters."""
        fx = fy = 100.0  # Very small focal length
        cx = cy = 50.0
        return np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])

    def _generate_corresponding_points(self, num_points, T_21, K):
        """Generate corresponding points in two camera views."""
        # Generate random 3D points
        points_3d = np.random.randn(3, num_points) * 2.0
        points_3d[2, :] = np.abs(points_3d[2, :]) + 1.0  # Ensure positive depth

        # Project to camera 1
        p_1_homogeneous = K @ points_3d
        p_1 = p_1_homogeneous / p_1_homogeneous[2, :]

        # Transform to camera 2
        points_3d_homogeneous = np.vstack([points_3d, np.ones((1, num_points))])
        points_3d_2 = T_21 @ points_3d_homogeneous
        points_3d_2 = points_3d_2[:3, :]

        # Project to camera 2
        p_2_homogeneous = K @ points_3d_2
        p_2 = p_2_homogeneous / p_2_homogeneous[2, :]

        return p_1, p_2
