"""
Test Data Generator for get_image_coor_from_world_points2 function.
Generates test cases with different configurations.
"""

import torch


class MockView:
    """Mock camera view object for testing."""

    def __init__(self, height, width, K, w2c, device='cpu'):
        """
        Args:
            height: Image height
            width: Image width
            K: Camera intrinsic matrix, shape (3, 3)
            w2c: World-to-camera transformation matrix, shape (4, 4)
            device: Device to use ('cpu' or 'cuda')
        """
        self.original_image = torch.zeros(3, height, width, device=device)
        self.K = K.to(device)
        self.w2c = w2c.to(device)


class TestDataGenerator:
    """Generate test data for get_image_coor_from_world_points2 function."""

    def __init__(self, seed=42, device='cpu'):
        self.seed = seed
        self.device = device
        torch.manual_seed(seed)

    def generate_random_camera(self, height, width, fov_deg=60):
        """Generate random camera parameters."""
        focal = width / (2 * torch.tan(torch.tensor(fov_deg * 3.14159 / 180 / 2)))
        K = torch.tensor([
            [focal, 0, width / 2],
            [0, focal, height / 2],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)

        angle_x = (torch.rand(1, device=self.device) - 0.5) * 0.5
        angle_y = (torch.rand(1, device=self.device) - 0.5) * 0.5
        angle_z = (torch.rand(1, device=self.device) - 0.5) * 0.5

        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(angle_x), -torch.sin(angle_x)],
            [0, torch.sin(angle_x), torch.cos(angle_x)]
        ], dtype=torch.float32, device=self.device).squeeze()

        Ry = torch.tensor([
            [torch.cos(angle_y), 0, torch.sin(angle_y)],
            [0, 1, 0],
            [-torch.sin(angle_y), 0, torch.cos(angle_y)]
        ], dtype=torch.float32, device=self.device).squeeze()

        Rz = torch.tensor([
            [torch.cos(angle_z), -torch.sin(angle_z), 0],
            [torch.sin(angle_z), torch.cos(angle_z), 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device).squeeze()

        R = Rz @ Ry @ Rx
        t = torch.randn(3, device=self.device) * 2.0
        w2c = torch.eye(4, dtype=torch.float32, device=self.device)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        return K, w2c

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        height, width = 480, 640
        K, w2c = self.generate_random_camera(height, width)
        view = MockView(height, width, K, w2c, device=self.device)
        points = torch.randn(10, 3, device=self.device) * 5.0
        points[:, 2] += 10.0
        test_cases.append({
            'points': points,
            'view': view,
            'mode': None,
            'scale': 1,
            'description': f'Basic: {points.shape[0]} points, mode=None, scale=1',
        })

        if num_tests > 1:
            height, width = 480, 640
            K, w2c = self.generate_random_camera(height, width)
            view = MockView(height, width, K, w2c, device=self.device)
            points = torch.randn(20, 3, device=self.device) * 10.0
            points[:, 2] += 15.0
            test_cases.append({
                'points': points,
                'view': view,
                'mode': 'scale',
                'scale': 1,
                'description': f'Scale mode: {points.shape[0]} points, mode="scale", scale=1',
            })

        if num_tests > 2:
            height, width = 720, 1280
            K, w2c = self.generate_random_camera(height, width)
            view = MockView(height, width, K, w2c, device=self.device)
            points = torch.randn(50, 3, device=self.device) * 8.0
            points[:, 2] += 12.0
            test_cases.append({
                'points': points,
                'view': view,
                'mode': None,
                'scale': 0.5,
                'description': f'Scaled: {points.shape[0]} points, mode=None, scale=0.5',
            })

        if num_tests > 3:
            height, width = 480, 640
            K, w2c = self.generate_random_camera(height, width)
            view = MockView(height, width, K, w2c, device=self.device)
            points = torch.randn(100, 3, device=self.device) * 15.0
            points[:, 2] += 20.0
            test_cases.append({
                'points': points,
                'view': view,
                'mode': 'scale',
                'scale': 1,
                'description': f'Many points: {points.shape[0]} points, mode="scale"',
            })

        if num_tests > 4:
            height, width = 1080, 1920
            K, w2c = self.generate_random_camera(height, width)
            view = MockView(height, width, K, w2c, device=self.device)
            points = torch.randn(80, 3, device=self.device) * 12.0
            points[:, 2] += 18.0
            test_cases.append({
                'points': points,
                'view': view,
                'mode': None,
                'scale': 2.0,
                'description': f'Large image: {height}x{width}, {points.shape[0]} points, scale=2.0',
            })

        if num_tests > 5:
            height, width = 480, 640
            K, w2c = self.generate_random_camera(height, width)
            view = MockView(height, width, K, w2c, device=self.device)
            points = torch.randn(60, 3, device=self.device) * 25.0
            points[:, 2] += 15.0
            test_cases.append({
                'points': points,
                'view': view,
                'mode': 'scale',
                'scale': 1,
                'description': f'Outside view: {points.shape[0]} points with some outside FOV',
            })

        if num_tests > 6:
            height, width = 240, 320
            K, w2c = self.generate_random_camera(height, width)
            view = MockView(height, width, K, w2c, device=self.device)
            points = torch.randn(3, 3, device=self.device) * 3.0
            points[:, 2] += 8.0
            test_cases.append({
                'points': points,
                'view': view,
                'mode': None,
                'scale': 1,
                'description': f'Few points: {points.shape[0]} points, small image',
            })

        if num_tests > 7:
            height, width = 480, 640
            K, w2c = self.generate_random_camera(height, width, fov_deg=90)
            view = MockView(height, width, K, w2c, device=self.device)
            points = torch.randn(70, 3, device=self.device) * 10.0
            points[:, 2] += 15.0
            test_cases.append({
                'points': points,
                'view': view,
                'mode': 'scale',
                'scale': 1,
                'description': f'Wide FOV: 90 degrees, {points.shape[0]} points',
            })

        if num_tests > 8:
            height, width = 640, 480
            K, w2c = self.generate_random_camera(height, width)
            view = MockView(height, width, K, w2c, device=self.device)
            points = torch.randn(40, 3, device=self.device) * 8.0
            points[:, 2] += 10.0
            test_cases.append({
                'points': points,
                'view': view,
                'mode': None,
                'scale': 1.5,
                'description': f'Portrait: {height}x{width}, scale=1.5',
            })

        if num_tests > 9:
            height, width = 480, 640
            K, w2c = self.generate_random_camera(height, width)
            view = MockView(height, width, K, w2c, device=self.device)
            points = torch.randn(30, 3, device=self.device) * 2.0
            points[:, 2] += 3.0
            test_cases.append({
                'points': points,
                'view': view,
                'mode': 'scale',
                'scale': 1,
                'description': f'Close points: {points.shape[0]} points close to camera',
            })

        for i in range(num_tests - len(test_cases)):
            height = 480 + (i % 3) * 120
            width = 640 + (i % 3) * 160
            K, w2c = self.generate_random_camera(height, width)
            view = MockView(height, width, K, w2c, device=self.device)
            n_points = 20 + i * 10
            points = torch.randn(n_points, 3, device=self.device) * (5 + i * 2)
            points[:, 2] += 10.0 + i * 2

            test_cases.append({
                'points': points,
                'view': view,
                'mode': 'scale' if i % 2 == 0 else None,
                'scale': 1 + i * 0.2,
                'description': f'Additional test {i+1}: {n_points} points, {height}x{width}',
            })

        return test_cases[:num_tests]
