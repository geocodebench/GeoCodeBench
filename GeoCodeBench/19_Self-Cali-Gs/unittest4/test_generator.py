"""Generate test data for apply_flow_up_down_left_right function."""

import torch


class MockViewpointCam:
    """Mock camera object for testing."""

    def __init__(self, width, height, K):
        self.image_width = width
        self.image_height = height
        self.get_K = K


class TestDataGenerator:
    """Generate test data for apply_flow_up_down_left_right function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        width, height = 64, 64
        K = torch.tensor([[100.0, 0.0, width/2],
                         [0.0, 100.0, height/2],
                         [0.0, 0.0, 1.0]], dtype=torch.float32)
        viewpoint_cam = MockViewpointCam(width, height, K)
        rays_dis_hom = torch.randn(height * width, 3, dtype=torch.float32)
        rays_dis_hom = rays_dis_hom / rays_dis_hom.norm(dim=1, keepdim=True)
        img = torch.randn(3, height, width, dtype=torch.float32)
        test_cases.append({
            'viewpoint_cam': viewpoint_cam,
            'rays_dis_hom': rays_dis_hom,
            'img': img,
            'types': 'forward',
            'is_fisheye': False,
            'iteration': None,
            'description': f'Basic forward: {width}x{height}, normalized rays',
        })

        if num_tests > 1:
            width, height = 64, 64
            K = torch.tensor([[120.0, 0.0, width/2],
                             [0.0, 120.0, height/2],
                             [0.0, 0.0, 1.0]], dtype=torch.float32)
            viewpoint_cam = MockViewpointCam(width, height, K)
            rays_dis_hom = torch.randn(height * width, 3, dtype=torch.float32)
            rays_dis_hom[:, 0] = torch.abs(rays_dis_hom[:, 0]) + 0.1
            img = torch.randn(3, height, width, dtype=torch.float32)
            test_cases.append({
                'viewpoint_cam': viewpoint_cam,
                'rays_dis_hom': rays_dis_hom,
                'img': img,
                'types': 'left',
                'is_fisheye': True,
                'iteration': 100,
                'description': f'Left direction: {width}x{height}, fisheye',
            })

        if num_tests > 2:
            width, height = 64, 64
            K = torch.tensor([[110.0, 0.0, width/2],
                             [0.0, 110.0, height/2],
                             [0.0, 0.0, 1.0]], dtype=torch.float32)
            viewpoint_cam = MockViewpointCam(width, height, K)
            rays_dis_hom = torch.randn(height * width, 3, dtype=torch.float32)
            rays_dis_hom[:, 0] = torch.abs(rays_dis_hom[:, 0]) + 0.1
            img = torch.randn(3, height, width, dtype=torch.float32)
            test_cases.append({
                'viewpoint_cam': viewpoint_cam,
                'rays_dis_hom': rays_dis_hom,
                'img': img,
                'types': 'right',
                'is_fisheye': False,
                'iteration': None,
                'description': f'Right direction: {width}x{height}',
            })

        if num_tests > 3:
            width, height = 64, 64
            K = torch.tensor([[100.0, 0.0, width/2],
                             [0.0, 100.0, height/2],
                             [0.0, 0.0, 1.0]], dtype=torch.float32)
            viewpoint_cam = MockViewpointCam(width, height, K)
            rays_dis_hom = torch.randn(height * width, 3, dtype=torch.float32)
            rays_dis_hom[:, 1] = torch.abs(rays_dis_hom[:, 1]) + 0.1
            img = torch.randn(3, height, width, dtype=torch.float32)
            test_cases.append({
                'viewpoint_cam': viewpoint_cam,
                'rays_dis_hom': rays_dis_hom,
                'img': img,
                'types': 'up',
                'is_fisheye': True,
                'iteration': 200,
                'description': f'Up direction: {width}x{height}, fisheye',
            })

        if num_tests > 4:
            width, height = 64, 64
            K = torch.tensor([[130.0, 0.0, width/2],
                             [0.0, 130.0, height/2],
                             [0.0, 0.0, 1.0]], dtype=torch.float32)
            viewpoint_cam = MockViewpointCam(width, height, K)
            rays_dis_hom = torch.randn(height * width, 3, dtype=torch.float32)
            rays_dis_hom[:, 1] = torch.abs(rays_dis_hom[:, 1]) + 0.1
            img = torch.randn(3, height, width, dtype=torch.float32)
            test_cases.append({
                'viewpoint_cam': viewpoint_cam,
                'rays_dis_hom': rays_dis_hom,
                'img': img,
                'types': 'down',
                'is_fisheye': False,
                'iteration': None,
                'description': f'Down direction: {width}x{height}',
            })

        if num_tests > 5:
            width, height = 128, 128
            K = torch.tensor([[200.0, 0.0, width/2],
                             [0.0, 200.0, height/2],
                             [0.0, 0.0, 1.0]], dtype=torch.float32)
            viewpoint_cam = MockViewpointCam(width, height, K)
            rays_dis_hom = torch.randn(height * width, 3, dtype=torch.float32)
            rays_dis_hom = rays_dis_hom / rays_dis_hom.norm(dim=1, keepdim=True)
            img = torch.randn(3, height, width, dtype=torch.float32)
            test_cases.append({
                'viewpoint_cam': viewpoint_cam,
                'rays_dis_hom': rays_dis_hom,
                'img': img,
                'types': 'forward',
                'is_fisheye': True,
                'iteration': 500,
                'description': f'Large image: {width}x{height}, fisheye',
            })

        if num_tests > 6:
            width, height = 128, 64
            K = torch.tensor([[150.0, 0.0, width/2],
                             [0.0, 150.0, height/2],
                             [0.0, 0.0, 1.0]], dtype=torch.float32)
            viewpoint_cam = MockViewpointCam(width, height, K)
            rays_dis_hom = torch.randn(height * width, 3, dtype=torch.float32)
            rays_dis_hom[:, 0] = torch.abs(rays_dis_hom[:, 0]) + 0.1
            img = torch.randn(3, height, width, dtype=torch.float32)
            test_cases.append({
                'viewpoint_cam': viewpoint_cam,
                'rays_dis_hom': rays_dis_hom,
                'img': img,
                'types': 'left',
                'is_fisheye': False,
                'iteration': None,
                'description': f'Non-square: {width}x{height}, left direction',
            })

        if num_tests > 7:
            width, height = 64, 64
            K = torch.tensor([[80.0, 0.0, width/2 + 5],
                             [0.0, 90.0, height/2 - 3],
                             [0.0, 0.0, 1.0]], dtype=torch.float32)
            viewpoint_cam = MockViewpointCam(width, height, K)
            rays_dis_hom = torch.randn(height * width, 3, dtype=torch.float32)
            rays_dis_hom[:, 0] = torch.abs(rays_dis_hom[:, 0]) + 0.1
            img = torch.randn(3, height, width, dtype=torch.float32)
            test_cases.append({
                'viewpoint_cam': viewpoint_cam,
                'rays_dis_hom': rays_dis_hom,
                'img': img,
                'types': 'right',
                'is_fisheye': True,
                'iteration': 1000,
                'description': f'Different K: {width}x{height}, right, off-center principal point',
            })

        if num_tests > 8:
            width, height = 64, 64
            K = torch.tensor([[100.0, 0.0, width/2],
                             [0.0, 100.0, height/2],
                             [0.0, 0.0, 1.0]], dtype=torch.float32)
            viewpoint_cam = MockViewpointCam(width, height, K)
            rays_dis_hom = torch.randn(height * width, 3, dtype=torch.float32)
            rays_dis_hom[:, 1] = torch.abs(rays_dis_hom[:, 1]) + 0.5
            img = torch.randn(3, height, width, dtype=torch.float32)
            test_cases.append({
                'viewpoint_cam': viewpoint_cam,
                'rays_dis_hom': rays_dis_hom,
                'img': img,
                'types': 'up',
                'is_fisheye': False,
                'iteration': None,
                'description': f'Edge case: {width}x{height}, up, larger y values',
            })

        if num_tests > 9:
            width, height = 64, 64
            K = torch.tensor([[100.0, 0.0, width/2],
                             [0.0, 100.0, height/2],
                             [0.0, 0.0, 1.0]], dtype=torch.float32)
            viewpoint_cam = MockViewpointCam(width, height, K)
            rays_dis_hom = torch.randn(height * width, 3, dtype=torch.float32)
            rays_dis_hom[:, 1] = torch.abs(rays_dis_hom[:, 1]) + 0.1
            img = torch.randn(3, height, width, dtype=torch.float32)
            test_cases.append({
                'viewpoint_cam': viewpoint_cam,
                'rays_dis_hom': rays_dis_hom,
                'img': img,
                'types': 'down',
                'is_fisheye': True,
                'iteration': 1500,
                'description': f'Final test: {width}x{height}, down, fisheye',
            })

        for i in range(num_tests - len(test_cases)):
            width = 64 + (i % 2) * 32
            height = 64 + (i % 3) * 16
            K = torch.tensor([[100.0 + i*10, 0.0, width/2],
                             [0.0, 100.0 + i*10, height/2],
                             [0.0, 0.0, 1.0]], dtype=torch.float32)
            viewpoint_cam = MockViewpointCam(width, height, K)
            rays_dis_hom = torch.randn(height * width, 3, dtype=torch.float32)
            rays_dis_hom = rays_dis_hom / rays_dis_hom.norm(dim=1, keepdim=True)
            img = torch.randn(3, height, width, dtype=torch.float32)
            types_list = ['forward', 'left', 'right', 'up', 'down']
            test_cases.append({
                'viewpoint_cam': viewpoint_cam,
                'rays_dis_hom': rays_dis_hom,
                'img': img,
                'types': types_list[i % len(types_list)],
                'is_fisheye': i % 2 == 0,
                'iteration': i * 100 if i % 2 == 0 else None,
                'description': f'Additional test {i+1}: {width}x{height}, {types_list[i % len(types_list)]}',
            })

        return test_cases[:num_tests]
