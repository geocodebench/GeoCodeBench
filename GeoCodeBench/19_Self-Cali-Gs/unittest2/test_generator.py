"""Generate test data for apply_distortion function."""

import torch
import torch.nn as nn


class MockLensNet(nn.Module):
    """Mock lens network for testing."""

    def __init__(self, input_dim=2, hidden_dim=32, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, sensor_to_frustum=False):
        scale = 1.1 if sensor_to_frustum else 1.0
        return self.net(x) * scale


class MockViewpointCam:
    """Mock camera viewpoint for testing."""

    def __init__(self, image_height=64, image_width=64, num_channels=3):
        self.image_height = image_height
        self.image_width = image_width
        self.num_channels = num_channels
        self.fish_gt_image = torch.rand(num_channels, image_height, image_width)
        self.projection_matrix = torch.eye(3) * 0.5
        self.flow4gt = torch.eye(3) * 0.6
        self.fish_gt_image_resolution = torch.tensor([num_channels, image_height, image_width])


class TestDataGenerator:
    """Generate test data for apply_distortion function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        lens_net = MockLensNet()
        lens_net.eval()
        H, W = 32, 32
        P_sensor = torch.randn(H, W, 2)
        P_view_insidelens_direction = torch.randn(H * W, 2)
        viewpoint_cam = MockViewpointCam(image_height=64, image_width=64)
        image = torch.rand(3, 64, 64)
        test_cases.append({
            'flow_apply2_gt_or_img': None,
            'lens_net': lens_net,
            'P_view_insidelens_direction': P_view_insidelens_direction,
            'P_sensor': P_sensor,
            'viewpoint_cam': viewpoint_cam,
            'image': image,
            'apply2gt': True,
            'flow_scale': None,
            'description': f'Basic: apply2gt=True, H={H}, W={W}',
        })

        if num_tests > 1:
            lens_net = MockLensNet()
            lens_net.eval()
            H, W = 32, 32
            P_sensor = torch.randn(H, W, 2)
            P_view_insidelens_direction = torch.randn(H * W, 2)
            viewpoint_cam = MockViewpointCam(image_height=64, image_width=64)
            image = torch.rand(3, 96, 96)
            flow_scale = [1.2, 1.2]
            test_cases.append({
                'flow_apply2_gt_or_img': None,
                'lens_net': lens_net,
                'P_view_insidelens_direction': P_view_insidelens_direction,
                'P_sensor': P_sensor,
                'viewpoint_cam': viewpoint_cam,
                'image': image,
                'apply2gt': False,
                'flow_scale': flow_scale,
                'description': f'Basic: apply2gt=False, H={H}, W={W}, flow_scale={flow_scale}',
            })

        if num_tests > 2:
            H, W = 24, 24
            lens_net = MockLensNet()
            lens_net.eval()
            P_sensor = torch.randn(H, W, 2)
            P_view_insidelens_direction = torch.randn(H * W, 2)
            viewpoint_cam = MockViewpointCam(image_height=48, image_width=48)
            image = torch.rand(3, 48, 48)
            flow = torch.randn(48, 48, 2) * 0.5
            test_cases.append({
                'flow_apply2_gt_or_img': flow,
                'lens_net': lens_net,
                'P_view_insidelens_direction': P_view_insidelens_direction,
                'P_sensor': P_sensor,
                'viewpoint_cam': viewpoint_cam,
                'image': image,
                'apply2gt': True,
                'flow_scale': None,
                'description': f'Pre-computed flow: apply2gt=True, flow_shape={flow.shape}',
            })

        if num_tests > 3:
            H, W = 24, 24
            lens_net = MockLensNet()
            lens_net.eval()
            P_sensor = torch.randn(H, W, 2)
            P_view_insidelens_direction = torch.randn(H * W, 2)
            viewpoint_cam = MockViewpointCam(image_height=48, image_width=48)
            image = torch.rand(3, 72, 72)
            flow_scale = [1.0, 1.0]
            flow = torch.randn(48, 48, 2) * 0.5
            test_cases.append({
                'flow_apply2_gt_or_img': flow,
                'lens_net': lens_net,
                'P_view_insidelens_direction': P_view_insidelens_direction,
                'P_sensor': P_sensor,
                'viewpoint_cam': viewpoint_cam,
                'image': image,
                'apply2gt': False,
                'flow_scale': flow_scale,
                'description': f'Pre-computed flow: apply2gt=False, flow_shape={flow.shape}',
            })

        if num_tests > 4:
            lens_net = MockLensNet()
            lens_net.eval()
            H, W = 48, 48
            P_sensor = torch.randn(H, W, 2)
            P_view_insidelens_direction = torch.randn(H * W, 2)
            viewpoint_cam = MockViewpointCam(image_height=128, image_width=128)
            image = torch.rand(3, 128, 128)
            test_cases.append({
                'flow_apply2_gt_or_img': None,
                'lens_net': lens_net,
                'P_view_insidelens_direction': P_view_insidelens_direction,
                'P_sensor': P_sensor,
                'viewpoint_cam': viewpoint_cam,
                'image': image,
                'apply2gt': True,
                'flow_scale': None,
                'description': f'Larger resolution: apply2gt=True, H={H}, W={W}, image={image.shape}',
            })

        if num_tests > 5:
            lens_net = MockLensNet()
            lens_net.eval()
            H, W = 16, 32
            P_sensor = torch.randn(H, W, 2)
            P_view_insidelens_direction = torch.randn(H * W, 2)
            viewpoint_cam = MockViewpointCam(image_height=64, image_width=128)
            viewpoint_cam.fish_gt_image_resolution = torch.tensor([3, 64, 128])
            image = torch.rand(3, 96, 192)
            flow_scale = [1.5, 1.5]
            test_cases.append({
                'flow_apply2_gt_or_img': None,
                'lens_net': lens_net,
                'P_view_insidelens_direction': P_view_insidelens_direction,
                'P_sensor': P_sensor,
                'viewpoint_cam': viewpoint_cam,
                'image': image,
                'apply2gt': False,
                'flow_scale': flow_scale,
                'description': f'Different aspect ratio: H={H}, W={W}, aspect={W/H:.2f}',
            })

        if num_tests > 6:
            lens_net = MockLensNet()
            lens_net.eval()
            H, W = 16, 16
            P_sensor = torch.randn(H, W, 2)
            P_view_insidelens_direction = torch.randn(H * W, 2)
            viewpoint_cam = MockViewpointCam(image_height=32, image_width=32)
            image = torch.rand(3, 32, 32)
            test_cases.append({
                'flow_apply2_gt_or_img': None,
                'lens_net': lens_net,
                'P_view_insidelens_direction': P_view_insidelens_direction,
                'P_sensor': P_sensor,
                'viewpoint_cam': viewpoint_cam,
                'image': image,
                'apply2gt': True,
                'flow_scale': None,
                'description': f'Small resolution: H={H}, W={W}',
            })

        if num_tests > 7:
            lens_net = MockLensNet()
            lens_net.eval()
            H, W = 20, 40
            P_sensor = torch.randn(H, W, 2)
            P_view_insidelens_direction = torch.randn(H * W, 2)
            viewpoint_cam = MockViewpointCam(image_height=80, image_width=80)
            image = torch.rand(3, 120, 120)
            flow_scale = [1.0, 1.0]
            test_cases.append({
                'flow_apply2_gt_or_img': None,
                'lens_net': lens_net,
                'P_view_insidelens_direction': P_view_insidelens_direction,
                'P_sensor': P_sensor,
                'viewpoint_cam': viewpoint_cam,
                'image': image,
                'apply2gt': False,
                'flow_scale': flow_scale,
                'description': f'Non-square sensor: H={H}, W={W}',
            })

        if num_tests > 8:
            lens_net = MockLensNet()
            lens_net.eval()
            H, W = 32, 32
            P_sensor = torch.randn(H, W, 2)
            P_view_insidelens_direction = torch.randn(H * W, 2)
            viewpoint_cam = MockViewpointCam(image_height=64, image_width=64)
            image = torch.rand(3, 128, 128)
            flow_scale = [0.5, 0.5]
            test_cases.append({
                'flow_apply2_gt_or_img': None,
                'lens_net': lens_net,
                'P_view_insidelens_direction': P_view_insidelens_direction,
                'P_sensor': P_sensor,
                'viewpoint_cam': viewpoint_cam,
                'image': image,
                'apply2gt': False,
                'flow_scale': flow_scale,
                'description': f'Small flow scale: flow_scale={flow_scale}',
            })

        if num_tests > 9:
            lens_net = MockLensNet()
            lens_net.eval()
            H, W = 32, 32
            P_sensor = torch.randn(H, W, 2)
            P_view_insidelens_direction = torch.randn(H * W, 2)
            viewpoint_cam = MockViewpointCam(image_height=64, image_width=64)
            image = torch.rand(3, 256, 256)
            flow_scale = [2.0, 2.0]
            test_cases.append({
                'flow_apply2_gt_or_img': None,
                'lens_net': lens_net,
                'P_view_insidelens_direction': P_view_insidelens_direction,
                'P_sensor': P_sensor,
                'viewpoint_cam': viewpoint_cam,
                'image': image,
                'apply2gt': False,
                'flow_scale': flow_scale,
                'description': f'Large flow scale: flow_scale={flow_scale}',
            })

        for i in range(num_tests - len(test_cases)):
            lens_net = MockLensNet()
            lens_net.eval()
            H = 24 + i * 4
            W = 24 + i * 4
            P_sensor = torch.randn(H, W, 2)
            P_view_insidelens_direction = torch.randn(H * W, 2)
            img_size = 64 + i * 16
            viewpoint_cam = MockViewpointCam(image_height=img_size, image_width=img_size)
            image = torch.rand(3, img_size, img_size)
            apply2gt = i % 2 == 0
            flow_scale = [1.0 + i * 0.1, 1.0 + i * 0.1] if not apply2gt else None
            test_cases.append({
                'flow_apply2_gt_or_img': None,
                'lens_net': lens_net,
                'P_view_insidelens_direction': P_view_insidelens_direction,
                'P_sensor': P_sensor,
                'viewpoint_cam': viewpoint_cam,
                'image': image,
                'apply2gt': apply2gt,
                'flow_scale': flow_scale,
                'description': f'Additional test {i+1}: H={H}, W={W}, apply2gt={apply2gt}',
            })

        return test_cases[:num_tests]
