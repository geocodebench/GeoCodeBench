"""
Test Data Generator for world_flow_to_optical_flow and scene_flow_to_optical_flow.
Generates various test cases with different configurations.
"""

import torch


def create_camera_pose(batch_size, device='cpu'):
    """Create a random valid camera pose (world to camera transformation)."""
    poses = []
    for _ in range(batch_size):
        # Create random rotation matrix (orthogonal)
        R = torch.randn(3, 3, device=device)
        U, _, V = torch.linalg.svd(R)
        R = U @ V.T  # Ensure orthogonal

        # Random translation
        t = torch.randn(3, device=device) * 2.0

        # Create 4x4 pose matrix
        pose = torch.eye(4, device=device)
        pose[:3, :3] = R
        pose[:3, 3] = t
        poses.append(pose)

    return torch.stack(poses)


def create_intrinsics(batch_size, H, W, device='cpu'):
    """Create camera intrinsics matrix."""
    intrinsics = []
    for _ in range(batch_size):
        fx = fy = float(max(H, W)) * 0.8 + torch.rand(1, device=device).item() * 0.4 * max(H, W)
        cx = W / 2.0 + (torch.rand(1, device=device).item() - 0.5) * W * 0.2
        cy = H / 2.0 + (torch.rand(1, device=device).item() - 0.5) * H * 0.2

        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], device=device, dtype=torch.float32)
        intrinsics.append(K)

    return torch.stack(intrinsics)


class TestDataGenerator:
    """Generate test data for the two functions."""

    def __init__(self, seed=42, device='cpu'):
        self.seed = seed
        self.device = device
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case - small batch, small image
        B, H, W = 2, 32, 32
        test_cases.append(self._create_world_flow_test(B, H, W, 'Basic: B=2, H=32, W=32'))
        test_cases.append(self._create_scene_flow_test(B, H, W, 'Basic: B=2, H=32, W=32'))

        if num_tests > 1:
            # Test 2: Single batch, medium image
            B, H, W = 1, 64, 64
            test_cases.append(self._create_world_flow_test(B, H, W, 'Single batch: B=1, H=64, W=64'))
            test_cases.append(self._create_scene_flow_test(B, H, W, 'Single batch: B=1, H=64, W=64'))

        if num_tests > 2:
            # Test 3: Larger batch, larger image
            B, H, W = 4, 128, 128
            test_cases.append(self._create_world_flow_test(B, H, W, 'Larger: B=4, H=128, W=128'))
            test_cases.append(self._create_scene_flow_test(B, H, W, 'Larger: B=4, H=128, W=128'))

        if num_tests > 3:
            # Test 4: Different aspect ratio
            B, H, W = 2, 48, 64
            test_cases.append(self._create_world_flow_test(B, H, W, 'Aspect ratio: B=2, H=48, W=64'))
            test_cases.append(self._create_scene_flow_test(B, H, W, 'Aspect ratio: B=2, H=48, W=64'))

        if num_tests > 4:
            # Test 5: Small flow values
            B, H, W = 2, 32, 32
            test_cases.append(self._create_world_flow_test(B, H, W, 'Small flow: B=2, H=32, W=32', flow_scale=0.01))
            test_cases.append(self._create_scene_flow_test(B, H, W, 'Small flow: B=2, H=32, W=32', flow_scale=0.01))

        if num_tests > 5:
            # Test 6: Large flow values
            B, H, W = 2, 64, 64
            test_cases.append(self._create_world_flow_test(B, H, W, 'Large flow: B=2, H=64, W=64', flow_scale=5.0))
            test_cases.append(self._create_scene_flow_test(B, H, W, 'Large flow: B=2, H=64, W=64', flow_scale=5.0))

        if num_tests > 6:
            # Test 7: Edge case - points close to camera
            B, H, W = 2, 32, 32
            test_cases.append(self._create_world_flow_test(B, H, W, 'Close points: B=2, H=32, W=32', depth_scale=0.5))
            test_cases.append(self._create_scene_flow_test(B, H, W, 'Close points: B=2, H=32, W=32', depth_scale=0.5))

        if num_tests > 7:
            # Test 8: Edge case - points far from camera
            B, H, W = 2, 32, 32
            test_cases.append(self._create_world_flow_test(B, H, W, 'Far points: B=2, H=32, W=32', depth_scale=10.0))
            test_cases.append(self._create_scene_flow_test(B, H, W, 'Far points: B=2, H=32, W=32', depth_scale=10.0))

        if num_tests > 8:
            # Test 9: Very small image
            B, H, W = 1, 16, 16
            test_cases.append(self._create_world_flow_test(B, H, W, 'Small image: B=1, H=16, W=16'))
            test_cases.append(self._create_scene_flow_test(B, H, W, 'Small image: B=1, H=16, W=16'))

        if num_tests > 9:
            # Test 10: Large batch
            B, H, W = 8, 64, 64
            test_cases.append(self._create_world_flow_test(B, H, W, 'Large batch: B=8, H=64, W=64'))
            test_cases.append(self._create_scene_flow_test(B, H, W, 'Large batch: B=8, H=64, W=64'))

        # Generate additional tests if needed
        remaining = num_tests * 2 - len(test_cases)
        for i in range(remaining // 2):
            B = 1 + (i % 4)
            H = 32 + (i % 5) * 16
            W = 32 + (i % 5) * 16
            test_cases.append(self._create_world_flow_test(B, H, W, f'Additional test {i+1}: B={B}, H={H}, W={W}'))
            test_cases.append(self._create_scene_flow_test(B, H, W, f'Additional test {i+1}: B={B}, H={H}, W={W}'))

        return test_cases[:num_tests * 2]  # Return num_tests pairs (one for each function)

    def _create_world_flow_test(self, B, H, W, description, flow_scale=1.0, depth_scale=1.0):
        """Create test case for world_flow_to_optical_flow."""
        # Generate random 3D flow in world space
        flow_3d_fwd_world = torch.randn(B, H, W, 3, device=self.device) * flow_scale

        # Generate 3D points in source camera space
        # Create points with reasonable depth
        u = torch.arange(W, device=self.device).float().view(1, 1, W).expand(B, H, W)
        v = torch.arange(H, device=self.device).float().view(1, H, 1).expand(B, H, W)
        u = (u - W / 2) / W
        v = (v - H / 2) / H

        # Random depth
        depth = (torch.rand(B, H, W, device=self.device) * 5.0 + 1.0) * depth_scale

        # Create intrinsics
        intrinsics_src = create_intrinsics(B, H, W, self.device)

        # Unproject to 3D points
        fx = intrinsics_src[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
        fy = intrinsics_src[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
        cx = intrinsics_src[:, 0, 2].unsqueeze(-1).unsqueeze(-1)
        cy = intrinsics_src[:, 1, 2].unsqueeze(-1).unsqueeze(-1)

        x = (u * W - cx) * depth / fx
        y = (v * H - cy) * depth / fy
        z = depth

        points_3d = torch.stack([x, y, z], dim=-1)

        # Create camera poses
        camera_pose_src = create_camera_pose(B, self.device)
        camera_pose_tgt = create_camera_pose(B, self.device)

        return {
            'function': 'world_flow_to_optical_flow',
            'description': description,
            'args': {
                'flow_3d_fwd_world': flow_3d_fwd_world,
                'points_3d': points_3d,
                'intrinsics_src': intrinsics_src,
                'camera_pose_src': camera_pose_src,
                'camera_pose_tgt': camera_pose_tgt,
                'eps': 1e-8
            }
        }

    def _create_scene_flow_test(self, B, H, W, description, flow_scale=1.0, depth_scale=1.0):
        """Create test case for scene_flow_to_optical_flow."""
        # Generate random 3D flow in camera space
        flow_3d_fwd = torch.randn(B, H, W, 3, device=self.device) * flow_scale

        # Generate 3D points in camera space
        u = torch.arange(W, device=self.device).float().view(1, 1, W).expand(B, H, W)
        v = torch.arange(H, device=self.device).float().view(1, H, 1).expand(B, H, W)
        u = (u - W / 2) / W
        v = (v - H / 2) / H

        # Random depth
        depth = (torch.rand(B, H, W, device=self.device) * 5.0 + 1.0) * depth_scale

        # Create intrinsics
        intrinsics = create_intrinsics(B, H, W, self.device)

        # Unproject to 3D points
        fx = intrinsics[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
        fy = intrinsics[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
        cx = intrinsics[:, 0, 2].unsqueeze(-1).unsqueeze(-1)
        cy = intrinsics[:, 1, 2].unsqueeze(-1).unsqueeze(-1)

        x = (u * W - cx) * depth / fx
        y = (v * H - cy) * depth / fy
        z = depth

        points_3d = torch.stack([x, y, z], dim=-1)

        return {
            'function': 'scene_flow_to_optical_flow',
            'description': description,
            'args': {
                'flow_3d_fwd': flow_3d_fwd,
                'intrinsics': intrinsics,
                'points_3d': points_3d,
                'eps': 1e-8
            }
        }
