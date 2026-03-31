"""
Test Data Generator for cross_warp_with_pose_depth_candidates function.
Generates various test cases with different configurations.
"""

import torch


class TestDataGenerator:
    """Generate test data for cross_warp_with_pose_depth_candidates."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_pose(self, b):
        """Generate random camera pose [B, 4, 4]."""
        # Random rotation (small angles for stability)
        angles = torch.randn(b, 3) * 0.3

        poses = []
        for i in range(b):
            # Rotation matrices
            rx = angles[i, 0]
            ry = angles[i, 1]
            rz = angles[i, 2]

            Rx = torch.tensor(
                [
                    [1, 0, 0],
                    [0, torch.cos(rx), -torch.sin(rx)],
                    [0, torch.sin(rx), torch.cos(rx)],
                ],
                dtype=torch.float32,
            )

            Ry = torch.tensor(
                [
                    [torch.cos(ry), 0, torch.sin(ry)],
                    [0, 1, 0],
                    [-torch.sin(ry), 0, torch.cos(ry)],
                ],
                dtype=torch.float32,
            )

            Rz = torch.tensor(
                [
                    [torch.cos(rz), -torch.sin(rz), 0],
                    [torch.sin(rz), torch.cos(rz), 0],
                    [0, 0, 1],
                ],
                dtype=torch.float32,
            )

            R = Rz @ Ry @ Rx

            # Translation
            t = torch.randn(3, 1) * 0.5

            # Construct pose
            pose = torch.eye(4)
            pose[:3, :3] = R
            pose[:3, 3:4] = t

            poses.append(pose)

        return torch.stack(poses, dim=0)

    def generate_intrinsics(self, b):
        """Generate camera intrinsics [B, 3, 3]."""
        intrinsics = []
        for _ in range(b):
            fx = fy = 500.0 + torch.randn(1).item() * 50
            cx = cy = 0.5
            K = torch.tensor(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                dtype=torch.float32,
            )
            intrinsics.append(K)
        return torch.stack(intrinsics, dim=0)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic small test
        test_cases.append(
            {
                "b": 1,
                "c": 4,
                "h": 8,
                "w": 8,
                "d": 4,
                "description": "Basic small test (b=1, c=4, h=8, w=8, d=4)",
            }
        )

        if num_tests > 1:
            # Test 2: Medium size
            test_cases.append(
                {
                    "b": 2,
                    "c": 8,
                    "h": 16,
                    "w": 16,
                    "d": 8,
                    "description": "Medium size (b=2, c=8, h=16, w=16, d=8)",
                }
            )

        if num_tests > 2:
            # Test 3: More channels
            test_cases.append(
                {
                    "b": 1,
                    "c": 16,
                    "h": 12,
                    "w": 12,
                    "d": 6,
                    "description": "More channels (b=1, c=16, h=12, w=12, d=6)",
                }
            )

        if num_tests > 3:
            # Test 4: Batch processing
            test_cases.append(
                {
                    "b": 4,
                    "c": 8,
                    "h": 16,
                    "w": 16,
                    "d": 8,
                    "description": "Batch processing (b=4, c=8, h=16, w=16, d=8)",
                }
            )

        if num_tests > 4:
            # Test 5: Non-square dimensions
            test_cases.append(
                {
                    "b": 2,
                    "c": 8,
                    "h": 12,
                    "w": 24,
                    "d": 10,
                    "description": "Non-square dimensions (b=2, c=8, h=12, w=24, d=10)",
                }
            )

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            h_val = 8 + (i + 1) * 4
            w_val = 8 + (i + 1) * 4
            test_cases.append(
                {
                    "b": 1 + (i % 3),
                    "c": 4 + (i % 4) * 4,
                    "h": h_val,
                    "w": w_val,
                    "d": 4 + (i % 3) * 2,
                    "description": (
                        f"Additional test {i+1} "
                        f"(b={1 + (i % 3)}, c={4 + (i % 4) * 4}, h={h_val}, w={w_val}, d={4 + (i % 3) * 2})"
                    ),
                }
            )

        return test_cases[:num_tests]
