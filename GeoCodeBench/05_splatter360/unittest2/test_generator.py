import torch


def create_random_pose(batch_size=1):
    """Create random camera pose matrix."""
    pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    # Add small random rotation and translation
    pose[:, :3, :3] = pose[:, :3, :3] + torch.randn(batch_size, 3, 3) * 0.1
    pose[:, :3, 3] = torch.randn(batch_size, 3) * 0.5
    return pose


def create_random_intrinsics(batch_size=1, h=32, w=32):
    """Create random camera intrinsics matrix."""
    intrinsics = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
    # Focal length
    intrinsics[:, 0, 0] = w * 0.8 + torch.randn(batch_size) * 0.1 * w
    intrinsics[:, 1, 1] = h * 0.8 + torch.randn(batch_size) * 0.1 * h
    # Principal point
    intrinsics[:, 0, 2] = w / 2 + torch.randn(batch_size) * 0.1 * w
    intrinsics[:, 1, 2] = h / 2 + torch.randn(batch_size) * 0.1 * h
    return intrinsics


class TestDataGenerator:
    """Generate test data for correlation_softmax_depth function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case
        b, c, h, w = 2, 64, 16, 16
        d = 8
        intrinsics = create_random_intrinsics(b, h, w)
        pose = create_random_pose(b)
        depth_candidates = torch.linspace(0.5, 10.0, d).view(1, d, 1, 1).repeat(b, 1, h, w)
        test_cases.append({
            "feature0": torch.randn(b, c, h, w),
            "feature1": torch.randn(b, c, h, w),
            "intrinsics": intrinsics,
            "pose": pose,
            "depth_candidates": depth_candidates,
            "depth_from_argmax": False,
            "pred_bidir_depth": False,
            "description": f"Basic: B={b}, C={c}, H={h}, W={w}, D={d}",
        })

        if num_tests > 1:
            # Test 2: With argmax
            b, c, h, w = 2, 128, 16, 16
            d = 16
            intrinsics = create_random_intrinsics(b, h, w)
            pose = create_random_pose(b)
            depth_candidates = torch.linspace(0.5, 10.0, d).view(1, d, 1, 1).repeat(b, 1, h, w)
            test_cases.append({
                "feature0": torch.randn(b, c, h, w),
                "feature1": torch.randn(b, c, h, w),
                "intrinsics": intrinsics,
                "pose": pose,
                "depth_candidates": depth_candidates,
                "depth_from_argmax": True,
                "pred_bidir_depth": False,
                "description": f"With argmax: B={b}, C={c}, H={h}, W={w}, D={d}",
            })

        if num_tests > 2:
            # Test 3: Bidirectional depth
            b, c, h, w = 1, 64, 16, 16
            d = 8
            intrinsics = create_random_intrinsics(b, h, w)
            pose = create_random_pose(b)
            depth_candidates = torch.linspace(1.0, 8.0, d).view(1, d, 1, 1).repeat(b, 1, h, w)
            test_cases.append({
                "feature0": torch.randn(b, c, h, w),
                "feature1": torch.randn(b, c, h, w),
                "intrinsics": intrinsics,
                "pose": pose,
                "depth_candidates": depth_candidates,
                "depth_from_argmax": False,
                "pred_bidir_depth": True,
                "description": f"Bidirectional: B={b}, C={c}, H={h}, W={w}, D={d}",
            })

        if num_tests > 3:
            # Test 4: Larger resolution
            b, c, h, w = 2, 128, 32, 32
            d = 12
            intrinsics = create_random_intrinsics(b, h, w)
            pose = create_random_pose(b)
            depth_candidates = torch.linspace(0.5, 15.0, d).view(1, d, 1, 1).repeat(b, 1, h, w)
            test_cases.append({
                "feature0": torch.randn(b, c, h, w),
                "feature1": torch.randn(b, c, h, w),
                "intrinsics": intrinsics,
                "pose": pose,
                "depth_candidates": depth_candidates,
                "depth_from_argmax": False,
                "pred_bidir_depth": False,
                "description": f"Large resolution: B={b}, C={c}, H={h}, W={w}, D={d}",
            })

        if num_tests > 4:
            # Test 5: More depth candidates
            b, c, h, w = 3, 64, 16, 16
            d = 32
            intrinsics = create_random_intrinsics(b, h, w)
            pose = create_random_pose(b)
            depth_candidates = torch.linspace(0.5, 20.0, d).view(1, d, 1, 1).repeat(b, 1, h, w)
            test_cases.append({
                "feature0": torch.randn(b, c, h, w),
                "feature1": torch.randn(b, c, h, w),
                "intrinsics": intrinsics,
                "pose": pose,
                "depth_candidates": depth_candidates,
                "depth_from_argmax": True,
                "pred_bidir_depth": False,
                "description": f"More depth candidates: B={b}, C={c}, H={h}, W={w}, D={d}",
            })

        if num_tests > 5:
            # Test 6: Bidirectional with argmax
            b, c, h, w = 2, 128, 24, 24
            d = 16
            intrinsics = create_random_intrinsics(b, h, w)
            pose = create_random_pose(b)
            depth_candidates = torch.linspace(1.0, 12.0, d).view(1, d, 1, 1).repeat(b, 1, h, w)
            test_cases.append({
                "feature0": torch.randn(b, c, h, w),
                "feature1": torch.randn(b, c, h, w),
                "intrinsics": intrinsics,
                "pose": pose,
                "depth_candidates": depth_candidates,
                "depth_from_argmax": True,
                "pred_bidir_depth": True,
                "description": f"Bidir + argmax: B={b}, C={c}, H={h}, W={w}, D={d}",
            })

        if num_tests > 6:
            # Test 7: Small features
            b, c, h, w = 2, 32, 8, 8
            d = 8
            intrinsics = create_random_intrinsics(b, h, w)
            pose = create_random_pose(b)
            depth_candidates = torch.linspace(0.5, 5.0, d).view(1, d, 1, 1).repeat(b, 1, h, w)
            test_cases.append({
                "feature0": torch.randn(b, c, h, w),
                "feature1": torch.randn(b, c, h, w),
                "intrinsics": intrinsics,
                "pose": pose,
                "depth_candidates": depth_candidates,
                "depth_from_argmax": False,
                "pred_bidir_depth": False,
                "description": f"Small features: B={b}, C={c}, H={h}, W={w}, D={d}",
            })

        if num_tests > 7:
            # Test 8: Large batch
            b, c, h, w = 8, 64, 16, 16
            d = 8
            intrinsics = create_random_intrinsics(b, h, w)
            pose = create_random_pose(b)
            depth_candidates = torch.linspace(0.5, 10.0, d).view(1, d, 1, 1).repeat(b, 1, h, w)
            test_cases.append({
                "feature0": torch.randn(b, c, h, w),
                "feature1": torch.randn(b, c, h, w),
                "intrinsics": intrinsics,
                "pose": pose,
                "depth_candidates": depth_candidates,
                "depth_from_argmax": False,
                "pred_bidir_depth": False,
                "description": f"Large batch: B={b}, C={c}, H={h}, W={w}, D={d}",
            })

        if num_tests > 8:
            # Test 9: Non-square resolution
            b, c, h, w = 2, 64, 16, 24
            d = 12
            intrinsics = create_random_intrinsics(b, h, w)
            pose = create_random_pose(b)
            depth_candidates = torch.linspace(0.5, 10.0, d).view(1, d, 1, 1).repeat(b, 1, h, w)
            test_cases.append({
                "feature0": torch.randn(b, c, h, w),
                "feature1": torch.randn(b, c, h, w),
                "intrinsics": intrinsics,
                "pose": pose,
                "depth_candidates": depth_candidates,
                "depth_from_argmax": False,
                "pred_bidir_depth": False,
                "description": f"Non-square: B={b}, C={c}, H={h}, W={w}, D={d}",
            })

        if num_tests > 9:
            # Test 10: High resolution with many candidates
            b, c, h, w = 1, 128, 32, 32
            d = 24
            intrinsics = create_random_intrinsics(b, h, w)
            pose = create_random_pose(b)
            depth_candidates = torch.linspace(0.5, 15.0, d).view(1, d, 1, 1).repeat(b, 1, h, w)
            test_cases.append({
                "feature0": torch.randn(b, c, h, w),
                "feature1": torch.randn(b, c, h, w),
                "intrinsics": intrinsics,
                "pose": pose,
                "depth_candidates": depth_candidates,
                "depth_from_argmax": True,
                "pred_bidir_depth": False,
                "description": f"High res + many candidates: B={b}, C={c}, H={h}, W={w}, D={d}",
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            b = 1 + (i % 3)
            c = 64 * (1 + (i % 2))
            h = 16 * (1 + i // 5)
            w = h
            d = 8 + (i % 3) * 4

            intrinsics = create_random_intrinsics(b, h, w)
            pose = create_random_pose(b)
            depth_candidates = torch.linspace(0.5, 10.0, d).view(1, d, 1, 1).repeat(b, 1, h, w)

            test_cases.append({
                "feature0": torch.randn(b, c, h, w),
                "feature1": torch.randn(b, c, h, w),
                "intrinsics": intrinsics,
                "pose": pose,
                "depth_candidates": depth_candidates,
                "depth_from_argmax": bool(i % 2),
                "pred_bidir_depth": bool(i % 3 == 0),
                "description": f"Additional test {i+1}: B={b}, C={c}, H={h}, W={w}, D={d}",
            })

        return test_cases[:num_tests]
