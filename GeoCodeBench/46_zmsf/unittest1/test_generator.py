"""
Test Data Generator for compute_scene_flow() function.
Generates various test cases with different configurations.
"""

import torch


class TestDataGenerator:
    """Generate test data for compute_scene_flow() function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []
        device = torch.device('cpu')  # Ensure CPU device

        # Test 1: Basic case with small batch and image size
        B, H, W = 1, 32, 32
        pts3d_left = torch.randn(B, H, W, 3, device=device)
        flow_left_to_right = torch.randn(B, H, W, 2, device=device) * 2.0  # Small flow
        pts3d_right = torch.randn(B, H, W, 3, device=device)
        test_cases.append({
            'pts3d_left': pts3d_left,
            'flow_left_to_right': flow_left_to_right,
            'pts3d_right': pts3d_right,
            'description': f'Basic: B={B}, H={H}, W={W}',
        })

        if num_tests > 1:
            # Test 2: Larger batch
            B, H, W = 4, 32, 32
            pts3d_left = torch.randn(B, H, W, 3, device=device)
            flow_left_to_right = torch.randn(B, H, W, 2, device=device) * 1.5
            pts3d_right = torch.randn(B, H, W, 3, device=device)
            test_cases.append({
                'pts3d_left': pts3d_left,
                'flow_left_to_right': flow_left_to_right,
                'pts3d_right': pts3d_right,
                'description': f'Larger batch: B={B}, H={H}, W={W}',
            })

        if num_tests > 2:
            # Test 3: Larger image size
            B, H, W = 2, 64, 64
            pts3d_left = torch.randn(B, H, W, 3, device=device)
            flow_left_to_right = torch.randn(B, H, W, 2, device=device) * 3.0
            pts3d_right = torch.randn(B, H, W, 3, device=device)
            test_cases.append({
                'pts3d_left': pts3d_left,
                'flow_left_to_right': flow_left_to_right,
                'pts3d_right': pts3d_right,
                'description': f'Larger image: B={B}, H={H}, W={W}',
            })

        if num_tests > 3:
            # Test 4: Small image size
            B, H, W = 1, 16, 16
            pts3d_left = torch.randn(B, H, W, 3, device=device)
            flow_left_to_right = torch.randn(B, H, W, 2, device=device) * 0.5
            pts3d_right = torch.randn(B, H, W, 3, device=device)
            test_cases.append({
                'pts3d_left': pts3d_left,
                'flow_left_to_right': flow_left_to_right,
                'pts3d_right': pts3d_right,
                'description': f'Small image: B={B}, H={H}, W={W}',
            })

        if num_tests > 4:
            # Test 5: Large flow values
            B, H, W = 2, 32, 32
            pts3d_left = torch.randn(B, H, W, 3, device=device)
            flow_left_to_right = torch.randn(B, H, W, 2, device=device) * 5.0
            pts3d_right = torch.randn(B, H, W, 3, device=device)
            test_cases.append({
                'pts3d_left': pts3d_left,
                'flow_left_to_right': flow_left_to_right,
                'pts3d_right': pts3d_right,
                'description': f'Large flow: B={B}, H={H}, W={W}',
            })

        if num_tests > 5:
            # Test 6: Very large batch
            B, H, W = 8, 32, 32
            pts3d_left = torch.randn(B, H, W, 3, device=device)
            flow_left_to_right = torch.randn(B, H, W, 2, device=device) * 2.0
            pts3d_right = torch.randn(B, H, W, 3, device=device)
            test_cases.append({
                'pts3d_left': pts3d_left,
                'flow_left_to_right': flow_left_to_right,
                'pts3d_right': pts3d_right,
                'description': f'Very large batch: B={B}, H={H}, W={W}',
            })

        if num_tests > 6:
            # Test 7: Edge case - zero flow
            B, H, W = 1, 32, 32
            pts3d_left = torch.randn(B, H, W, 3, device=device)
            flow_left_to_right = torch.zeros(B, H, W, 2, device=device)
            pts3d_right = torch.randn(B, H, W, 3, device=device)
            test_cases.append({
                'pts3d_left': pts3d_left,
                'flow_left_to_right': flow_left_to_right,
                'pts3d_right': pts3d_right,
                'description': f'Zero flow: B={B}, H={H}, W={W}',
            })

        if num_tests > 7:
            # Test 8: High resolution
            B, H, W = 1, 128, 128
            pts3d_left = torch.randn(B, H, W, 3, device=device)
            flow_left_to_right = torch.randn(B, H, W, 2, device=device) * 2.0
            pts3d_right = torch.randn(B, H, W, 3, device=device)
            test_cases.append({
                'pts3d_left': pts3d_left,
                'flow_left_to_right': flow_left_to_right,
                'pts3d_right': pts3d_right,
                'description': f'High resolution: B={B}, H={H}, W={W}',
            })

        if num_tests > 8:
            # Test 9: Non-square image
            B, H, W = 2, 32, 64
            pts3d_left = torch.randn(B, H, W, 3, device=device)
            flow_left_to_right = torch.randn(B, H, W, 2, device=device) * 2.0
            pts3d_right = torch.randn(B, H, W, 3, device=device)
            test_cases.append({
                'pts3d_left': pts3d_left,
                'flow_left_to_right': flow_left_to_right,
                'pts3d_right': pts3d_right,
                'description': f'Non-square: B={B}, H={H}, W={W}',
            })

        if num_tests > 9:
            # Test 10: Small flow values
            B, H, W = 1, 32, 32
            pts3d_left = torch.randn(B, H, W, 3, device=device)
            flow_left_to_right = torch.randn(B, H, W, 2, device=device) * 0.1
            pts3d_right = torch.randn(B, H, W, 3, device=device)
            test_cases.append({
                'pts3d_left': pts3d_left,
                'flow_left_to_right': flow_left_to_right,
                'pts3d_right': pts3d_right,
                'description': f'Small flow: B={B}, H={H}, W={W}',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            B = 1 + (i % 4)
            H = 16 + (i % 5) * 8
            W = 16 + (i % 5) * 8
            pts3d_left = torch.randn(B, H, W, 3, device=device)
            flow_left_to_right = torch.randn(B, H, W, 2, device=device) * (1.0 + (i % 3))
            pts3d_right = torch.randn(B, H, W, 3, device=device)

            test_cases.append({
                'pts3d_left': pts3d_left,
                'flow_left_to_right': flow_left_to_right,
                'pts3d_right': pts3d_right,
                'description': f'Additional test {i+1}: B={B}, H={H}, W={W}',
            })

        return test_cases[:num_tests]
