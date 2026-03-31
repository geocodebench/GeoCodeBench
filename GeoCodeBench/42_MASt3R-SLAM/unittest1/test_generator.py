"""
Test Data Generator for project_calib() function.
Generates test cases with different configurations.
"""

import torch


class TestDataGenerator:
    """Generate test data for project_calib() function."""

    def __init__(self, seed=42, device='cpu'):
        self.seed = seed
        self.device = device
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case - single point, no jacobian
        P = torch.randn(1, 3, device=self.device)
        P[0, 2] = abs(P[0, 2]) + 1.0  # Ensure z > 0
        K = torch.tensor([[500.0, 0.0, 320.0],
                          [0.0, 500.0, 240.0],
                          [0.0, 0.0, 1.0]], device=self.device)
        img_size = (480, 640)
        test_cases.append({
            'P': P,
            'K': K,
            'img_size': img_size,
            'jacobian': False,
            'border': 0,
            'z_eps': 0.0,
            'description': 'Basic: single point, no jacobian',
        })

        if num_tests > 1:
            # Test 2: Batch of points, with jacobian
            P = torch.randn(5, 3, device=self.device)
            P[:, 2] = abs(P[:, 2]) + 1.0  # Ensure z > 0
            K = torch.tensor([[800.0, 0.0, 400.0],
                             [0.0, 800.0, 300.0],
                             [0.0, 0.0, 1.0]], device=self.device)
            img_size = (600, 800)
            test_cases.append({
                'P': P,
                'K': K,
                'img_size': img_size,
                'jacobian': True,
                'border': 0,
                'z_eps': 0.0,
                'description': 'Batch: 5 points, with jacobian',
            })

        if num_tests > 2:
            # Test 3: Points with border
            P = torch.randn(3, 3, device=self.device)
            P[:, 2] = abs(P[:, 2]) + 2.0
            K = torch.tensor([[600.0, 0.0, 320.0],
                             [0.0, 600.0, 240.0],
                             [0.0, 0.0, 1.0]], device=self.device)
            img_size = (480, 640)
            test_cases.append({
                'P': P,
                'K': K,
                'img_size': img_size,
                'jacobian': False,
                'border': 10,
                'z_eps': 0.0,
                'description': 'With border=10, no jacobian',
            })

        if num_tests > 3:
            # Test 4: Points with z_eps
            P = torch.randn(4, 3, device=self.device)
            P[:, 2] = abs(P[:, 2]) + 0.5
            K = torch.tensor([[700.0, 0.0, 350.0],
                             [0.0, 700.0, 250.0],
                             [0.0, 0.0, 1.0]], device=self.device)
            img_size = (500, 700)
            test_cases.append({
                'P': P,
                'K': K,
                'img_size': img_size,
                'jacobian': True,
                'border': 0,
                'z_eps': 0.1,
                'description': 'With z_eps=0.1, with jacobian',
            })

        if num_tests > 4:
            # Test 5: Large batch
            P = torch.randn(10, 3, device=self.device)
            P[:, 2] = abs(P[:, 2]) + 1.0
            K = torch.tensor([[1000.0, 0.0, 500.0],
                             [0.0, 1000.0, 400.0],
                             [0.0, 0.0, 1.0]], device=self.device)
            img_size = (800, 1000)
            test_cases.append({
                'P': P,
                'K': K,
                'img_size': img_size,
                'jacobian': False,
                'border': 5,
                'z_eps': 0.05,
                'description': 'Large batch: 10 points, border=5, z_eps=0.05',
            })

        if num_tests > 5:
            # Test 6: Some points behind camera (should be invalid)
            P = torch.randn(6, 3, device=self.device)
            P[:3, 2] = abs(P[:3, 2]) + 1.0  # First 3 in front
            P[3:, 2] = -abs(P[3:, 2]) - 0.5  # Last 3 behind
            K = torch.tensor([[500.0, 0.0, 320.0],
                             [0.0, 500.0, 240.0],
                             [0.0, 0.0, 1.0]], device=self.device)
            img_size = (480, 640)
            test_cases.append({
                'P': P,
                'K': K,
                'img_size': img_size,
                'jacobian': True,
                'border': 0,
                'z_eps': 0.0,
                'description': 'Mixed: some points behind camera',
            })

        if num_tests > 6:
            # Test 7: Points outside image bounds
            P = torch.randn(3, 3, device=self.device)
            P[:, 2] = abs(P[:, 2]) + 1.0
            P[:, 0] = P[:, 0] * 10  # Large x
            P[:, 1] = P[:, 1] * 10  # Large y
            K = torch.tensor([[500.0, 0.0, 320.0],
                             [0.0, 500.0, 240.0],
                             [0.0, 0.0, 1.0]], device=self.device)
            img_size = (480, 640)
            test_cases.append({
                'P': P,
                'K': K,
                'img_size': img_size,
                'jacobian': False,
                'border': 0,
                'z_eps': 0.0,
                'description': 'Points outside image bounds',
            })

        if num_tests > 7:
            # Test 8: 3D batch shape (e.g., for video sequences)
            P = torch.randn(2, 5, 3, device=self.device)
            P[:, :, 2] = abs(P[:, :, 2]) + 1.0
            K = torch.tensor([[600.0, 0.0, 320.0],
                             [0.0, 600.0, 240.0],
                             [0.0, 0.0, 1.0]], device=self.device)
            img_size = (480, 640)
            test_cases.append({
                'P': P,
                'K': K,
                'img_size': img_size,
                'jacobian': True,
                'border': 0,
                'z_eps': 0.0,
                'description': '3D batch shape: (2, 5, 3)',
            })

        if num_tests > 8:
            # Test 9: Very small z values
            P = torch.randn(4, 3, device=self.device)
            P[:, 2] = torch.rand(4, device=self.device) * 0.05 + 0.01
            K = torch.tensor([[500.0, 0.0, 320.0],
                             [0.0, 500.0, 240.0],
                             [0.0, 0.0, 1.0]], device=self.device)
            img_size = (480, 640)
            test_cases.append({
                'P': P,
                'K': K,
                'img_size': img_size,
                'jacobian': False,
                'border': 0,
                'z_eps': 0.0,
                'description': 'Very small z values',
            })

        if num_tests > 9:
            # Test 10: Different K matrix (non-square pixels)
            P = torch.randn(5, 3, device=self.device)
            P[:, 2] = abs(P[:, 2]) + 1.0
            K = torch.tensor([[800.0, 0.0, 400.0],
                             [0.0, 600.0, 300.0],
                             [0.0, 0.0, 1.0]], device=self.device)
            img_size = (600, 800)
            test_cases.append({
                'P': P,
                'K': K,
                'img_size': img_size,
                'jacobian': True,
                'border': 10,
                'z_eps': 0.1,
                'description': 'Non-square pixels, with border and z_eps',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            batch_size = 3 + (i % 8)
            P = torch.randn(batch_size, 3, device=self.device)
            P[:, 2] = abs(P[:, 2]) + 1.0
            fx = 400.0 + (i % 5) * 100.0
            fy = 400.0 + (i % 5) * 100.0
            K = torch.tensor([[fx, 0.0, 320.0],
                             [0.0, fy, 240.0],
                             [0.0, 0.0, 1.0]], device=self.device)
            img_size = (480, 640)
            jacobian = (i % 2 == 0)
            border = (i % 3) * 5
            z_eps = (i % 3) * 0.05

            test_cases.append({
                'P': P,
                'K': K,
                'img_size': img_size,
                'jacobian': jacobian,
                'border': border,
                'z_eps': z_eps,
                'description': f'Additional test {i+1}: batch={batch_size}, jacobian={jacobian}, border={border}, z_eps={z_eps}',
            })

        return test_cases[:num_tests]
