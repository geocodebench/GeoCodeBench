"""
Test Data Generator for sphere2pose function.
"""

import torch


class TestDataGenerator:
    """Generate test data for sphere2pose function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []
        device = torch.device('cpu')  # Use CPU as no CUDA available in test environment

        # Test 1: Basic case with single batch
        batch_size = 1
        c2ws_input = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        c2ws_input[:, :3, 3] = torch.randn(batch_size, 3)  # Random translation
        test_cases.append({
            'c2ws_input': c2ws_input,
            'theta': 10.0,
            'phi': 15.0,
            'r': 1.0,
            'device': device,
            'x': None,
            'y': None,
            'description': f'Basic: batch_size={batch_size}, theta=10°, phi=15°, r=1.0',
        })

        if num_tests > 1:
            # Test 2: Multiple batches
            batch_size = 3
            c2ws_input = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
            c2ws_input[:, :3, 3] = torch.randn(batch_size, 3)
            test_cases.append({
                'c2ws_input': c2ws_input,
                'theta': 20.0,
                'phi': -30.0,
                'r': 2.5,
                'device': device,
                'x': None,
                'y': None,
                'description': f'Multiple batches: batch_size={batch_size}, theta=20°, phi=-30°, r=2.5',
            })

        if num_tests > 2:
            # Test 3: With x and y translations
            batch_size = 2
            c2ws_input = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
            c2ws_input[:, :3, 3] = torch.randn(batch_size, 3)
            test_cases.append({
                'c2ws_input': c2ws_input,
                'theta': 5.0,
                'phi': 10.0,
                'r': 0.5,
                'device': device,
                'x': 0.3,
                'y': 0.4,
                'description': f'With x,y: batch_size={batch_size}, theta=5°, phi=10°, r=0.5, x=0.3, y=0.4',
            })

        if num_tests > 3:
            # Test 4: Zero rotation
            batch_size = 4
            c2ws_input = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
            c2ws_input[:, :3, 3] = torch.randn(batch_size, 3)
            test_cases.append({
                'c2ws_input': c2ws_input,
                'theta': 0.0,
                'phi': 0.0,
                'r': 1.0,
                'device': device,
                'x': None,
                'y': None,
                'description': f'Zero rotation: batch_size={batch_size}, theta=0°, phi=0°, r=1.0',
            })

        if num_tests > 4:
            # Test 5: Large angles
            batch_size = 2
            c2ws_input = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
            c2ws_input[:, :3, 3] = torch.randn(batch_size, 3)
            test_cases.append({
                'c2ws_input': c2ws_input,
                'theta': 45.0,
                'phi': 90.0,
                'r': 3.0,
                'device': device,
                'x': 0.5,
                'y': 0.6,
                'description': f'Large angles: batch_size={batch_size}, theta=45°, phi=90°, r=3.0, x=0.5, y=0.6',
            })

        if num_tests > 5:
            # Test 6: Negative angles
            batch_size = 3
            c2ws_input = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
            c2ws_input[:, :3, 3] = torch.randn(batch_size, 3)
            test_cases.append({
                'c2ws_input': c2ws_input,
                'theta': -25.0,
                'phi': -40.0,
                'r': 1.5,
                'device': device,
                'x': None,
                'y': None,
                'description': f'Negative angles: batch_size={batch_size}, theta=-25°, phi=-40°, r=1.5',
            })

        if num_tests > 6:
            # Test 7: Small batch with rotation
            batch_size = 1
            c2ws_input = torch.eye(4).unsqueeze(0)
            # Add a non-identity rotation
            angle = torch.tensor(30.0)
            cos_a = torch.cos(torch.deg2rad(angle))
            sin_a = torch.sin(torch.deg2rad(angle))
            c2ws_input[0, :3, :3] = torch.tensor([[cos_a, -sin_a, 0],
                                                   [sin_a, cos_a, 0],
                                                   [0, 0, 1]])
            c2ws_input[:, :3, 3] = torch.tensor([[1.0, 2.0, 3.0]])
            test_cases.append({
                'c2ws_input': c2ws_input,
                'theta': 15.0,
                'phi': 20.0,
                'r': 0.8,
                'device': device,
                'x': 0.1,
                'y': 0.2,
                'description': f'Non-identity input: batch_size={batch_size}, with initial rotation',
            })

        if num_tests > 7:
            # Test 8: Large batch
            batch_size = 10
            c2ws_input = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
            c2ws_input[:, :3, 3] = torch.randn(batch_size, 3)
            test_cases.append({
                'c2ws_input': c2ws_input,
                'theta': 12.5,
                'phi': 18.3,
                'r': 2.0,
                'device': device,
                'x': None,
                'y': None,
                'description': f'Large batch: batch_size={batch_size}, theta=12.5°, phi=18.3°',
            })

        if num_tests > 8:
            # Test 9: 180 degree rotation
            batch_size = 2
            c2ws_input = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
            c2ws_input[:, :3, 3] = torch.randn(batch_size, 3)
            test_cases.append({
                'c2ws_input': c2ws_input,
                'theta': 180.0,
                'phi': 0.0,
                'r': 1.0,
                'device': device,
                'x': None,
                'y': None,
                'description': f'180° rotation: batch_size={batch_size}, theta=180°',
            })

        if num_tests > 9:
            # Test 10: Full 3D rotation
            batch_size = 5
            c2ws_input = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
            c2ws_input[:, :3, 3] = torch.randn(batch_size, 3) * 5  # Larger translations
            test_cases.append({
                'c2ws_input': c2ws_input,
                'theta': 60.0,
                'phi': 120.0,
                'r': 4.0,
                'device': device,
                'x': 1.0,
                'y': 1.5,
                'description': f'Full 3D: batch_size={batch_size}, theta=60°, phi=120°, r=4.0, x=1.0, y=1.5',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            batch_size = 2 + (i % 4)
            c2ws_input = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
            c2ws_input[:, :3, 3] = torch.randn(batch_size, 3)

            test_cases.append({
                'c2ws_input': c2ws_input,
                'theta': 10.0 + i * 5.0,
                'phi': 15.0 + i * 7.0,
                'r': 1.0 + i * 0.5,
                'device': device,
                'x': None if i % 2 == 0 else 0.3,
                'y': None if i % 2 == 0 else 0.4,
                'description': f'Additional test {i+1}: batch_size={batch_size}',
            })

        return test_cases[:num_tests]
