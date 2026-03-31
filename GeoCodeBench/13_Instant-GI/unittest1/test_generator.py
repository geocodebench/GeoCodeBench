"""
Test Data Generator for sample_operation function.
Generates various test cases with different configurations.
"""

import torch


class TestDataGenerator:
    """Generate test data for sample_operation function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case - small N
        N = 10
        H, W = 64, 64
        C = 64
        triangles = torch.randn(N, 3, 2) * 0.5
        cir_centers = torch.randn(N, 1, 2) * 0.5
        feature_map = torch.randn(1, C, H, W)
        image = torch.randn(1, 3, H, W)
        test_cases.append({
            'triangles': triangles,
            'cir_centers': cir_centers,
            'feature_map': feature_map,
            'image': image,
            'description': f'Basic: N={N}, H={H}, W={W}, C={C}',
        })

        if num_tests > 1:
            # Test 2: Larger N
            N = 50
            H, W = 128, 128
            C = 64
            triangles = torch.randn(N, 3, 2) * 0.5
            cir_centers = torch.randn(N, 1, 2) * 0.5
            feature_map = torch.randn(1, C, H, W)
            image = torch.randn(1, 3, H, W)
            test_cases.append({
                'triangles': triangles,
                'cir_centers': cir_centers,
                'feature_map': feature_map,
                'image': image,
                'description': f'Large N: N={N}, H={H}, W={W}',
            })

        if num_tests > 2:
            # Test 3: Different feature dimensions
            N = 20
            H, W = 96, 96
            C = 128
            triangles = torch.randn(N, 3, 2) * 0.5
            cir_centers = torch.randn(N, 1, 2) * 0.5
            feature_map = torch.randn(1, C, H, W)
            image = torch.randn(1, 3, H, W)
            test_cases.append({
                'triangles': triangles,
                'cir_centers': cir_centers,
                'feature_map': feature_map,
                'image': image,
                'description': f'Diff C: N={N}, H={H}, C={C}',
            })

        if num_tests > 3:
            # Test 4: Boundary coordinates
            N = 15
            H, W = 64, 64
            C = 64
            # Boundary values with small noise
            base_tri = torch.tensor([[-0.9, -0.9], [-0.9, 0.9], [0.9, 0.9]])
            triangles = base_tri.unsqueeze(0).repeat(N, 1, 1) + torch.randn(N, 3, 2) * 0.1
            cir_centers = torch.randn(N, 1, 2) * 0.9
            feature_map = torch.randn(1, C, H, W)
            image = torch.randn(1, 3, H, W)
            test_cases.append({
                'triangles': triangles,
                'cir_centers': cir_centers,
                'feature_map': feature_map,
                'image': image,
                'description': f'Boundary: near edge coordinates',
            })

        if num_tests > 4:
            # Test 5: Multiple triangles in center
            N = 30
            H, W = 128, 128
            C = 64
            triangles = torch.randn(N, 3, 2) * 0.2
            cir_centers = torch.randn(N, 1, 2) * 0.2
            feature_map = torch.randn(1, C, H, W)
            image = torch.randn(1, 3, H, W)
            test_cases.append({
                'triangles': triangles,
                'cir_centers': cir_centers,
                'feature_map': feature_map,
                'image': image,
                'description': f'Center cluster: N={N}',
            })

        if num_tests > 5:
            # Test 6: Non-square dimensions
            N = 25
            H, W = 96, 128
            C = 64
            triangles = torch.randn(N, 3, 2) * 0.5
            cir_centers = torch.randn(N, 1, 2) * 0.5
            feature_map = torch.randn(1, C, H, W)
            image = torch.randn(1, 3, H, W)
            test_cases.append({
                'triangles': triangles,
                'cir_centers': cir_centers,
                'feature_map': feature_map,
                'image': image,
                'description': f'Non-square: H={H}, W={W}',
            })

        if num_tests > 6:
            # Test 7: Very small images
            N = 5
            H, W = 32, 32
            C = 32
            triangles = torch.randn(N, 3, 2) * 0.5
            cir_centers = torch.randn(N, 1, 2) * 0.5
            feature_map = torch.randn(1, C, H, W)
            image = torch.randn(1, 3, H, W)
            test_cases.append({
                'triangles': triangles,
                'cir_centers': cir_centers,
                'feature_map': feature_map,
                'image': image,
                'description': f'Small: H={H}, W={W}, C={C}',
            })

        if num_tests > 7:
            # Test 8: Very large N
            N = 100
            H, W = 64, 64
            C = 64
            triangles = torch.randn(N, 3, 2) * 0.5
            cir_centers = torch.randn(N, 1, 2) * 0.5
            feature_map = torch.randn(1, C, H, W)
            image = torch.randn(1, 3, H, W)
            test_cases.append({
                'triangles': triangles,
                'cir_centers': cir_centers,
                'feature_map': feature_map,
                'image': image,
                'description': f'Large N: N={N}',
            })

        if num_tests > 8:
            # Test 9: Random seed variation
            N = 20
            H, W = 64, 64
            C = 64
            torch.manual_seed(123)
            triangles = torch.randn(N, 3, 2) * 0.5
            cir_centers = torch.randn(N, 1, 2) * 0.5
            feature_map = torch.randn(1, C, H, W)
            image = torch.randn(1, 3, H, W)
            test_cases.append({
                'triangles': triangles,
                'cir_centers': cir_centers,
                'feature_map': feature_map,
                'image': image,
                'description': f'Seed 123: N={N}',
            })

        if num_tests > 9:
            # Test 10: Different aspect ratios for feature map
            N = 20
            H, W = 128, 96
            C = 96
            torch.manual_seed(42)
            triangles = torch.randn(N, 3, 2) * 0.5
            cir_centers = torch.randn(N, 1, 2) * 0.5
            feature_map = torch.randn(1, C, H, W)
            image = torch.randn(1, 3, H, W)
            test_cases.append({
                'triangles': triangles,
                'cir_centers': cir_centers,
                'feature_map': feature_map,
                'image': image,
                'description': f'Wide feature: C={C}, H={H}, W={W}',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            N = 10 + i * 5
            H = W = 64 + i * 8
            C = 64 + (i % 3) * 32
            torch.manual_seed(42 + i)
            triangles = torch.randn(N, 3, 2) * 0.5
            cir_centers = torch.randn(N, 1, 2) * 0.5
            feature_map = torch.randn(1, C, H, W)
            image = torch.randn(1, 3, H, W)

            test_cases.append({
                'triangles': triangles,
                'cir_centers': cir_centers,
                'feature_map': feature_map,
                'image': image,
                'description': f'Extra test {i+1}: N={N}, H=W={H}, C={C}',
            })

        return test_cases[:num_tests]
