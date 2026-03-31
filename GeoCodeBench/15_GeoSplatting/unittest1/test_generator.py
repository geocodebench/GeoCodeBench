"""
Test Data Generator for bary2gs function.
Generates test cases with different configurations.
"""

from __future__ import annotations

import torch


class TestDataGenerator:
    """Generate test data for bary2gs function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)
        self.device = torch.device('cpu')  # Force CPU

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []
        adapter_g_scale_ratio = 1.6

        # Test 1: Basic case with small number of points
        N = 10
        p0 = torch.randn(N, 3, device=self.device)
        p1 = torch.randn(N, 3, device=self.device)
        area = torch.rand(N, 1, device=self.device).abs() + 0.1  # Ensure positive
        normals = torch.randn(N, 3, device=self.device)
        normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-10)  # Normalize
        max_scale_ratio = 1.0
        test_cases.append({
            'p0': p0,
            'p1': p1,
            'area': area,
            'normals': normals,
            'max_scale_ratio': max_scale_ratio,
            'g_scale_ratio': adapter_g_scale_ratio,
            'description': f'Basic: N={N}, max_scale_ratio={max_scale_ratio}',
        })

        if num_tests > 1:
            # Test 2: Medium size with different max_scale_ratio
            N = 50
            p0 = torch.randn(N, 3, device=self.device)
            p1 = torch.randn(N, 3, device=self.device)
            area = torch.rand(N, 1, device=self.device).abs() + 0.1
            normals = torch.randn(N, 3, device=self.device)
            normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            max_scale_ratio = 2.0
            test_cases.append({
                'p0': p0,
                'p1': p1,
                'area': area,
                'normals': normals,
                'max_scale_ratio': max_scale_ratio,
                'g_scale_ratio': adapter_g_scale_ratio,
                'description': f'Medium: N={N}, max_scale_ratio={max_scale_ratio}',
            })

        if num_tests > 2:
            # Test 3: Large size
            N = 100
            p0 = torch.randn(N, 3, device=self.device)
            p1 = torch.randn(N, 3, device=self.device)
            area = torch.rand(N, 1, device=self.device).abs() + 0.1
            normals = torch.randn(N, 3, device=self.device)
            normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            max_scale_ratio = 0.5
            test_cases.append({
                'p0': p0,
                'p1': p1,
                'area': area,
                'normals': normals,
                'max_scale_ratio': max_scale_ratio,
                'g_scale_ratio': adapter_g_scale_ratio,
                'description': f'Large: N={N}, max_scale_ratio={max_scale_ratio}',
            })

        if num_tests > 3:
            # Test 4: Small max_scale_ratio
            N = 30
            p0 = torch.randn(N, 3, device=self.device) * 2.0  # Larger spread
            p1 = torch.randn(N, 3, device=self.device) * 2.0
            area = torch.rand(N, 1, device=self.device).abs() + 0.5  # Larger area
            normals = torch.randn(N, 3, device=self.device)
            normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            max_scale_ratio = 0.25
            test_cases.append({
                'p0': p0,
                'p1': p1,
                'area': area,
                'normals': normals,
                'max_scale_ratio': max_scale_ratio,
                'g_scale_ratio': adapter_g_scale_ratio,
                'description': f'Small ratio: N={N}, max_scale_ratio={max_scale_ratio}',
            })

        if num_tests > 4:
            # Test 5: Large max_scale_ratio
            N = 75
            p0 = torch.randn(N, 3, device=self.device)
            p1 = torch.randn(N, 3, device=self.device)
            area = torch.rand(N, 1, device=self.device).abs() + 0.01  # Small area
            normals = torch.randn(N, 3, device=self.device)
            normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            max_scale_ratio = 4.0
            test_cases.append({
                'p0': p0,
                'p1': p1,
                'area': area,
                'normals': normals,
                'max_scale_ratio': max_scale_ratio,
                'g_scale_ratio': adapter_g_scale_ratio,
                'description': f'Large ratio: N={N}, max_scale_ratio={max_scale_ratio}',
            })

        if num_tests > 5:
            # Test 6: Very small N (edge case)
            N = 1
            p0 = torch.randn(N, 3, device=self.device)
            p1 = torch.randn(N, 3, device=self.device)
            area = torch.rand(N, 1, device=self.device).abs() + 0.1
            normals = torch.randn(N, 3, device=self.device)
            normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            max_scale_ratio = 1.5
            test_cases.append({
                'p0': p0,
                'p1': p1,
                'area': area,
                'normals': normals,
                'max_scale_ratio': max_scale_ratio,
                'g_scale_ratio': adapter_g_scale_ratio,
                'description': f'Edge case: N={N} (single point)',
            })

        if num_tests > 6:
            # Test 7: Very large N
            N = 200
            p0 = torch.randn(N, 3, device=self.device)
            p1 = torch.randn(N, 3, device=self.device)
            area = torch.rand(N, 1, device=self.device).abs() + 0.1
            normals = torch.randn(N, 3, device=self.device)
            normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            max_scale_ratio = 1.0
            test_cases.append({
                'p0': p0,
                'p1': p1,
                'area': area,
                'normals': normals,
                'max_scale_ratio': max_scale_ratio,
                'g_scale_ratio': adapter_g_scale_ratio,
                'description': f'Very large: N={N}',
            })

        if num_tests > 7:
            # Test 8: Points very close together
            N = 40
            p0 = torch.randn(N, 3, device=self.device)
            p1 = p0 + torch.randn(N, 3, device=self.device) * 0.01  # Very close
            area = torch.rand(N, 1, device=self.device).abs() + 0.001  # Very small area
            normals = torch.randn(N, 3, device=self.device)
            normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            max_scale_ratio = 1.0
            test_cases.append({
                'p0': p0,
                'p1': p1,
                'area': area,
                'normals': normals,
                'max_scale_ratio': max_scale_ratio,
                'g_scale_ratio': adapter_g_scale_ratio,
                'description': f'Close points: N={N}, points very close together',
            })

        if num_tests > 8:
            # Test 9: Points far apart
            N = 60
            p0 = torch.randn(N, 3, device=self.device) * 5.0
            p1 = torch.randn(N, 3, device=self.device) * 5.0
            area = torch.rand(N, 1, device=self.device).abs() + 1.0  # Large area
            normals = torch.randn(N, 3, device=self.device)
            normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            max_scale_ratio = 2.5
            test_cases.append({
                'p0': p0,
                'p1': p1,
                'area': area,
                'normals': normals,
                'max_scale_ratio': max_scale_ratio,
                'g_scale_ratio': adapter_g_scale_ratio,
                'description': f'Far apart: N={N}, points far apart',
            })

        if num_tests > 9:
            # Test 10: Different g_scale_ratio
            N = 25
            p0 = torch.randn(N, 3, device=self.device)
            p1 = torch.randn(N, 3, device=self.device)
            area = torch.rand(N, 1, device=self.device).abs() + 0.1
            normals = torch.randn(N, 3, device=self.device)
            normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            max_scale_ratio = 1.0
            test_cases.append({
                'p0': p0,
                'p1': p1,
                'area': area,
                'normals': normals,
                'max_scale_ratio': max_scale_ratio,
                'g_scale_ratio': 2.0,  # Different g_scale_ratio
                'description': f'Different g_scale: N={N}, g_scale_ratio=2.0',
            })

        # Generate additional tests if needed
        for i in range(len(test_cases), num_tests):
            N = 20 + i * 5
            p0 = torch.randn(N, 3, device=self.device)
            p1 = torch.randn(N, 3, device=self.device)
            area = torch.rand(N, 1, device=self.device).abs() + 0.1
            normals = torch.randn(N, 3, device=self.device)
            normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            max_scale_ratio = 0.5 + i * 0.3

            test_cases.append({
                'p0': p0,
                'p1': p1,
                'area': area,
                'normals': normals,
                'max_scale_ratio': max_scale_ratio,
                'g_scale_ratio': adapter_g_scale_ratio,
                'description': f'Additional test {i+1}: N={N}, max_scale_ratio={max_scale_ratio:.2f}',
            })

        return test_cases[:num_tests]
