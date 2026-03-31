"""
Test Data Generator for curve_split_computation function.
"""

from __future__ import annotations

import torch


class TestDataGenerator:
    """Generate test data for curve_split_computation function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small number of curves
        num_curves = 5
        n_gaussians = 12
        rotation_matrix = torch.randn(num_curves * n_gaussians, 3, 3, device=self.device)
        for i in range(num_curves * n_gaussians):
            q, r = torch.linalg.qr(rotation_matrix[i])
            rotation_matrix[i] = q
        sample_t = torch.linspace(0.5 / n_gaussians, 1 - 0.5 / n_gaussians, n_gaussians, device=self.device)
        sample_t = sample_t[:, None, None]
        test_cases.append({
            'rotation_matrix': rotation_matrix,
            'sample_t': sample_t,
            'n_gaussians': n_gaussians,
            'threshold_angle': 20,
            'threshold_radian_skip': 30,
            'description': f'Basic: {num_curves} curves, {n_gaussians} gaussians',
        })

        if num_tests > 1:
            num_curves = 20
            n_gaussians = 8
            rotation_matrix = torch.randn(num_curves * n_gaussians, 3, 3, device=self.device)
            for i in range(num_curves * n_gaussians):
                q, r = torch.linalg.qr(rotation_matrix[i])
                rotation_matrix[i] = q
            sample_t = torch.linspace(0.5 / n_gaussians, 1 - 0.5 / n_gaussians, n_gaussians, device=self.device)
            sample_t = sample_t[:, None, None]
            test_cases.append({
                'rotation_matrix': rotation_matrix,
                'sample_t': sample_t,
                'n_gaussians': n_gaussians,
                'threshold_angle': 15,
                'threshold_radian_skip': 25,
                'description': f'More curves: {num_curves} curves, {n_gaussians} gaussians',
            })

        if num_tests > 2:
            num_curves = 10
            n_gaussians = 16
            rotation_matrix = torch.randn(num_curves * n_gaussians, 3, 3, device=self.device)
            for i in range(num_curves * n_gaussians):
                q, r = torch.linalg.qr(rotation_matrix[i])
                rotation_matrix[i] = q
            sample_t = torch.linspace(0.5 / n_gaussians, 1 - 0.5 / n_gaussians, n_gaussians, device=self.device)
            sample_t = sample_t[:, None, None]
            test_cases.append({
                'rotation_matrix': rotation_matrix,
                'sample_t': sample_t,
                'n_gaussians': n_gaussians,
                'threshold_angle': 30,
                'threshold_radian_skip': 40,
                'description': f'多大阈值: {num_curves} curves, thresholds (30, 40)',
            })

        if num_tests > 3:
            num_curves = 8
            n_gaussians = 6
            rotation_matrix = torch.randn(num_curves * n_gaussians, 3, 3, device=self.device)
            for i in range(num_curves * n_gaussians):
                q, r = torch.linalg.qr(rotation_matrix[i])
                rotation_matrix[i] = q
            sample_t = torch.linspace(0.5 / n_gaussians, 1 - 0.5 / n_gaussians, n_gaussians, device=self.device)
            sample_t = sample_t[:, None, None]
            test_cases.append({
                'rotation_matrix': rotation_matrix,
                'sample_t': sample_t,
                'n_gaussians': n_gaussians,
                'threshold_angle': 25,
                'threshold_radian_skip': 35,
                'description': f'Few gaussians: {num_curves} curves, {n_gaussians} gaussians',
            })

        if num_tests > 4:
            num_curves = 50
            n_gaussians = 12
            rotation_matrix = torch.randn(num_curves * n_gaussians, 3, 3, device=self.device)
            for i in range(num_curves * n_gaussians):
                q, r = torch.linalg.qr(rotation_matrix[i])
                rotation_matrix[i] = q
            sample_t = torch.linspace(0.5 / n_gaussians, 1 - 0.5 / n_gaussians, n_gaussians, device=self.device)
            sample_t = sample_t[:, None, None]
            test_cases.append({
                'rotation_matrix': rotation_matrix,
                'sample_t': sample_t,
                'n_gaussians': n_gaussians,
                'threshold_angle': 20,
                'threshold_radian_skip': 30,
                'description': f'Large: {num_curves} curves, {n_gaussians} gaussians',
            })

        for i in range(max(0, num_tests - 5)):
            num_curves = 10 + i * 5
            n_gaussians = 8 + i % 10
            rotation_matrix = torch.randn(num_curves * n_gaussians, 3, 3, device=self.device)
            for j in range(num_curves * n_gaussians):
                q, r = torch.linalg.qr(rotation_matrix[j])
                rotation_matrix[j] = q
            sample_t = torch.linspace(0.5 / n_gaussians, 1 - 0.5 / n_gaussians, n_gaussians, device=self.device)
            sample_t = sample_t[:, None, None]
            test_cases.append({
                'rotation_matrix': rotation_matrix,
                'sample_t': sample_t,
                'n_gaussians': n_gaussians,
                'threshold_angle': 15 + i * 2,
                'threshold_radian_skip': 25 + i * 2,
                'description': f'Additional {i+1}: {num_curves} curves, {n_gaussians} gaussians',
            })

        return test_cases[:num_tests]
