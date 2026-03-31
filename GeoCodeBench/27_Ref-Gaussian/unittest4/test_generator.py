"""
Test Data Generator for bsdf_pbr_specular function.
Generates test cases with different configurations.
"""

import torch


class TestDataGenerator:
    """Generate test data for bsdf_pbr_specular function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)
        self.device = torch.device('cpu')  # No CUDA support

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case - single point
        col = torch.tensor([0.8, 0.8, 0.8], device=self.device).unsqueeze(0).unsqueeze(0)  # [1, 1, 3]
        nrm = torch.nn.functional.normalize(torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0).unsqueeze(0), dim=-1)
        wo = torch.nn.functional.normalize(torch.tensor([0.0, 0.5, 0.866], device=self.device).unsqueeze(0).unsqueeze(0), dim=-1)
        wi = torch.nn.functional.normalize(torch.tensor([0.0, -0.5, 0.866], device=self.device).unsqueeze(0).unsqueeze(0), dim=-1)
        alpha = torch.tensor([0.1], device=self.device).unsqueeze(0).unsqueeze(0)  # [1, 1, 1]

        test_cases.append({
            'col': col,
            'nrm': nrm,
            'wo': wo,
            'wi': wi,
            'alpha': alpha,
            'min_roughness': 0.08,
            'description': 'Basic: single point, moderate roughness',
        })

        if num_tests > 1:
            batch_size = 4
            col = torch.rand(batch_size, 1, 3, device=self.device) * 0.5 + 0.3
            nrm = torch.nn.functional.normalize(torch.randn(batch_size, 1, 3, device=self.device), dim=-1)
            wo = torch.nn.functional.normalize(torch.randn(batch_size, 1, 3, device=self.device), dim=-1)
            wi = torch.nn.functional.normalize(torch.randn(batch_size, 1, 3, device=self.device), dim=-1)
            alpha = torch.rand(batch_size, 1, 1, device=self.device) * 0.8 + 0.1

            test_cases.append({
                'col': col,
                'nrm': nrm,
                'wo': wo,
                'wi': wi,
                'alpha': alpha,
                'min_roughness': 0.08,
                'description': f'Batch: {batch_size} points, varying roughness',
            })

        if num_tests > 2:
            batch_size = 8
            col = torch.tensor([1.0, 1.0, 1.0], device=self.device).expand(batch_size, 1, 3)
            nrm = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(batch_size, 1, 3)
            nrm = torch.nn.functional.normalize(nrm, dim=-1)
            wo = torch.nn.functional.normalize(torch.randn(batch_size, 1, 3, device=self.device), dim=-1)
            wi = torch.nn.functional.normalize(torch.randn(batch_size, 1, 3, device=self.device), dim=-1)
            alpha = torch.ones(batch_size, 1, 1, device=self.device) * 0.01

            test_cases.append({
                'col': col,
                'nrm': nrm,
                'wo': wo,
                'wi': wi,
                'alpha': alpha,
                'min_roughness': 0.08,
                'description': f'Smooth surface: {batch_size} points, very low roughness',
            })

        if num_tests > 3:
            batch_size = 6
            col = torch.rand(batch_size, 1, 3, device=self.device)
            nrm = torch.nn.functional.normalize(torch.randn(batch_size, 1, 3, device=self.device), dim=-1)
            wo = torch.nn.functional.normalize(torch.randn(batch_size, 1, 3, device=self.device), dim=-1)
            wi = torch.nn.functional.normalize(torch.randn(batch_size, 1, 3, device=self.device), dim=-1)
            alpha = torch.ones(batch_size, 1, 1, device=self.device) * 0.9

            test_cases.append({
                'col': col,
                'nrm': nrm,
                'wo': wo,
                'wi': wi,
                'alpha': alpha,
                'min_roughness': 0.08,
                'description': f'Rough surface: {batch_size} points, high roughness',
            })

        if num_tests > 4:
            batch_size = 10
            col = torch.rand(batch_size, 1, 3, device=self.device)
            nrm = torch.nn.functional.normalize(torch.randn(batch_size, 1, 3, device=self.device), dim=-1)
            wo = torch.nn.functional.normalize(torch.randn(batch_size, 1, 3, device=self.device), dim=-1)
            wi = torch.nn.functional.normalize(torch.randn(batch_size, 1, 3, device=self.device), dim=-1)
            alpha = torch.rand(batch_size, 1, 1, device=self.device) * 0.6 + 0.2

            test_cases.append({
                'col': col,
                'nrm': nrm,
                'wo': wo,
                'wi': wi,
                'alpha': alpha,
                'min_roughness': 0.08,
                'description': f'Colored materials: {batch_size} points, random colors',
            })

        if num_tests > 5:
            batch_size = 5
            col = torch.rand(batch_size, 1, 3, device=self.device)
            nrm = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(batch_size, 1, 3)
            wo = torch.nn.functional.normalize(torch.tensor([0.9, 0.0, 0.1], device=self.device).expand(batch_size, 1, 3), dim=-1)
            wi = torch.nn.functional.normalize(torch.tensor([0.8, 0.2, 0.15], device=self.device).expand(batch_size, 1, 3), dim=-1)
            alpha = torch.rand(batch_size, 1, 1, device=self.device) * 0.5 + 0.2

            test_cases.append({
                'col': col,
                'nrm': nrm,
                'wo': wo,
                'wi': wi,
                'alpha': alpha,
                'min_roughness': 0.08,
                'description': f'Grazing angles: {batch_size} points, near-parallel directions',
            })

        if num_tests > 6:
            batch_size = 7
            col = torch.rand(batch_size, 1, 3, device=self.device)
            nrm = torch.nn.functional.normalize(torch.randn(batch_size, 1, 3, device=self.device), dim=-1)
            wo = torch.nn.functional.normalize(torch.randn(batch_size, 1, 3, device=self.device), dim=-1)
            wi = torch.nn.functional.normalize(torch.randn(batch_size, 1, 3, device=self.device), dim=-1)
            alpha = torch.rand(batch_size, 1, 1, device=self.device) * 0.05

            test_cases.append({
                'col': col,
                'nrm': nrm,
                'wo': wo,
                'wi': wi,
                'alpha': alpha,
                'min_roughness': 0.1,
                'description': f'Different min_roughness: {batch_size} points, min_roughness=0.1',
            })

        if num_tests > 7:
            batch_size = 4
            col = torch.rand(batch_size, 1, 3, device=self.device)
            nrm = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(batch_size, 1, 3)
            wo = torch.nn.functional.normalize(torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(batch_size, 1, 3), dim=-1)
            wi = torch.nn.functional.normalize(torch.tensor([0.2, 0.2, 0.96], device=self.device).expand(batch_size, 1, 3), dim=-1)
            alpha = torch.rand(batch_size, 1, 1, device=self.device) * 0.5 + 0.2

            test_cases.append({
                'col': col,
                'nrm': nrm,
                'wo': wo,
                'wi': wi,
                'alpha': alpha,
                'min_roughness': 0.08,
                'description': f'Backfacing: {batch_size} points, should return zeros',
            })

        if num_tests > 8:
            batch_size = 32
            col = torch.rand(batch_size, 1, 3, device=self.device)
            nrm = torch.nn.functional.normalize(torch.randn(batch_size, 1, 3, device=self.device), dim=-1)
            wo = torch.nn.functional.normalize(torch.randn(batch_size, 1, 3, device=self.device), dim=-1)
            wi = torch.nn.functional.normalize(torch.randn(batch_size, 1, 3, device=self.device), dim=-1)
            alpha = torch.rand(batch_size, 1, 1, device=self.device)

            test_cases.append({
                'col': col,
                'nrm': nrm,
                'wo': wo,
                'wi': wi,
                'alpha': alpha,
                'min_roughness': 0.08,
                'description': f'Large batch: {batch_size} points',
            })

        if num_tests > 9:
            batch_size = 3
            col = torch.rand(batch_size, 1, 3, device=self.device)
            nrm = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(batch_size, 1, 3)
            direction = torch.nn.functional.normalize(torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(batch_size, 1, 3), dim=-1)
            wo = direction
            wi = direction
            alpha = torch.rand(batch_size, 1, 1, device=self.device) * 0.5 + 0.2

            test_cases.append({
                'col': col,
                'nrm': nrm,
                'wo': wo,
                'wi': wi,
                'alpha': alpha,
                'min_roughness': 0.08,
                'description': f'Edge case: {batch_size} points, wo == wi (perfect reflection)',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            batch_size = 5 + i * 2
            col = torch.rand(batch_size, 1, 3, device=self.device)
            nrm = torch.nn.functional.normalize(torch.randn(batch_size, 1, 3, device=self.device), dim=-1)
            wo = torch.nn.functional.normalize(torch.randn(batch_size, 1, 3, device=self.device), dim=-1)
            wi = torch.nn.functional.normalize(torch.randn(batch_size, 1, 3, device=self.device), dim=-1)
            alpha = torch.rand(batch_size, 1, 1, device=self.device)

            test_cases.append({
                'col': col,
                'nrm': nrm,
                'wo': wo,
                'wi': wi,
                'alpha': alpha,
                'min_roughness': 0.08,
                'description': f'Additional test {i+1}: {batch_size} points, random parameters',
            })

        return test_cases[:num_tests]
