"""
Test Data Generator for unmixField.get_outputs() function.
Generates various test cases with different configurations.
"""

import torch

from reference_implementation import MockRaySamples


class TestDataGenerator:
    """Generate test data for unmixField.get_outputs() function."""

    def __init__(self, seed=42, device='cpu'):
        self.seed = seed
        self.device = device
        torch.manual_seed(seed)

    def create_mock_ray_samples(self, batch_size=1, num_samples=64, num_cameras=1):
        """Create mock RaySamples object."""
        # Create directions: (batch_size, num_samples, 3)
        directions = torch.randn(batch_size, num_samples, 3, device=self.device)
        directions = directions / (directions.norm(dim=-1, keepdim=True) + 1e-8)

        # Create positions: (batch_size, num_samples, 3)
        positions = torch.randn(batch_size, num_samples, 3, device=self.device) * 2.0 - 1.0

        # Create camera indices: (batch_size, num_samples, 1)
        camera_indices = torch.randint(0, num_cameras, (batch_size, num_samples, 1), device=self.device)

        return MockRaySamples(directions, positions, camera_indices)

    def generate_test_suite(self, num_tests=5, field_model=None):
        """Generate test cases with different configurations."""
        test_cases = []

        # Get field dimensions from model if available
        if field_model is not None:
            geo_feat_dim = field_model.geo_feat_dim
        else:
            geo_feat_dim = 256  # default

        # Test 1: Basic case with small batch
        batch_size = 1
        num_samples = 32
        ray_samples = self.create_mock_ray_samples(batch_size, num_samples)
        density_embedding = torch.randn(batch_size, num_samples, geo_feat_dim, device=self.device)
        test_cases.append({
            'ray_samples': ray_samples,
            'density_embedding': density_embedding,
            'description': f'Basic: batch={batch_size}, samples={num_samples}',
        })

        if num_tests > 1:
            # Test 2: More samples (keep batch_size=1 for current reference code path)
            batch_size = 1
            num_samples = 48
            ray_samples = self.create_mock_ray_samples(batch_size, num_samples)
            density_embedding = torch.randn(batch_size, num_samples, geo_feat_dim, device=self.device)
            test_cases.append({
                'ray_samples': ray_samples,
                'density_embedding': density_embedding,
                'description': f'More samples: batch={batch_size}, samples={num_samples}',
            })

        if num_tests > 2:
            # Test 3: Single sample per ray
            batch_size = 1
            num_samples = 1
            ray_samples = self.create_mock_ray_samples(batch_size, num_samples)
            density_embedding = torch.randn(batch_size, num_samples, geo_feat_dim, device=self.device)
            test_cases.append({
                'ray_samples': ray_samples,
                'density_embedding': density_embedding,
                'description': f'Single sample: batch={batch_size}, samples={num_samples}',
            })

        if num_tests > 3:
            # Test 4: Many samples
            batch_size = 1
            num_samples = 128
            ray_samples = self.create_mock_ray_samples(batch_size, num_samples)
            density_embedding = torch.randn(batch_size, num_samples, geo_feat_dim, device=self.device)
            test_cases.append({
                'ray_samples': ray_samples,
                'density_embedding': density_embedding,
                'description': f'Many samples: batch={batch_size}, samples={num_samples}',
            })

        if num_tests > 4:
            # Test 5: Medium sample count
            batch_size = 1
            num_samples = 64
            ray_samples = self.create_mock_ray_samples(batch_size, num_samples)
            density_embedding = torch.randn(batch_size, num_samples, geo_feat_dim, device=self.device)
            test_cases.append({
                'ray_samples': ray_samples,
                'density_embedding': density_embedding,
                'description': f'Medium samples: batch={batch_size}, samples={num_samples}',
            })

        if num_tests > 5:
            # Test 6: Edge case - all zeros density
            batch_size = 1
            num_samples = 32
            ray_samples = self.create_mock_ray_samples(batch_size, num_samples)
            density_embedding = torch.zeros(batch_size, num_samples, geo_feat_dim, device=self.device)
            test_cases.append({
                'ray_samples': ray_samples,
                'density_embedding': density_embedding,
                'description': f'Zero density: batch={batch_size}, samples={num_samples}',
            })

        if num_tests > 6:
            # Test 7: Small values
            batch_size = 1
            num_samples = 16
            ray_samples = self.create_mock_ray_samples(batch_size, num_samples)
            density_embedding = torch.randn(batch_size, num_samples, geo_feat_dim, device=self.device) * 0.01
            test_cases.append({
                'ray_samples': ray_samples,
                'density_embedding': density_embedding,
                'description': f'Small values: batch={batch_size}, samples={num_samples}',
            })

        if num_tests > 7:
            # Test 8: Large sample count
            batch_size = 1
            num_samples = 32
            ray_samples = self.create_mock_ray_samples(batch_size, num_samples)
            density_embedding = torch.randn(batch_size, num_samples, geo_feat_dim, device=self.device)
            test_cases.append({
                'ray_samples': ray_samples,
                'density_embedding': density_embedding,
                'description': f'Large sample count: batch={batch_size}, samples={num_samples}',
            })

        if num_tests > 8:
            # Test 9: Normalized density
            batch_size = 1
            num_samples = 48
            ray_samples = self.create_mock_ray_samples(batch_size, num_samples)
            density_embedding = torch.randn(batch_size, num_samples, geo_feat_dim, device=self.device)
            density_embedding = (density_embedding - density_embedding.mean()) / (density_embedding.std() + 1e-8)
            test_cases.append({
                'ray_samples': ray_samples,
                'density_embedding': density_embedding,
                'description': f'Normalized density: batch={batch_size}, samples={num_samples}',
            })

        if num_tests > 9:
            # Test 10: Different camera indices
            batch_size = 1
            num_samples = 64
            ray_samples = self.create_mock_ray_samples(batch_size, num_samples, num_cameras=5)
            density_embedding = torch.randn(batch_size, num_samples, geo_feat_dim, device=self.device)
            test_cases.append({
                'ray_samples': ray_samples,
                'density_embedding': density_embedding,
                'description': f'Multiple cameras: batch={batch_size}, samples={num_samples}',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            batch_size = 1
            num_samples = 16 + (i % 64) * 2
            ray_samples = self.create_mock_ray_samples(batch_size, num_samples)
            density_embedding = torch.randn(batch_size, num_samples, geo_feat_dim, device=self.device)

            test_cases.append({
                'ray_samples': ray_samples,
                'density_embedding': density_embedding,
                'description': f'Additional test {i+1}: batch={batch_size}, samples={num_samples}',
            })

        return test_cases[:num_tests]
