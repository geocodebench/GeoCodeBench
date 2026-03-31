"""
Test Data Generator for get_opacity_with_3D_filter function.
"""

import torch

from reference_implementation import MockGaussianModel


class TestDataGenerator:
    """Generate test data for get_opacity_with_3D_filter function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small number of points
        num_points = 100
        model = MockGaussianModel(num_points)
        test_cases.append({
            'model': model,
            'description': f'Basic: {num_points} points',
        })

        if num_tests > 1:
            # Test 2: Medium number of points
            num_points = 500
            model = MockGaussianModel(num_points)
            test_cases.append({
                'model': model,
                'description': f'Medium: {num_points} points',
            })

        if num_tests > 2:
            # Test 3: Large number of points
            num_points = 2000
            model = MockGaussianModel(num_points)
            test_cases.append({
                'model': model,
                'description': f'Large: {num_points} points',
            })

        if num_tests > 3:
            # Test 4: Very small number of points
            num_points = 10
            model = MockGaussianModel(num_points)
            test_cases.append({
                'model': model,
                'description': f'Small: {num_points} points',
            })

        if num_tests > 4:
            # Test 5: Custom filter values
            num_points = 300
            model = MockGaussianModel(num_points)
            model.filter_3D = torch.ones(num_points, 1, device=model.device) * 0.1
            test_cases.append({
                'model': model,
                'description': f'Uniform filter: {num_points} points',
            })

        if num_tests > 5:
            # Test 6: Varying filter magnitudes
            num_points = 400
            model = MockGaussianModel(num_points)
            model.filter_3D = torch.linspace(0.01, 0.5, num_points, device=model.device).unsqueeze(-1)
            test_cases.append({
                'model': model,
                'description': f'Varying filter: {num_points} points',
            })

        if num_tests > 6:
            # Test 7: Very large scales
            num_points = 250
            model = MockGaussianModel(num_points)
            model._scaling.data = torch.randn(num_points, 3, device=model.device) * 3 + 2  # Larger scales
            test_cases.append({
                'model': model,
                'description': f'Large scales: {num_points} points',
            })

        if num_tests > 7:
            # Test 8: Very small scales
            num_points = 250
            model = MockGaussianModel(num_points)
            model._scaling.data = torch.randn(num_points, 3, device=model.device) * 0.5 - 1  # Smaller scales
            test_cases.append({
                'model': model,
                'description': f'Small scales: {num_points} points',
            })

        if num_tests > 8:
            # Test 9: Very small opacity values
            num_points = 350
            model = MockGaussianModel(num_points)
            model._opacity.data = torch.randn(num_points, 1, device=model.device) - 3  # Low opacity
            test_cases.append({
                'model': model,
                'description': f'Low opacity: {num_points} points',
            })

        if num_tests > 9:
            # Test 10: Very high opacity values
            num_points = 350
            model = MockGaussianModel(num_points)
            model._opacity.data = torch.randn(num_points, 1, device=model.device) + 2  # High opacity
            test_cases.append({
                'model': model,
                'description': f'High opacity: {num_points} points',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            num_points = 100 + i * 50
            model = MockGaussianModel(num_points)
            # Vary the parameters
            if i % 2 == 0:
                model.filter_3D = torch.randn(num_points, 1, device=model.device) * 0.2 + 0.1
            test_cases.append({
                'model': model,
                'description': f'Additional test {i+1}: {num_points} points',
            })

        return test_cases[:num_tests]
