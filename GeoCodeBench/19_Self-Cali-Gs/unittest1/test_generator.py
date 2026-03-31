"""Generate test data for init_from_coeff function."""

import torch
from reference_implementation import MockDataset


class TestDataGenerator:
    """Generate test data for init_from_coeff function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with 4 coefficients (fisheye distortion)
        coeff = [0.1, -0.05, 0.02, -0.01]
        dataset = MockDataset('/tmp/test_dataset')
        ref_points = torch.randn(10, 10, 2) * 0.5
        test_cases.append({
            'coeff': coeff,
            'dataset': dataset,
            'ref_points': ref_points.clone(),
            'description': f'Fisheye (4 coeff): shape {list(ref_points.shape)}',
        })

        if num_tests > 1:
            # Test 2: With 2 coefficients (radial distortion)
            coeff = [0.15, -0.1]
            dataset = MockDataset('/tmp/test_dataset')
            ref_points = torch.randn(15, 15, 2) * 0.8
            test_cases.append({
                'coeff': coeff,
                'dataset': dataset,
                'ref_points': ref_points.clone(),
                'description': f'Radial 2 (2 coeff): shape {list(ref_points.shape)}',
            })

        if num_tests > 2:
            # Test 3: With 3 coefficients (extended radial)
            coeff = [0.1, -0.08, 0.03]
            dataset = MockDataset('/tmp/test_dataset')
            ref_points = torch.randn(20, 20, 2) * 1.0
            test_cases.append({
                'coeff': coeff,
                'dataset': dataset,
                'ref_points': ref_points.clone(),
                'description': f'Radial 3 (3 coeff): shape {list(ref_points.shape)}',
            })

        if num_tests > 3:
            # Test 4: With 8 coefficients (full model with tangential)
            coeff = [0.1, -0.05, 0.02, -0.01, 0.0, 0.001, -0.001, 0.0]
            dataset = MockDataset('/tmp/test_dataset')
            ref_points = torch.randn(12, 12, 2) * 0.6
            test_cases.append({
                'coeff': coeff,
                'dataset': dataset,
                'ref_points': ref_points.clone(),
                'description': f'Full model (8 coeff): shape {list(ref_points.shape)}',
            })

        if num_tests > 4:
            # Test 5: Edge case - small values
            coeff = [0.01, -0.005, 0.002, -0.001]
            dataset = MockDataset('/tmp/test_dataset')
            ref_points = torch.randn(8, 8, 2) * 0.1
            test_cases.append({
                'coeff': coeff,
                'dataset': dataset,
                'ref_points': ref_points.clone(),
                'description': f'Small values (4 coeff): shape {list(ref_points.shape)}',
            })

        if num_tests > 5:
            # Test 6: Different shape - 1D batch
            coeff = [0.12, -0.06]
            dataset = MockDataset('/tmp/test_dataset')
            ref_points = torch.randn(100, 2) * 0.7
            test_cases.append({
                'coeff': coeff,
                'dataset': dataset,
                'ref_points': ref_points.clone(),
                'description': f'1D batch (2 coeff): shape {list(ref_points.shape)}',
            })

        if num_tests > 6:
            # Test 7: Larger coefficients
            coeff = [0.2, -0.15, 0.1]
            dataset = MockDataset('/tmp/test_dataset')
            ref_points = torch.randn(16, 16, 2) * 0.9
            test_cases.append({
                'coeff': coeff,
                'dataset': dataset,
                'ref_points': ref_points.clone(),
                'description': f'Large coeff (3 coeff): shape {list(ref_points.shape)}',
            })

        if num_tests > 7:
            # Test 8: Zero coefficients
            coeff = [0.0, 0.0, 0.0, 0.0]
            dataset = MockDataset('/tmp/test_dataset')
            ref_points = torch.randn(10, 10, 2) * 0.5
            test_cases.append({
                'coeff': coeff,
                'dataset': dataset,
                'ref_points': ref_points.clone(),
                'description': f'Zero coeff (4 coeff): shape {list(ref_points.shape)}',
            })

        if num_tests > 8:
            # Test 9: 3D batch
            coeff = [0.1, -0.05]
            dataset = MockDataset('/tmp/test_dataset')
            ref_points = torch.randn(5, 6, 7, 2) * 0.4
            test_cases.append({
                'coeff': coeff,
                'dataset': dataset,
                'ref_points': ref_points.clone(),
                'description': f'3D batch (2 coeff): shape {list(ref_points.shape)}',
            })

        if num_tests > 9:
            # Test 10: Mixed positive/negative coefficients
            coeff = [-0.1, 0.08, -0.05, 0.03, 0.0, -0.002, 0.002, 0.0]
            dataset = MockDataset('/tmp/test_dataset')
            ref_points = torch.randn(14, 14, 2) * 0.65
            test_cases.append({
                'coeff': coeff,
                'dataset': dataset,
                'ref_points': ref_points.clone(),
                'description': f'Mixed signs (8 coeff): shape {list(ref_points.shape)}',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            coeff_len = [2, 3, 4, 8][i % 4]
            if coeff_len == 2:
                coeff = [0.1 + i * 0.01, -0.05 - i * 0.005]
            elif coeff_len == 3:
                coeff = [0.1 + i * 0.01, -0.05 - i * 0.005, 0.02 + i * 0.002]
            elif coeff_len == 4:
                coeff = [0.1 + i * 0.01, -0.05 - i * 0.005, 0.02 + i * 0.002, -0.01 - i * 0.001]
            else:  # 8
                coeff = [0.1, -0.05, 0.02, -0.01, 0.0, 0.001 * (i+1), -0.001 * (i+1), 0.0]

            dataset = MockDataset('/tmp/test_dataset')
            size = 10 + i * 2
            ref_points = torch.randn(size, size, 2) * (0.5 + i * 0.05)

            test_cases.append({
                'coeff': coeff,
                'dataset': dataset,
                'ref_points': ref_points.clone(),
                'description': f'Additional test {i+1} ({coeff_len} coeff): shape {list(ref_points.shape)}',
            })

        return test_cases[:num_tests]
