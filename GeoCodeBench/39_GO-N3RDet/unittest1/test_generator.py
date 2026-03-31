"""
Test Data Generator for raw2outputs() function.
Generates test cases with different configurations.
"""

import torch


class TestDataGenerator:
    """Generate test data for raw2outputs() function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small batch
        N_rays = 4
        N_samples = 32
        raw = torch.randn(N_rays, N_samples, 4)
        z_vals = torch.linspace(0, 10, N_samples).unsqueeze(0).repeat(N_rays, 1)
        mask = torch.randint(0, 2, (N_rays, N_samples)).bool()
        white_bkgd = False
        test_cases.append({
            'raw': raw,
            'z_vals': z_vals,
            'mask': mask,
            'white_bkgd': white_bkgd,
            'description': f'Basic: N_rays={N_rays}, N_samples={N_samples}, white_bkgd={white_bkgd}',
        })

        if num_tests > 1:
            # Test 2: Single ray, white_bkgd=True
            N_rays = 1
            N_samples = 16
            raw = torch.randn(N_rays, N_samples, 4)
            z_vals = torch.linspace(0, 5, N_samples).unsqueeze(0).repeat(N_rays, 1)
            mask = None
            white_bkgd = True
            test_cases.append({
                'raw': raw,
                'z_vals': z_vals,
                'mask': mask,
                'white_bkgd': white_bkgd,
                'description': f'Single ray: N_rays={N_rays}, N_samples={N_samples}, white_bkgd={white_bkgd}, mask=None',
            })

        if num_tests > 2:
            # Test 3: Larger batch, with mask
            N_rays = 8
            N_samples = 64
            raw = torch.randn(N_rays, N_samples, 4)
            z_vals = torch.linspace(0, 20, N_samples).unsqueeze(0).repeat(N_rays, 1)
            mask = torch.randint(0, 2, (N_rays, N_samples)).bool()
            white_bkgd = False
            test_cases.append({
                'raw': raw,
                'z_vals': z_vals,
                'mask': mask,
                'white_bkgd': white_bkgd,
                'description': f'Larger batch: N_rays={N_rays}, N_samples={N_samples}, white_bkgd={white_bkgd}',
            })

        if num_tests > 3:
            # Test 4: Medium batch, white_bkgd=True
            N_rays = 5
            N_samples = 48
            raw = torch.randn(N_rays, N_samples, 4)
            z_vals = torch.linspace(0, 15, N_samples).unsqueeze(0).repeat(N_rays, 1)
            mask = torch.randint(0, 2, (N_rays, N_samples)).bool()
            white_bkgd = True
            test_cases.append({
                'raw': raw,
                'z_vals': z_vals,
                'mask': mask,
                'white_bkgd': white_bkgd,
                'description': f'Medium batch: N_rays={N_rays}, N_samples={N_samples}, white_bkgd={white_bkgd}',
            })

        if num_tests > 4:
            # Test 5: Small N_samples
            N_rays = 6
            N_samples = 8
            raw = torch.randn(N_rays, N_samples, 4)
            z_vals = torch.linspace(0, 10, N_samples).unsqueeze(0).repeat(N_rays, 1)
            mask = None
            white_bkgd = False
            test_cases.append({
                'raw': raw,
                'z_vals': z_vals,
                'mask': mask,
                'white_bkgd': white_bkgd,
                'description': f'Small N_samples: N_rays={N_rays}, N_samples={N_samples}, mask=None',
            })

        if num_tests > 5:
            # Test 6: Large batch
            N_rays = 16
            N_samples = 32
            raw = torch.randn(N_rays, N_samples, 4)
            z_vals = torch.linspace(0, 10, N_samples).unsqueeze(0).repeat(N_rays, 1)
            mask = torch.randint(0, 2, (N_rays, N_samples)).bool()
            white_bkgd = False
            test_cases.append({
                'raw': raw,
                'z_vals': z_vals,
                'mask': mask,
                'white_bkgd': white_bkgd,
                'description': f'Large batch: N_rays={N_rays}, N_samples={N_samples}',
            })

        if num_tests > 6:
            # Test 7: Edge case - all zeros input
            N_rays = 3
            N_samples = 16
            raw = torch.zeros(N_rays, N_samples, 4)
            z_vals = torch.linspace(0, 10, N_samples).unsqueeze(0).repeat(N_rays, 1)
            mask = None
            white_bkgd = False
            test_cases.append({
                'raw': raw,
                'z_vals': z_vals,
                'mask': mask,
                'white_bkgd': white_bkgd,
                'description': f'Edge case: N_rays={N_rays}, zero input, mask=None',
            })

        if num_tests > 7:
            # Test 8: Very large N_samples
            N_rays = 4
            N_samples = 128
            raw = torch.randn(N_rays, N_samples, 4)
            z_vals = torch.linspace(0, 10, N_samples).unsqueeze(0).repeat(N_rays, 1)
            mask = torch.randint(0, 2, (N_rays, N_samples)).bool()
            white_bkgd = True
            test_cases.append({
                'raw': raw,
                'z_vals': z_vals,
                'mask': mask,
                'white_bkgd': white_bkgd,
                'description': f'Very large N_samples: N_rays={N_rays}, N_samples={N_samples}, white_bkgd={white_bkgd}',
            })

        if num_tests > 8:
            # Test 9: Different z_vals ranges
            N_rays = 7
            N_samples = 24
            raw = torch.randn(N_rays, N_samples, 4)
            z_vals = torch.linspace(-5, 5, N_samples).unsqueeze(0).repeat(N_rays, 1)
            mask = torch.randint(0, 2, (N_rays, N_samples)).bool()
            white_bkgd = False
            test_cases.append({
                'raw': raw,
                'z_vals': z_vals,
                'mask': mask,
                'white_bkgd': white_bkgd,
                'description': f'Negative z_vals: N_rays={N_rays}, N_samples={N_samples}, z_range=[-5, 5]',
            })

        if num_tests > 9:
            # Test 10: High sigma values
            N_rays = 5
            N_samples = 32
            raw = torch.randn(N_rays, N_samples, 4)
            raw[:, :, 3] = torch.abs(raw[:, :, 3]) * 10  # High sigma
            z_vals = torch.linspace(0, 10, N_samples).unsqueeze(0).repeat(N_rays, 1)
            mask = None
            white_bkgd = True
            test_cases.append({
                'raw': raw,
                'z_vals': z_vals,
                'mask': mask,
                'white_bkgd': white_bkgd,
                'description': f'High sigma: N_rays={N_rays}, N_samples={N_samples}, white_bkgd={white_bkgd}',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            N_rays = 3 + (i % 10)
            N_samples = 16 + (i % 32) * 2
            raw = torch.randn(N_rays, N_samples, 4)
            z_vals = torch.linspace(0, 10, N_samples).unsqueeze(0).repeat(N_rays, 1)
            mask = torch.randint(0, 2, (N_rays, N_samples)).bool() if (i % 2 == 0) else None
            white_bkgd = (i % 3 == 0)

            test_cases.append({
                'raw': raw,
                'z_vals': z_vals,
                'mask': mask,
                'white_bkgd': white_bkgd,
                'description': f'Additional test {i+1}: N_rays={N_rays}, N_samples={N_samples}, white_bkgd={white_bkgd}',
            })

        return test_cases[:num_tests]
