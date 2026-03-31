"""
Test Data Generator for BSDF functions.
Generates test cases with different configurations.
"""

import torch


class TestDataGenerator:
    """Generate test data for BSDF functions."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic scalar inputs
        test_cases.append({
            'name': 'fresnel_scalar',
            'function': 'bsdf_fresnel_shlick',
            'inputs': {
                'f0': torch.tensor(0.04),
                'f90': torch.tensor(1.0),
                'cosTheta': torch.tensor(0.8)
            },
            'description': 'Fresnel: scalar inputs',
        })

        if num_tests > 1:
            test_cases.append({
                'name': 'fresnel_tensor',
                'function': 'bsdf_fresnel_shlick',
                'inputs': {
                    'f0': torch.rand(10, 3) * 0.1,
                    'f90': torch.ones(10, 3),
                    'cosTheta': torch.rand(10, 3) * 0.9 + 0.1
                },
                'description': 'Fresnel: tensor inputs (10x3)',
            })

        if num_tests > 2:
            test_cases.append({
                'name': 'ndf_basic',
                'function': 'bsdf_ndf_ggx',
                'inputs': {
                    'alphaSqr': torch.tensor(0.25),
                    'cosTheta': torch.tensor(0.9)
                },
                'description': 'NDF GGX: scalar inputs',
            })

        if num_tests > 3:
            test_cases.append({
                'name': 'ndf_tensor',
                'function': 'bsdf_ndf_ggx',
                'inputs': {
                    'alphaSqr': torch.rand(8, 4) * 0.5 + 0.01,
                    'cosTheta': torch.rand(8, 4) * 0.9 + 0.1
                },
                'description': 'NDF GGX: tensor inputs (8x4)',
            })

        if num_tests > 4:
            test_cases.append({
                'name': 'lambda_basic',
                'function': 'bsdf_lambda_ggx',
                'inputs': {
                    'alphaSqr': torch.tensor(0.16),
                    'cosTheta': torch.tensor(0.7)
                },
                'description': 'Lambda GGX: scalar inputs',
            })

        if num_tests > 5:
            test_cases.append({
                'name': 'lambda_tensor',
                'function': 'bsdf_lambda_ggx',
                'inputs': {
                    'alphaSqr': torch.rand(12, 5) * 0.4 + 0.01,
                    'cosTheta': torch.rand(12, 5) * 0.8 + 0.2
                },
                'description': 'Lambda GGX: tensor inputs (12x5)',
            })

        if num_tests > 6:
            test_cases.append({
                'name': 'masking_basic',
                'function': 'bsdf_masking_smith_ggx_correlated',
                'inputs': {
                    'alphaSqr': torch.tensor(0.09),
                    'cosThetaI': torch.tensor(0.85),
                    'cosThetaO': torch.tensor(0.75)
                },
                'description': 'Masking Smith: scalar inputs',
            })

        if num_tests > 7:
            test_cases.append({
                'name': 'masking_tensor',
                'function': 'bsdf_masking_smith_ggx_correlated',
                'inputs': {
                    'alphaSqr': torch.rand(6, 8) * 0.3 + 0.01,
                    'cosThetaI': torch.rand(6, 8) * 0.7 + 0.3,
                    'cosThetaO': torch.rand(6, 8) * 0.7 + 0.3
                },
                'description': 'Masking Smith: tensor inputs (6x8)',
            })

        if num_tests > 8:
            test_cases.append({
                'name': 'fresnel_edge',
                'function': 'bsdf_fresnel_shlick',
                'inputs': {
                    'f0': torch.tensor([0.02, 0.05, 0.1]),
                    'f90': torch.tensor([0.8, 0.9, 1.0]),
                    'cosTheta': torch.tensor([0.05, 0.5, 0.95])
                },
                'description': 'Fresnel: edge cases (near grazing and normal)',
            })

        if num_tests > 9:
            test_cases.append({
                'name': 'ndf_roughness',
                'function': 'bsdf_ndf_ggx',
                'inputs': {
                    'alphaSqr': torch.tensor([0.01, 0.1, 0.25, 0.5, 0.9]),
                    'cosTheta': torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
                },
                'description': 'NDF GGX: various roughness values',
            })

        if num_tests > 10:
            test_cases.append({
                'name': 'lambda_boundary',
                'function': 'bsdf_lambda_ggx',
                'inputs': {
                    'alphaSqr': torch.tensor([0.001, 0.05, 0.2, 0.5, 1.0]),
                    'cosTheta': torch.tensor([0.99, 0.9, 0.7, 0.5, 0.3])
                },
                'description': 'Lambda GGX: boundary values',
            })

        if num_tests > 11:
            test_cases.append({
                'name': 'masking_angles',
                'function': 'bsdf_masking_smith_ggx_correlated',
                'inputs': {
                    'alphaSqr': torch.tensor([0.1, 0.2, 0.3, 0.4]),
                    'cosThetaI': torch.tensor([0.9, 0.7, 0.5, 0.3]),
                    'cosThetaO': torch.tensor([0.8, 0.6, 0.4, 0.2])
                },
                'description': 'Masking Smith: different angle combinations',
            })

        if num_tests > 12:
            batch_size = (20, 10)
            test_cases.append({
                'name': 'fresnel_large_batch',
                'function': 'bsdf_fresnel_shlick',
                'inputs': {
                    'f0': torch.rand(*batch_size) * 0.2,
                    'f90': torch.ones(*batch_size),
                    'cosTheta': torch.rand(*batch_size) * 0.8 + 0.2
                },
                'description': f'Fresnel: large batch {batch_size}',
            })

        if num_tests > 13:
            test_cases.append({
                'name': 'ndf_smooth',
                'function': 'bsdf_ndf_ggx',
                'inputs': {
                    'alphaSqr': torch.rand(10) * 0.01 + 0.001,
                    'cosTheta': torch.rand(10) * 0.3 + 0.7
                },
                'description': 'NDF GGX: smooth surfaces (small alpha)',
            })

        if num_tests > 14:
            test_cases.append({
                'name': 'lambda_rough',
                'function': 'bsdf_lambda_ggx',
                'inputs': {
                    'alphaSqr': torch.rand(15) * 0.5 + 0.5,
                    'cosTheta': torch.rand(15) * 0.6 + 0.4
                },
                'description': 'Lambda GGX: rough surfaces (large alpha)',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            func_idx = i % 4
            if func_idx == 0:
                test_cases.append({
                    'name': f'fresnel_auto_{i}',
                    'function': 'bsdf_fresnel_shlick',
                    'inputs': {
                        'f0': torch.rand(5 + i, 3) * 0.15,
                        'f90': torch.ones(5 + i, 3),
                        'cosTheta': torch.rand(5 + i, 3) * 0.8 + 0.2
                    },
                    'description': f'Fresnel: auto-generated test {i+1}',
                })
            elif func_idx == 1:
                test_cases.append({
                    'name': f'ndf_auto_{i}',
                    'function': 'bsdf_ndf_ggx',
                    'inputs': {
                        'alphaSqr': torch.rand(4 + i, 4) * 0.6 + 0.02,
                        'cosTheta': torch.rand(4 + i, 4) * 0.85 + 0.15
                    },
                    'description': f'NDF GGX: auto-generated test {i+1}',
                })
            elif func_idx == 2:
                test_cases.append({
                    'name': f'lambda_auto_{i}',
                    'function': 'bsdf_lambda_ggx',
                    'inputs': {
                        'alphaSqr': torch.rand(6 + i, 2) * 0.5 + 0.05,
                        'cosTheta': torch.rand(6 + i, 2) * 0.75 + 0.25
                    },
                    'description': f'Lambda GGX: auto-generated test {i+1}',
                })
            else:
                test_cases.append({
                    'name': f'masking_auto_{i}',
                    'function': 'bsdf_masking_smith_ggx_correlated',
                    'inputs': {
                        'alphaSqr': torch.rand(3 + i, 3) * 0.4 + 0.05,
                        'cosThetaI': torch.rand(3 + i, 3) * 0.6 + 0.4,
                        'cosThetaO': torch.rand(3 + i, 3) * 0.6 + 0.4
                    },
                    'description': f'Masking Smith: auto-generated test {i+1}',
                })

        return test_cases[:num_tests]
