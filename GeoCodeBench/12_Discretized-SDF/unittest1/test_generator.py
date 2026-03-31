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

        # Test 1: Basic case with single batch
        batch_size = tuple()  # No batch
        test_cases.append({
            'function': 'bsdf_fresnel_shlick',
            'args': {
                'f0': torch.tensor(0.04),
                'f90': torch.tensor(1.0),
                'cosTheta': torch.tensor(0.5)
            },
            'description': f'Basic Fresnel: f0=0.04, f90=1.0, cosTheta=0.5',
        })

        if num_tests > 1:
            # Test 2: With batch dimension
            batch_size = (3,)
            test_cases.append({
                'function': 'bsdf_ndf_ggx',
                'args': {
                    'alphaSqr': torch.rand(*batch_size, 1) * 0.5 + 0.01,
                    'cosTheta': torch.rand(*batch_size, 1) * 0.8 + 0.1
                },
                'description': f'Batch NDF: batch_size={batch_size}',
            })

        if num_tests > 2:
            # Test 3: Lambda GGX
            batch_size = (2,)
            test_cases.append({
                'function': 'bsdf_lambda_ggx',
                'args': {
                    'alphaSqr': torch.rand(*batch_size, 1) * 0.3 + 0.01,
                    'cosTheta': torch.rand(*batch_size, 1) * 0.9 + 0.05
                },
                'description': f'Lambda GGX: batch_size={batch_size}',
            })

        if num_tests > 3:
            # Test 4: Masking Smith GGX
            batch_size = (4,)
            test_cases.append({
                'function': 'bsdf_masking_smith_ggx_correlated',
                'args': {
                    'alphaSqr': torch.rand(*batch_size, 1) * 0.4 + 0.01,
                    'cosThetaI': torch.rand(*batch_size, 1) * 0.8 + 0.1,
                    'cosThetaO': torch.rand(*batch_size, 1) * 0.8 + 0.1
                },
                'description': f'Masking Smith: batch_size={batch_size}',
            })

        if num_tests > 4:
            # Test 5: PBR Specular
            batch_size = (5, 2)
            test_cases.append({
                'function': 'bsdf_pbr_specular',
                'args': {
                    'col': torch.rand(*batch_size, 3),
                    'nrm': torch.randn(*batch_size, 3),
                    'wo': torch.randn(*batch_size, 3),
                    'wi': torch.randn(*batch_size, 3),
                    'alpha': torch.rand(*batch_size, 1) * 0.5 + 0.1
                },
                'description': f'PBR Specular: batch_size={batch_size}',
            })

        if num_tests > 5:
            # Test 6: Edge cases for Fresnel
            test_cases.append({
                'function': 'bsdf_fresnel_shlick',
                'args': {
                    'f0': torch.tensor([0.04, 0.5, 0.9]),
                    'f90': torch.tensor([1.0, 1.0, 1.0]),
                    'cosTheta': torch.tensor([0.0, 0.5, 1.0])
                },
                'description': f'Edge cases Fresnel: extreme cosTheta values',
            })

        if num_tests > 6:
            # Test 7: Large batch NDF
            batch_size = (10, 5)
            test_cases.append({
                'function': 'bsdf_ndf_ggx',
                'args': {
                    'alphaSqr': torch.rand(*batch_size, 1) * 0.2 + 0.01,
                    'cosTheta': torch.rand(*batch_size, 1) * 0.7 + 0.2
                },
                'description': f'Large batch NDF: batch_size={batch_size}',
            })

        if num_tests > 7:
            # Test 8: Complex PBR Specular
            batch_size = (3, 4, 2)
            test_cases.append({
                'function': 'bsdf_pbr_specular',
                'args': {
                    'col': torch.rand(*batch_size, 3),
                    'nrm': torch.randn(*batch_size, 3),
                    'wo': torch.randn(*batch_size, 3),
                    'wi': torch.randn(*batch_size, 3),
                    'alpha': torch.rand(*batch_size, 1) * 0.3 + 0.05
                },
                'description': f'Complex PBR: batch_size={batch_size}',
            })

        if num_tests > 8:
            # Test 9: Lambda GGX edge cases
            test_cases.append({
                'function': 'bsdf_lambda_ggx',
                'args': {
                    'alphaSqr': torch.tensor([0.01, 0.1, 0.5, 1.0]).unsqueeze(-1),
                    'cosTheta': torch.tensor([0.1, 0.5, 0.8, 0.99]).unsqueeze(-1)
                },
                'description': f'Lambda edge cases: various alpha and cosTheta',
            })

        if num_tests > 9:
            # Test 10: Mixed function test
            test_cases.append({
                'function': 'bsdf_masking_smith_ggx_correlated',
                'args': {
                    'alphaSqr': torch.rand(8, 1) * 0.4 + 0.01,
                    'cosThetaI': torch.rand(8, 1) * 0.9 + 0.05,
                    'cosThetaO': torch.rand(8, 1) * 0.9 + 0.05
                },
                'description': f'Mixed masking: various roughness and angles',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            batch_size = tuple([2 + (i % 3)] * (1 + i % 2))
            function_names = ['bsdf_fresnel_shlick', 'bsdf_ndf_ggx', 'bsdf_lambda_ggx',
                             'bsdf_masking_smith_ggx_correlated', 'bsdf_pbr_specular']
            func_name = function_names[i % len(function_names)]

            if func_name == 'bsdf_fresnel_shlick':
                test_cases.append({
                    'function': func_name,
                    'args': {
                        'f0': torch.rand(*batch_size, 1) * 0.5 + 0.01,
                        'f90': torch.rand(*batch_size, 1) * 0.5 + 0.5,
                        'cosTheta': torch.rand(*batch_size, 1) * 0.8 + 0.1
                    },
                    'description': f'Additional Fresnel {i+1}: batch_size={batch_size}',
                })
            elif func_name == 'bsdf_ndf_ggx':
                test_cases.append({
                    'function': func_name,
                    'args': {
                        'alphaSqr': torch.rand(*batch_size, 1) * 0.3 + 0.01,
                        'cosTheta': torch.rand(*batch_size, 1) * 0.8 + 0.1
                    },
                    'description': f'Additional NDF {i+1}: batch_size={batch_size}',
                })
            elif func_name == 'bsdf_lambda_ggx':
                test_cases.append({
                    'function': func_name,
                    'args': {
                        'alphaSqr': torch.rand(*batch_size, 1) * 0.4 + 0.01,
                        'cosTheta': torch.rand(*batch_size, 1) * 0.9 + 0.05
                    },
                    'description': f'Additional Lambda {i+1}: batch_size={batch_size}',
                })
            elif func_name == 'bsdf_masking_smith_ggx_correlated':
                test_cases.append({
                    'function': func_name,
                    'args': {
                        'alphaSqr': torch.rand(*batch_size, 1) * 0.3 + 0.01,
                        'cosThetaI': torch.rand(*batch_size, 1) * 0.8 + 0.1,
                        'cosThetaO': torch.rand(*batch_size, 1) * 0.8 + 0.1
                    },
                    'description': f'Additional Masking {i+1}: batch_size={batch_size}',
                })
            else:  # bsdf_pbr_specular
                test_cases.append({
                    'function': func_name,
                    'args': {
                        'col': torch.rand(*batch_size, 3),
                        'nrm': torch.randn(*batch_size, 3),
                        'wo': torch.randn(*batch_size, 3),
                        'wi': torch.randn(*batch_size, 3),
                        'alpha': torch.rand(*batch_size, 1) * 0.4 + 0.05
                    },
                    'description': f'Additional PBR {i+1}: batch_size={batch_size}',
                })

        return test_cases[:num_tests]
