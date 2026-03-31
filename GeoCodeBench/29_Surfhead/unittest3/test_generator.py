"""
Test Data Generator for RenderingEquationEncoding.forward() function.
"""

import torch
import numpy as np


class TestDataGenerator:
    """Generate test data for RenderingEquationEncoding.forward() function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with 'asg' type
        N = 10
        num_theta = 4
        num_phi = 8
        omega_o = torch.randn(N, 3)
        omega_o = omega_o / torch.norm(omega_o, dim=-1, keepdim=True)
        a = torch.randn(N, num_theta, num_phi, 2)
        la = torch.randn(N, num_theta, num_phi, 1)
        mu = torch.randn(N, num_theta, num_phi, 1)
        test_cases.append({
            'omega_o': omega_o,
            'a': a,
            'la': la,
            'mu': mu,
            'sg_type': 'asg',
            'num_theta': num_theta,
            'num_phi': num_phi,
            'description': f"Basic 'asg': N={N}, theta={num_theta}, phi={num_phi}",
        })

        if num_tests > 1:
            N = 15
            num_theta = 2
            num_phi = 8
            omega_o = torch.randn(N, 3)
            omega_o = omega_o / torch.norm(omega_o, dim=-1, keepdim=True)
            a = torch.randn(N, num_theta, num_phi, 2)
            la = torch.randn(N, num_theta, num_phi, 1)
            mu = torch.randn(N, num_theta, num_phi, 1)
            test_cases.append({
                'omega_o': omega_o,
                'a': a,
                'la': la,
                'mu': mu,
                'sg_type': 'lasg',
                'num_theta': num_theta,
                'num_phi': num_phi,
                'description': f"Test 'lasg': N={N}, theta={num_theta}, phi={num_phi}",
            })

        if num_tests > 2:
            N = 20
            num_theta = 4
            num_phi = 8
            omega_o = torch.randn(N, 3)
            omega_o = omega_o / torch.norm(omega_o, dim=-1, keepdim=True)
            a = torch.randn(N, num_theta, num_phi, 2)
            la = torch.randn(N, num_theta, num_phi, 1)
            mu = torch.randn(N, num_theta, num_phi, 1)
            test_cases.append({
                'omega_o': omega_o,
                'a': a,
                'la': la,
                'mu': mu,
                'sg_type': 'sg',
                'num_theta': num_theta,
                'num_phi': num_phi,
                'description': f"Test 'sg': N={N}, theta={num_theta}, phi={num_phi}",
            })

        if num_tests > 3:
            N = 12
            num_theta = 4
            num_phi = 8
            omega_o = torch.randn(N, 3)
            omega_o = omega_o / torch.norm(omega_o, dim=-1, keepdim=True)
            a = torch.randn(N, num_theta, num_phi, 2)
            la = torch.randn(N, num_theta, num_phi, 1)
            mu = torch.randn(N, num_theta, num_phi, 1)
            test_cases.append({
                'omega_o': omega_o,
                'a': a,
                'la': la,
                'mu': mu,
                'sg_type': 'sg_angle',
                'num_theta': num_theta,
                'num_phi': num_phi,
                'description': f"Test 'sg_angle': N={N}, theta={num_theta}, phi={num_phi}",
            })

        if num_tests > 4:
            N = 50
            num_theta = 4
            num_phi = 8
            omega_o = torch.randn(N, 3)
            omega_o = omega_o / torch.norm(omega_o, dim=-1, keepdim=True)
            a = torch.randn(N, num_theta, num_phi, 2)
            la = torch.randn(N, num_theta, num_phi, 1)
            mu = torch.randn(N, num_theta, num_phi, 1)
            test_cases.append({
                'omega_o': omega_o,
                'a': a,
                'la': la,
                'mu': mu,
                'sg_type': 'asg',
                'num_theta': num_theta,
                'num_phi': num_phi,
                'description': f"Large batch 'asg': N={N}, theta={num_theta}, phi={num_phi}",
            })

        if num_tests > 5:
            N = 1
            num_theta = 4
            num_phi = 8
            omega_o = torch.randn(N, 3)
            omega_o = omega_o / torch.norm(omega_o, dim=-1, keepdim=True)
            a = torch.randn(N, num_theta, num_phi, 2)
            la = torch.randn(N, num_theta, num_phi, 1)
            mu = torch.randn(N, num_theta, num_phi, 1)
            test_cases.append({
                'omega_o': omega_o,
                'a': a,
                'la': la,
                'mu': mu,
                'sg_type': 'sg',
                'num_theta': num_theta,
                'num_phi': num_phi,
                'description': f"Edge case 'sg': N={N} (single sample)",
            })

        if num_tests > 6:
            N = 25
            num_theta = 6
            num_phi = 12
            omega_o = torch.randn(N, 3)
            omega_o = omega_o / torch.norm(omega_o, dim=-1, keepdim=True)
            a = torch.randn(N, num_theta, num_phi, 2)
            la = torch.randn(N, num_theta, num_phi, 1)
            mu = torch.randn(N, num_theta, num_phi, 1)
            test_cases.append({
                'omega_o': omega_o,
                'a': a,
                'la': la,
                'mu': mu,
                'sg_type': 'asg',
                'num_theta': num_theta,
                'num_phi': num_phi,
                'description': f"Different dims 'asg': N={N}, theta={num_theta}, phi={num_phi}",
            })

        if num_tests > 7:
            N = 8
            num_theta = 4
            num_phi = 8
            omega_o = torch.randn(N, 3)
            omega_o = omega_o / torch.norm(omega_o, dim=-1, keepdim=True)
            a = torch.randn(N, num_theta, num_phi, 2) * 10
            la = torch.randn(N, num_theta, num_phi, 1) * 5
            mu = torch.randn(N, num_theta, num_phi, 1) * 5
            test_cases.append({
                'omega_o': omega_o,
                'a': a,
                'la': la,
                'mu': mu,
                'sg_type': 'sg_angle',
                'num_theta': num_theta,
                'num_phi': num_phi,
                'description': f"Extreme values 'sg_angle': N={N}, larger parameter values",
            })

        if num_tests > 8:
            N = 30
            num_theta = 2
            num_phi = 8
            omega_o = torch.randn(N, 3)
            omega_o = omega_o / torch.norm(omega_o, dim=-1, keepdim=True)
            a = torch.randn(N, num_theta, num_phi, 2)
            la = torch.randn(N, num_theta, num_phi, 1)
            mu = torch.randn(N, num_theta, num_phi, 1)
            test_cases.append({
                'omega_o': omega_o,
                'a': a,
                'la': la,
                'mu': mu,
                'sg_type': 'lasg',
                'num_theta': num_theta,
                'num_phi': num_phi,
                'description': f"Medium batch 'lasg': N={N}, theta={num_theta}, phi={num_phi}",
            })

        if num_tests > 9:
            N = 18
            num_theta = 4
            num_phi = 8
            omega_o = torch.randn(N, 3)
            omega_o = omega_o / torch.norm(omega_o, dim=-1, keepdim=True)
            a = torch.randn(N, num_theta, num_phi, 2) * 0.5
            la = torch.randn(N, num_theta, num_phi, 1) * 0.5
            mu = torch.randn(N, num_theta, num_phi, 1) * 0.5
            test_cases.append({
                'omega_o': omega_o,
                'a': a,
                'la': la,
                'mu': mu,
                'sg_type': 'sg',
                'num_theta': num_theta,
                'num_phi': num_phi,
                'description': f"Small values 'sg': N={N}, smaller parameter values",
            })

        for i in range(num_tests - len(test_cases)):
            sg_types = ['asg', 'lasg', 'sg', 'sg_angle']
            sg_type = sg_types[i % len(sg_types)]
            N = 10 + i * 3
            num_theta = 2 if sg_type == 'lasg' else 4
            num_phi = 8
            omega_o = torch.randn(N, 3)
            omega_o = omega_o / torch.norm(omega_o, dim=-1, keepdim=True)
            a = torch.randn(N, num_theta, num_phi, 2)
            la = torch.randn(N, num_theta, num_phi, 1)
            mu = torch.randn(N, num_theta, num_phi, 1)
            test_cases.append({
                'omega_o': omega_o,
                'a': a,
                'la': la,
                'mu': mu,
                'sg_type': sg_type,
                'num_theta': num_theta,
                'num_phi': num_phi,
                'description': f"Additional test {i+1}: '{sg_type}', N={N}",
            })

        return test_cases[:num_tests]
