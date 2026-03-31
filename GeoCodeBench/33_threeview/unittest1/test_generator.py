"""
Test Data Generator for run_test_dE23dr12, run_test_dE23dt12, run_test_dE23dr13.
Generates test cases with different configurations.
"""

import numpy as np


class TestDataGenerator:
    """Generate test data for the three test functions."""

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with i=0, default eps
        test_cases.append({
            'function': 'run_test_dE23dr12',
            'i': 0,
            'eps': 1e-8,
            'description': 'dE23dr12: i=0, eps=1e-8',
        })

        if num_tests > 1:
            test_cases.append({
                'function': 'run_test_dE23dr12',
                'i': 1,
                'eps': 1e-8,
                'description': 'dE23dr12: i=1, eps=1e-8',
            })

        if num_tests > 2:
            test_cases.append({
                'function': 'run_test_dE23dr12',
                'i': 2,
                'eps': 1e-8,
                'description': 'dE23dr12: i=2, eps=1e-8',
            })

        if num_tests > 3:
            test_cases.append({
                'function': 'run_test_dE23dt12',
                'i': 0,
                'eps': 1e-8,
                'description': 'dE23dt12: i=0, eps=1e-8',
            })

        if num_tests > 4:
            test_cases.append({
                'function': 'run_test_dE23dt12',
                'i': 1,
                'eps': 1e-8,
                'description': 'dE23dt12: i=1, eps=1e-8',
            })

        if num_tests > 5:
            test_cases.append({
                'function': 'run_test_dE23dt12',
                'i': 2,
                'eps': 1e-8,
                'description': 'dE23dt12: i=2, eps=1e-8',
            })

        if num_tests > 6:
            test_cases.append({
                'function': 'run_test_dE23dr13',
                'i': 0,
                'eps': 1e-8,
                'description': 'dE23dr13: i=0, eps=1e-8',
            })

        if num_tests > 7:
            test_cases.append({
                'function': 'run_test_dE23dr13',
                'i': 1,
                'eps': 1e-8,
                'description': 'dE23dr13: i=1, eps=1e-8',
            })

        if num_tests > 8:
            test_cases.append({
                'function': 'run_test_dE23dr13',
                'i': 2,
                'eps': 1e-8,
                'description': 'dE23dr13: i=2, eps=1e-8',
            })

        if num_tests > 9:
            test_cases.append({
                'function': 'run_test_dE23dr12',
                'i': 0,
                'eps': 1e-9,
                'description': 'dE23dr12: i=0, eps=1e-9',
            })

        func_names = ['run_test_dE23dr12', 'run_test_dE23dt12', 'run_test_dE23dr13']
        for i in range(num_tests - len(test_cases)):
            func_idx = i % 3
            i_val = i % 3
            test_cases.append({
                'function': func_names[func_idx],
                'i': i_val,
                'eps': 1e-8,
                'description': f'{func_names[func_idx]}: i={i_val}, eps=1e-8 (additional)',
            })

        return test_cases[:num_tests]
