"""
Test data generator for get_res_scale method.
"""


class TestDataGenerator:
    """Generate test data for get_res_scale method."""

    def __init__(self, seed=42):
        self.seed = seed

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Constant resolution mode
        test_cases.append({
            'resolution_mode': 'const',
            'reso_scales': None,
            'reso_level_begin': None,
            'increase_reso_until': 1000,
            'test_iterations': [0, 100, 500, 999, 1000, 1500],
            'description': 'Constant mode: should always return 1',
        })

        if num_tests > 1:
            # Test 2: Frequency mode - simple case (before first level)
            test_cases.append({
                'resolution_mode': 'freq',
                'reso_scales': [8.0, 4.0, 2.0, 1.0],
                'reso_level_begin': [0, 100, 500, 800, 1000],
                'increase_reso_until': 1000,
                'test_iterations': [0, 50, 99],
                'description': 'Freq mode: early iterations (before level 1)',
            })

        if num_tests > 2:
            # Test 3: Frequency mode - after increase_reso_until
            test_cases.append({
                'resolution_mode': 'freq',
                'reso_scales': [8.0, 4.0, 2.0, 1.0],
                'reso_level_begin': [0, 100, 500, 800, 1000],
                'increase_reso_until': 1000,
                'test_iterations': [1000, 1100, 1500, 2000],
                'description': 'Freq mode: after increase_reso_until',
            })

        if num_tests > 3:
            # Test 4: Frequency mode - interpolation between levels
            test_cases.append({
                'resolution_mode': 'freq',
                'reso_scales': [8.0, 6.0, 4.0, 2.0, 1.0],
                'reso_level_begin': [0, 100, 300, 600, 900, 1200],
                'increase_reso_until': 1200,
                'test_iterations': [150, 200, 250, 400, 500, 700, 800, 950, 1000],
                'description': 'Freq mode: interpolation between multiple levels',
            })

        if num_tests > 4:
            # Test 5: Frequency mode - many levels
            test_cases.append({
                'resolution_mode': 'freq',
                'reso_scales': [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0],
                'reso_level_begin': [0, 50, 150, 300, 500, 750, 1050, 1400, 1800, 2000],
                'increase_reso_until': 2000,
                'test_iterations': [0, 25, 100, 200, 400, 600, 900, 1200, 1600, 1900, 2000, 2500],
                'description': 'Freq mode: many resolution levels',
            })

        if num_tests > 5:
            # Test 6: Frequency mode - tight levels
            test_cases.append({
                'resolution_mode': 'freq',
                'reso_scales': [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0],
                'reso_level_begin': [0, 100, 200, 300, 400, 500, 600, 700],
                'increase_reso_until': 700,
                'test_iterations': [0, 50, 150, 250, 350, 450, 550, 650, 700, 800],
                'description': 'Freq mode: closely spaced levels',
            })

        if num_tests > 6:
            # Test 7: Frequency mode - edge case at boundaries
            test_cases.append({
                'resolution_mode': 'freq',
                'reso_scales': [8.0, 4.0, 2.0, 1.0],
                'reso_level_begin': [0, 100, 500, 800, 1000],
                'increase_reso_until': 1000,
                'test_iterations': [100, 500, 800, 999],
                'description': 'Freq mode: exact boundary iterations',
            })

        if num_tests > 7:
            # Test 8: Frequency mode - large scale values
            test_cases.append({
                'resolution_mode': 'freq',
                'reso_scales': [16.0, 12.0, 8.0, 4.0, 2.0, 1.0],
                'reso_level_begin': [0, 200, 500, 1000, 1500, 2000, 2500],
                'increase_reso_until': 2500,
                'test_iterations': [0, 100, 350, 750, 1250, 1750, 2250, 2500],
                'description': 'Freq mode: large scale values',
            })

        if num_tests > 8:
            # Test 9: Frequency mode - fractional scales
            test_cases.append({
                'resolution_mode': 'freq',
                'reso_scales': [5.5, 4.2, 3.1, 2.3, 1.5, 1.0],
                'reso_level_begin': [0, 100, 300, 600, 1000, 1500, 2000],
                'increase_reso_until': 2000,
                'test_iterations': [50, 200, 450, 800, 1250, 1750, 2000],
                'description': 'Freq mode: fractional scale values',
            })

        if num_tests > 9:
            # Test 10: Frequency mode - sequential calls (stateful)
            test_cases.append({
                'resolution_mode': 'freq',
                'reso_scales': [8.0, 6.0, 4.0, 3.0, 2.0, 1.0],
                'reso_level_begin': [0, 100, 300, 600, 900, 1200, 1500],
                'increase_reso_until': 1500,
                'test_iterations': [0, 50, 150, 250, 450, 700, 850, 1000, 1300, 1500],
                'description': 'Freq mode: sequential stateful calls',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            # Create varied test cases
            num_levels = 4 + (i % 4)
            max_iter = 1000 + i * 200
            scales = [8.0 / (j + 1) for j in range(num_levels)]
            scales[-1] = 1.0
            begins = [int(max_iter * j / num_levels) for j in range(num_levels + 1)]

            iterations = [begins[0], begins[1] - 10, begins[1] + 50]
            for j in range(1, num_levels):
                iterations.append(begins[j] + (begins[j+1] - begins[j]) // 2)
            iterations.extend([begins[-1], begins[-1] + 100])

            test_cases.append({
                'resolution_mode': 'freq',
                'reso_scales': scales,
                'reso_level_begin': begins,
                'increase_reso_until': begins[-1],
                'test_iterations': iterations,
                'description': f'Additional test {i+1}: {num_levels} levels, max_iter={max_iter}',
            })

        return test_cases[:num_tests]
