"""
Test data generator for get_densify_rate function.
"""


class MockScheduler:
    """Mock scheduler to provide necessary attributes."""
    def __init__(self, densify_mode, init_n_gaussian, max_n_gaussian,
                 densify_until_iter, densification_interval,
                 increase_reso_until, max_densify_rate_per_step):
        self.densify_mode = densify_mode
        self.init_n_gaussian = init_n_gaussian
        self.max_n_gaussian = max_n_gaussian
        self.densify_until_iter = densify_until_iter
        self.densification_interval = densification_interval
        self.increase_reso_until = increase_reso_until
        self.max_densify_rate_per_step = max_densify_rate_per_step


class TestDataGenerator:
    """Generate test data for get_densify_rate function."""

    def __init__(self, seed=42):
        self.seed = seed

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Free mode - basic case
        scheduler_config = {
            'densify_mode': 'free',
            'init_n_gaussian': 10000,
            'max_n_gaussian': 50000,
            'densify_until_iter': 15000,
            'densification_interval': 100,
            'increase_reso_until': 15000,
            'max_densify_rate_per_step': 0.2
        }
        test_cases.append({
            'scheduler_config': scheduler_config,
            'iteration': 1000,
            'cur_n_gaussian': 15000,
            'cur_scale': 2,
            'description': 'Free mode: basic case',
        })

        if num_tests > 1:
            # Test 2: Freq mode - early stage (before increase_reso_until)
            scheduler_config = {
                'densify_mode': 'freq',
                'init_n_gaussian': 10000,
                'max_n_gaussian': 50000,
                'densify_until_iter': 15000,
                'densification_interval': 100,
                'increase_reso_until': 15000,
                'max_densify_rate_per_step': 0.2
            }
            test_cases.append({
                'scheduler_config': scheduler_config,
                'iteration': 1000,
                'cur_n_gaussian': 15000,
                'cur_scale': 4,
                'description': 'Freq mode: early stage with scale=4',
            })

        if num_tests > 2:
            # Test 3: Freq mode - after increase_reso_until
            scheduler_config = {
                'densify_mode': 'freq',
                'init_n_gaussian': 10000,
                'max_n_gaussian': 50000,
                'densify_until_iter': 15000,
                'densification_interval': 100,
                'increase_reso_until': 15000,
                'max_densify_rate_per_step': 0.2
            }
            test_cases.append({
                'scheduler_config': scheduler_config,
                'iteration': 16000,
                'cur_n_gaussian': 40000,
                'cur_scale': 1,
                'description': 'Freq mode: after increase_reso_until',
            })

        if num_tests > 3:
            # Test 4: Freq mode - boundary case at increase_reso_until
            scheduler_config = {
                'densify_mode': 'freq',
                'init_n_gaussian': 5000,
                'max_n_gaussian': 30000,
                'densify_until_iter': 10000,
                'densification_interval': 50,
                'increase_reso_until': 10000,
                'max_densify_rate_per_step': 0.15
            }
            test_cases.append({
                'scheduler_config': scheduler_config,
                'iteration': 9950,
                'cur_n_gaussian': 20000,
                'cur_scale': 2,
                'description': 'Freq mode: boundary case at increase_reso_until',
            })

        if num_tests > 4:
            # Test 5: Freq mode - high scale value
            scheduler_config = {
                'densify_mode': 'freq',
                'init_n_gaussian': 10000,
                'max_n_gaussian': 100000,
                'densify_until_iter': 20000,
                'densification_interval': 200,
                'increase_reso_until': 20000,
                'max_densify_rate_per_step': 0.3
            }
            test_cases.append({
                'scheduler_config': scheduler_config,
                'iteration': 5000,
                'cur_n_gaussian': 30000,
                'cur_scale': 8,
                'description': 'Freq mode: high scale value (8)',
            })

        if num_tests > 5:
            # Test 6: Freq mode - small scale
            scheduler_config = {
                'densify_mode': 'freq',
                'init_n_gaussian': 8000,
                'max_n_gaussian': 40000,
                'densify_until_iter': 12000,
                'densification_interval': 150,
                'increase_reso_until': 12000,
                'max_densify_rate_per_step': 0.25
            }
            test_cases.append({
                'scheduler_config': scheduler_config,
                'iteration': 8000,
                'cur_n_gaussian': 25000,
                'cur_scale': 1.5,
                'description': 'Freq mode: small scale (1.5)',
            })

        if num_tests > 6:
            # Test 7: Freq mode - current gaussians close to max
            scheduler_config = {
                'densify_mode': 'freq',
                'init_n_gaussian': 10000,
                'max_n_gaussian': 50000,
                'densify_until_iter': 15000,
                'densification_interval': 100,
                'increase_reso_until': 15000,
                'max_densify_rate_per_step': 0.2
            }
            test_cases.append({
                'scheduler_config': scheduler_config,
                'iteration': 18000,
                'cur_n_gaussian': 48000,
                'cur_scale': 1,
                'description': 'Freq mode: current gaussians close to max',
            })

        if num_tests > 7:
            # Test 8: Freq mode - mid training with moderate scale
            scheduler_config = {
                'densify_mode': 'freq',
                'init_n_gaussian': 12000,
                'max_n_gaussian': 60000,
                'densify_until_iter': 18000,
                'densification_interval': 120,
                'increase_reso_until': 18000,
                'max_densify_rate_per_step': 0.18
            }
            test_cases.append({
                'scheduler_config': scheduler_config,
                'iteration': 9000,
                'cur_n_gaussian': 35000,
                'cur_scale': 3,
                'description': 'Freq mode: mid training with scale=3',
            })

        if num_tests > 8:
            # Test 9: Free mode with different parameters
            scheduler_config = {
                'densify_mode': 'free',
                'init_n_gaussian': 5000,
                'max_n_gaussian': 25000,
                'densify_until_iter': 10000,
                'densification_interval': 80,
                'increase_reso_until': 10000,
                'max_densify_rate_per_step': 0.15
            }
            test_cases.append({
                'scheduler_config': scheduler_config,
                'iteration': 5000,
                'cur_n_gaussian': 15000,
                'cur_scale': None,
                'description': 'Free mode: different parameters',
            })

        if num_tests > 9:
            # Test 10: Freq mode - very early stage
            scheduler_config = {
                'densify_mode': 'freq',
                'init_n_gaussian': 10000,
                'max_n_gaussian': 50000,
                'densify_until_iter': 15000,
                'densification_interval': 100,
                'increase_reso_until': 15000,
                'max_densify_rate_per_step': 0.2
            }
            test_cases.append({
                'scheduler_config': scheduler_config,
                'iteration': 100,
                'cur_n_gaussian': 10500,
                'cur_scale': 6,
                'description': 'Freq mode: very early stage (iteration=100)',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            densify_mode = 'freq' if i % 2 == 0 else 'free'
            scheduler_config = {
                'densify_mode': densify_mode,
                'init_n_gaussian': 10000 + i * 1000,
                'max_n_gaussian': 50000 + i * 5000,
                'densify_until_iter': 15000 + i * 1000,
                'densification_interval': 100 + i * 10,
                'increase_reso_until': 15000 + i * 1000,
                'max_densify_rate_per_step': 0.2 - i * 0.01
            }
            test_cases.append({
                'scheduler_config': scheduler_config,
                'iteration': 5000 + i * 500,
                'cur_n_gaussian': 20000 + i * 2000,
                'cur_scale': 2 + i % 4 if densify_mode == 'freq' else None,
                'description': f'Additional test {i+1}: {densify_mode} mode',
            })

        return test_cases[:num_tests]
