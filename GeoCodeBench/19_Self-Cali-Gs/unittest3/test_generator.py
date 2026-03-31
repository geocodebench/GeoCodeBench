"""Generate test data for init_cubemap function."""

import torch
import torch.nn as nn


class MockLensNet(nn.Module):
    """Mock lens network for testing."""

    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, sensor_to_frustum=False):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MockDataset:
    """Mock dataset object."""

    def __init__(self, source_path="/tmp/test_dataset"):
        self.source_path = source_path


class TestDataGenerator:
    """Generate test data for init_cubemap function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        dataset = MockDataset("/tmp/test1")
        test_cases.append({
            'dataset': dataset,
            'description': 'Basic: default coefficients [0, 0, 0, 0]',
        })

        if num_tests > 1:
            dataset = MockDataset("/tmp/test2_fish")
            test_cases.append({
                'dataset': dataset,
                'description': 'Fish dataset: path contains "fish"',
            })

        if num_tests > 2:
            torch.manual_seed(123)
            dataset = MockDataset("/tmp/test3")
            test_cases.append({
                'dataset': dataset,
                'description': 'Different random seed: 123',
            })
            torch.manual_seed(self.seed)

        if num_tests > 3:
            dataset = MockDataset("/tmp/test4")
            test_cases.append({
                'dataset': dataset,
                'description': 'Larger network: hidden_dim=128',
                'hidden_dim': 128,
            })

        if num_tests > 4:
            dataset = MockDataset("/tmp/test5")
            test_cases.append({
                'dataset': dataset,
                'description': 'Smaller network: hidden_dim=32',
                'hidden_dim': 32,
            })

        for i in range(num_tests - len(test_cases)):
            dataset = MockDataset(f"/tmp/test{i+6}")
            test_cases.append({
                'dataset': dataset,
                'description': f'Additional test {i+1}: seed {self.seed + i + 1}',
            })

        return test_cases[:num_tests]
