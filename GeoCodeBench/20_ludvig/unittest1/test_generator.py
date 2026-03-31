"""
Test Data Generator for get_stationary function.
Generates test cases with different configurations.
"""

import numpy as np
import torch


class TestDataGenerator:
    """Generate test data for get_stationary function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with symmetrize and normalize
        n = 50  # number of nodes
        k = 10  # number of neighbors
        feature_dim = 16
        num_iterations = 3

        knn_neighbor_indices = torch.randint(0, n, (n, k))
        initial_features = torch.randn(n, feature_dim)

        # Create neighbors array (2, n*k)
        node_indices = np.arange(n).repeat(k)
        neighbors = np.stack((node_indices, knn_neighbor_indices.flatten().numpy()))
        similarities = torch.rand(n * k) * 0.5 + 0.5  # [0.5, 1.0]

        test_cases.append({
            'knn_neighbor_indices': knn_neighbor_indices,
            'initial_features': initial_features,
            'num_iterations': num_iterations,
            'eps': 1e-8,
            'neighbors': neighbors,
            'similarities': similarities,
            'normalize': True,
            'normalize_f': True,
            'f': None,
            'binarize': None,
            'unary_term': None,
            'symmetrize': True,
            'description': f'Basic: n={n}, k={k}, feature_dim={feature_dim}, symmetrize=True, normalize=True',
        })

        if num_tests > 1:
            # Test 2: Without normalization
            n = 40
            k = 8
            feature_dim = 12
            num_iterations = 2

            knn_neighbor_indices = torch.randint(0, n, (n, k))
            initial_features = torch.randn(n, feature_dim)

            node_indices = np.arange(n).repeat(k)
            neighbors = np.stack((node_indices, knn_neighbor_indices.flatten().numpy()))
            similarities = torch.rand(n * k)

            test_cases.append({
                'knn_neighbor_indices': knn_neighbor_indices,
                'initial_features': initial_features,
                'num_iterations': num_iterations,
                'eps': 1e-8,
                'neighbors': neighbors,
                'similarities': similarities,
                'normalize': False,
                'normalize_f': True,
                'f': None,
                'binarize': None,
                'unary_term': None,
                'symmetrize': True,
                'description': f'No normalize: n={n}, k={k}, feature_dim={feature_dim}, normalize=False',
            })

        if num_tests > 2:
            # Test 3: Without symmetrize
            n = 60
            k = 12
            feature_dim = 20
            num_iterations = 4

            knn_neighbor_indices = torch.randint(0, n, (n, k))
            initial_features = torch.randn(n, feature_dim)

            node_indices = np.arange(n).repeat(k)
            neighbors = np.stack((node_indices, knn_neighbor_indices.flatten().numpy()))
            similarities = torch.rand(n * k)

            test_cases.append({
                'knn_neighbor_indices': knn_neighbor_indices,
                'initial_features': initial_features,
                'num_iterations': num_iterations,
                'eps': 1e-8,
                'neighbors': neighbors,
                'similarities': similarities,
                'normalize': True,
                'normalize_f': True,
                'f': None,
                'binarize': None,
                'unary_term': None,
                'symmetrize': False,
                'description': f'No symmetrize: n={n}, k={k}, feature_dim={feature_dim}, symmetrize=False',
            })

        if num_tests > 3:
            # Test 4: With binarize
            n = 30
            k = 6
            feature_dim = 10
            num_iterations = 2

            knn_neighbor_indices = torch.randint(0, n, (n, k))
            initial_features = torch.randn(n, feature_dim)

            node_indices = np.arange(n).repeat(k)
            neighbors = np.stack((node_indices, knn_neighbor_indices.flatten().numpy()))
            similarities = torch.rand(n * k)

            test_cases.append({
                'knn_neighbor_indices': knn_neighbor_indices,
                'initial_features': initial_features,
                'num_iterations': num_iterations,
                'eps': 1e-8,
                'neighbors': neighbors,
                'similarities': similarities,
                'normalize': True,
                'normalize_f': True,
                'f': None,
                'binarize': 0.5,
                'unary_term': None,
                'symmetrize': True,
                'description': f'With binarize: n={n}, k={k}, binarize=0.5',
            })

        if num_tests > 4:
            # Test 5: With unary_term
            n = 45
            k = 10
            feature_dim = 14
            num_iterations = 3

            knn_neighbor_indices = torch.randint(0, n, (n, k))
            initial_features = torch.randn(n, feature_dim)
            unary_term = torch.rand(n, feature_dim) + 0.1

            node_indices = np.arange(n).repeat(k)
            neighbors = np.stack((node_indices, knn_neighbor_indices.flatten().numpy()))
            similarities = torch.rand(n * k)

            test_cases.append({
                'knn_neighbor_indices': knn_neighbor_indices,
                'initial_features': initial_features,
                'num_iterations': num_iterations,
                'eps': 1e-8,
                'neighbors': neighbors,
                'similarities': similarities,
                'normalize': True,
                'normalize_f': True,
                'f': None,
                'binarize': None,
                'unary_term': unary_term,
                'symmetrize': True,
                'description': f'With unary_term: n={n}, k={k}, has unary regularization',
            })

        if num_tests > 5:
            # Test 6: Without normalize_f
            n = 35
            k = 8
            feature_dim = 16
            num_iterations = 2

            knn_neighbor_indices = torch.randint(0, n, (n, k))
            initial_features = torch.randn(n, feature_dim)

            node_indices = np.arange(n).repeat(k)
            neighbors = np.stack((node_indices, knn_neighbor_indices.flatten().numpy()))
            similarities = torch.rand(n * k)

            test_cases.append({
                'knn_neighbor_indices': knn_neighbor_indices,
                'initial_features': initial_features,
                'num_iterations': num_iterations,
                'eps': 1e-8,
                'neighbors': neighbors,
                'similarities': similarities,
                'normalize': True,
                'normalize_f': False,
                'f': None,
                'binarize': None,
                'unary_term': None,
                'symmetrize': True,
                'description': f'No normalize_f: n={n}, k={k}, normalize_f=False',
            })

        if num_tests > 6:
            # Test 7: Custom initial features
            n = 40
            k = 10
            feature_dim = 18
            num_iterations = 4

            knn_neighbor_indices = torch.randint(0, n, (n, k))
            initial_features = torch.randn(n, feature_dim)
            custom_f = torch.randn(n, feature_dim) * 2

            node_indices = np.arange(n).repeat(k)
            neighbors = np.stack((node_indices, knn_neighbor_indices.flatten().numpy()))
            similarities = torch.rand(n * k)

            test_cases.append({
                'knn_neighbor_indices': knn_neighbor_indices,
                'initial_features': initial_features,
                'num_iterations': num_iterations,
                'eps': 1e-8,
                'neighbors': neighbors,
                'similarities': similarities,
                'normalize': True,
                'normalize_f': True,
                'f': custom_f,
                'binarize': None,
                'unary_term': None,
                'symmetrize': True,
                'description': f'Custom f: n={n}, k={k}, custom initial features provided',
            })

        if num_tests > 7:
            # Test 8: Large graph
            n = 100
            k = 15
            feature_dim = 32
            num_iterations = 5

            knn_neighbor_indices = torch.randint(0, n, (n, k))
            initial_features = torch.randn(n, feature_dim)

            node_indices = np.arange(n).repeat(k)
            neighbors = np.stack((node_indices, knn_neighbor_indices.flatten().numpy()))
            similarities = torch.rand(n * k)

            test_cases.append({
                'knn_neighbor_indices': knn_neighbor_indices,
                'initial_features': initial_features,
                'num_iterations': num_iterations,
                'eps': 1e-8,
                'neighbors': neighbors,
                'similarities': similarities,
                'normalize': True,
                'normalize_f': True,
                'f': None,
                'binarize': None,
                'unary_term': None,
                'symmetrize': True,
                'description': f'Large graph: n={n}, k={k}, feature_dim={feature_dim}',
            })

        if num_tests > 8:
            # Test 9: All features combined
            n = 50
            k = 10
            feature_dim = 16
            num_iterations = 3

            knn_neighbor_indices = torch.randint(0, n, (n, k))
            initial_features = torch.randn(n, feature_dim)
            unary_term = torch.rand(n, feature_dim) + 0.1

            node_indices = np.arange(n).repeat(k)
            neighbors = np.stack((node_indices, knn_neighbor_indices.flatten().numpy()))
            similarities = torch.rand(n * k)

            test_cases.append({
                'knn_neighbor_indices': knn_neighbor_indices,
                'initial_features': initial_features,
                'num_iterations': num_iterations,
                'eps': 1e-6,
                'neighbors': neighbors,
                'similarities': similarities,
                'normalize': True,
                'normalize_f': True,
                'f': None,
                'binarize': 0.3,
                'unary_term': unary_term,
                'symmetrize': True,
                'description': f'Complex: n={n}, k={k}, binarize + unary_term',
            })

        if num_tests > 9:
            # Test 10: Minimal case
            n = 20
            k = 5
            feature_dim = 8
            num_iterations = 1

            knn_neighbor_indices = torch.randint(0, n, (n, k))
            initial_features = torch.randn(n, feature_dim)

            node_indices = np.arange(n).repeat(k)
            neighbors = np.stack((node_indices, knn_neighbor_indices.flatten().numpy()))
            similarities = torch.rand(n * k)

            test_cases.append({
                'knn_neighbor_indices': knn_neighbor_indices,
                'initial_features': initial_features,
                'num_iterations': num_iterations,
                'eps': 1e-8,
                'neighbors': neighbors,
                'similarities': similarities,
                'normalize': False,
                'normalize_f': False,
                'f': None,
                'binarize': None,
                'unary_term': None,
                'symmetrize': True,
                'description': f'Minimal: n={n}, k={k}, 1 iteration, no normalization',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            n = 30 + i * 5
            k = 6 + i * 2
            feature_dim = 10 + i * 2
            num_iterations = 2 + i

            knn_neighbor_indices = torch.randint(0, n, (n, k))
            initial_features = torch.randn(n, feature_dim)

            node_indices = np.arange(n).repeat(k)
            neighbors = np.stack((node_indices, knn_neighbor_indices.flatten().numpy()))
            similarities = torch.rand(n * k)

            test_cases.append({
                'knn_neighbor_indices': knn_neighbor_indices,
                'initial_features': initial_features,
                'num_iterations': num_iterations,
                'eps': 1e-8,
                'neighbors': neighbors,
                'similarities': similarities,
                'normalize': i % 2 == 0,
                'normalize_f': True,
                'f': None,
                'binarize': None,
                'unary_term': None,
                'symmetrize': True,
                'description': f'Additional test {i+1}: n={n}, k={k}, feature_dim={feature_dim}',
            })

        return test_cases[:num_tests]
