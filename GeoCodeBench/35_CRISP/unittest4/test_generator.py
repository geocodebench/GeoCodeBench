"""
Test Data Generator for shape_recovery_from_pc_cvxpy() function.
Includes MockSDFModel and test case generation.
"""

import torch
import torch.nn as nn
import numpy as np


class MockSDFModel(nn.Module):
    """Mock SDF model for testing."""

    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        # Simple linear layer to simulate SDF prediction
        self.linear = nn.Linear(latent_dim + 3, 1)  # shape_code + coords (x,y,z)

    def forward(self, shape_code, coords):
        """
        shape_code: [B, latent_dim] or [latent_dim]
        coords: [B, N, 3]
        Returns: [B, N, 1]
        """
        if shape_code.dim() == 1:
            shape_code = shape_code.unsqueeze(0)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)

        B, N, _ = coords.shape
        latent_dim = shape_code.shape[-1]

        # Expand shape_code to match coords
        shape_code_expanded = shape_code.unsqueeze(1).expand(B, N, latent_dim)

        # Concatenate shape_code and coords
        combined = torch.cat([shape_code_expanded, coords], dim=-1)

        # Pass through linear layer
        output = self.linear(combined)

        return output


class TestDataGenerator:
    """Generate test data for shape_recovery_from_pc_cvxpy() function."""

    def __init__(self, seed=42, device='cpu'):
        self.seed = seed
        self.device = device
        torch.manual_seed(seed)
        np.random.seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case - small batch, no special flags
        test_cases.append({
            'B': 1,
            'N': 10,
            'latent_dim': 32,
            'K': 5,
            'use_L1_reg': False,
            'use_onehot': False,
            'use_initial_shape_code_basis': False,
            'normalize_F_matrix': False,
            'L1_weight': 5,
            'description': 'Basic: B=1, N=10, K=5, all flags False',
        })

        if num_tests > 1:
            # Test 2: With L1 regularization
            test_cases.append({
                'B': 1,
                'N': 15,
                'latent_dim': 32,
                'K': 5,
                'use_L1_reg': True,
                'use_onehot': False,
                'use_initial_shape_code_basis': False,
                'normalize_F_matrix': False,
                'L1_weight': 5,
                'description': 'With L1 reg: B=1, N=15, K=5, use_L1_reg=True',
            })

        if num_tests > 2:
            # Test 3: With onehot
            test_cases.append({
                'B': 1,
                'N': 12,
                'latent_dim': 32,
                'K': 5,
                'use_L1_reg': False,
                'use_onehot': True,
                'use_initial_shape_code_basis': False,
                'normalize_F_matrix': False,
                'L1_weight': 5,
                'description': 'With onehot: B=1, N=12, K=5, use_onehot=True',
            })

        if num_tests > 3:
            # Test 4: With initial shape code basis
            test_cases.append({
                'B': 1,
                'N': 8,
                'latent_dim': 32,
                'K': 5,
                'use_L1_reg': False,
                'use_onehot': False,
                'use_initial_shape_code_basis': True,
                'normalize_F_matrix': False,
                'L1_weight': 5,
                'description': 'With initial basis: B=1, N=8, K=5, use_initial_shape_code_basis=True',
            })

        if num_tests > 4:
            # Test 5: With normalize F matrix
            test_cases.append({
                'B': 1,
                'N': 10,
                'latent_dim': 32,
                'K': 5,
                'use_L1_reg': False,
                'use_onehot': False,
                'use_initial_shape_code_basis': False,
                'normalize_F_matrix': True,
                'L1_weight': 5,
                'description': 'With normalize: B=1, N=10, K=5, normalize_F_matrix=True',
            })

        if num_tests > 5:
            # Test 6: Batch size > 1
            test_cases.append({
                'B': 2,
                'N': 10,
                'latent_dim': 32,
                'K': 5,
                'use_L1_reg': False,
                'use_onehot': False,
                'use_initial_shape_code_basis': False,
                'normalize_F_matrix': False,
                'L1_weight': 5,
                'description': 'Batch=2: B=2, N=10, K=5',
            })

        if num_tests > 6:
            # Test 7: Combined flags
            test_cases.append({
                'B': 1,
                'N': 10,
                'latent_dim': 32,
                'K': 5,
                'use_L1_reg': True,
                'use_onehot': False,
                'use_initial_shape_code_basis': True,
                'normalize_F_matrix': True,
                'L1_weight': 5,
                'description': 'Combined: L1+initial_basis+normalize',
            })

        if num_tests > 7:
            # Test 8: Larger dimensions
            test_cases.append({
                'B': 1,
                'N': 20,
                'latent_dim': 64,
                'K': 8,
                'use_L1_reg': False,
                'use_onehot': False,
                'use_initial_shape_code_basis': False,
                'normalize_F_matrix': False,
                'L1_weight': 5,
                'description': 'Larger dims: N=20, latent_dim=64, K=8',
            })

        if num_tests > 8:
            # Test 9: Different L1 weight
            test_cases.append({
                'B': 1,
                'N': 10,
                'latent_dim': 32,
                'K': 5,
                'use_L1_reg': True,
                'use_onehot': False,
                'use_initial_shape_code_basis': False,
                'normalize_F_matrix': False,
                'L1_weight': 10,
                'description': 'L1 weight=10: use_L1_reg=True, L1_weight=10',
            })

        if num_tests > 9:
            # Test 10: All flags True
            test_cases.append({
                'B': 1,
                'N': 10,
                'latent_dim': 32,
                'K': 5,
                'use_L1_reg': True,
                'use_onehot': True,
                'use_initial_shape_code_basis': True,
                'normalize_F_matrix': True,
                'L1_weight': 5,
                'description': 'All flags True',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            test_cases.append({
                'B': 1 + (i % 3),
                'N': 8 + (i % 15),
                'latent_dim': 32 + (i % 32),
                'K': 3 + (i % 8),
                'use_L1_reg': (i % 2 == 0),
                'use_onehot': (i % 3 == 0),
                'use_initial_shape_code_basis': (i % 4 == 0),
                'normalize_F_matrix': (i % 5 == 0),
                'L1_weight': 5 + (i % 10),
                'description': f'Additional test {i+1}',
            })

        return test_cases[:num_tests]

    def create_test_data(self, test_case):
        """Create test data for a specific test case."""
        B = test_case['B']
        N = test_case['N']
        latent_dim = test_case['latent_dim']
        K = test_case['K']

        # Create mock SDF model
        sdf_model = MockSDFModel(latent_dim=latent_dim).to(self.device)
        sdf_model.eval()

        # Create initial shape code: [B, latent_dim]
        initial_shape_code = torch.randn(B, latent_dim, device=self.device)

        # Create nocs: [B, N, 3] - normalized coordinates
        nocs = torch.randn(B, N, 3, device=self.device)
        nocs = (nocs - nocs.mean(dim=1, keepdim=True)) / (nocs.std(dim=1, keepdim=True) + 1e-8)

        # Create masks: [B, N] - binary masks
        masks = torch.ones(B, N, device=self.device, dtype=torch.float32)

        # Create shape code library: [latent_dim, K]
        shape_code_library = torch.randn(latent_dim, K, device=self.device).cpu().numpy()

        return {
            'sdf_model': sdf_model,
            'initial_shape_code': initial_shape_code,
            'nocs': nocs,
            'masks': masks,
            'shape_code_library': shape_code_library,
        }
