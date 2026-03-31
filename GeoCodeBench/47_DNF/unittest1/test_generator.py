"""
Test Data Generator for Dict_pose_acc.forward() function.
Generates test cases with different configurations.
"""

from __future__ import annotations

import torch


class TestDataGenerator:
    """Generate test data for Dict_pose_acc.forward() function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_model_params(self, num_layers=4, input_dim=64, hidden_dim=128, rank=2, half=2, positional_enc=False, n_positional_freqs=8, latent_in=[4]):
        """Generate model parameters (U, Vt, bias) for initialization."""
        U = []
        Vt = []
        bias = []

        # Calculate positional encoding output dimension if used
        if positional_enc:
            # NeRF positional encoding: 3 * (2 * n_positional_freqs + 1)
            pos_embed_dim = 3 * (2 * n_positional_freqs + 1)
            # First layer input: input_dim (without xyz) + pos_embed_dim
            first_layer_input = input_dim + pos_embed_dim
        else:
            # First layer input: input_dim (without xyz) + 3 (xyz) = input_dim + 3
            first_layer_input = input_dim + 3

        for i in range(num_layers):
            # Vt: input_dim -> hidden_dim
            if i == 0:
                vt_in = first_layer_input
            else:
                # For subsequent layers, if i in latent_in, input will be concatenated
                if i in latent_in:
                    # Input will be: hidden_dim (from previous layer) + first_layer_input (concatenated)
                    vt_in = hidden_dim + first_layer_input
                else:
                    vt_in = hidden_dim

            vt_weight = torch.randn(hidden_dim, vt_in) * 0.1
            Vt.append(vt_weight)

            # U: hidden_dim -> hidden_dim (or output_dim for last layer)
            if i == num_layers - 1:
                u_out = 3  # Output dimension for xyz
            else:
                u_out = hidden_dim

            u_weight = torch.randn(u_out, hidden_dim) * 0.1
            U.append(u_weight)

            # Bias for U
            u_bias = torch.randn(u_out) * 0.01
            bias.append(u_bias)

        return U, Vt, bias

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case - small batch, no positional encoding, no residual
        num_layers = 4
        input_dim = 64
        hidden_dim = 128
        rank = 0  # No residual
        half = 2
        bs = 2
        num_points = 10
        positional_enc = False

        latent_in = [4]  # Default latent_in
        U, Vt, bias = self.generate_model_params(num_layers, input_dim, hidden_dim, rank, half, positional_enc, 8, latent_in)
        input_tensor = torch.randn(num_points * bs, input_dim + 3)  # +3 for xyz
        Sigma = torch.randn(bs, num_layers, hidden_dim) * 0.1 - 2.0  # Negative values for exp

        test_cases.append({
            'rank': rank,
            'half': half,
            'U': U,
            'Vt': Vt,
            'bias': bias,
            'input': input_tensor,
            'Sigma': Sigma,
            'bs': bs,
            'positional_enc': False,
            'latent_in': latent_in,
            'description': f'Basic: {num_layers} layers, bs={bs}, no positional encoding, no residual',
        })

        if num_tests > 1:
            # Test 2: With residual connections
            rank = 2
            latent_in = [4]  # Default latent_in
            U, Vt, bias = self.generate_model_params(num_layers, input_dim, hidden_dim, rank, half, positional_enc, 8, latent_in)
            input_tensor = torch.randn(num_points * bs, input_dim + 3)
            # Sigma needs extra dimension for residual
            Sigma = torch.randn(bs, num_layers, hidden_dim + rank) * 0.1 - 2.0

            test_cases.append({
                'rank': rank,
                'half': half,
                'U': U,
                'Vt': Vt,
                'bias': bias,
                'input': input_tensor,
                'Sigma': Sigma,
                'bs': bs,
                'positional_enc': False,
                'latent_in': latent_in,
                'description': f'With residual: {num_layers} layers, bs={bs}, rank={rank}',
            })

        if num_tests > 2:
            # Test 3: With positional encoding
            rank = 0
            positional_enc = True
            n_positional_freqs = 8
            latent_in = [4]  # Default latent_in
            U, Vt, bias = self.generate_model_params(num_layers, input_dim, hidden_dim, rank, half, positional_enc, n_positional_freqs, latent_in)
            input_tensor = torch.randn(num_points * bs, input_dim + 3)
            Sigma = torch.randn(bs, num_layers, hidden_dim) * 0.1 - 2.0

            test_cases.append({
                'rank': rank,
                'half': half,
                'U': U,
                'Vt': Vt,
                'bias': bias,
                'input': input_tensor,
                'Sigma': Sigma,
                'bs': bs,
                'positional_enc': True,
                'n_positional_freqs': 8,
                'n_alpha_epochs': 10,
                'latent_in': latent_in,
                'description': f'With positional encoding: {num_layers} layers, bs={bs}',
            })

        if num_tests > 3:
            # Test 4: Larger batch size
            bs = 4
            num_points = 8
            positional_enc = False
            latent_in = [4]  # Default latent_in
            U, Vt, bias = self.generate_model_params(num_layers, input_dim, hidden_dim, rank, half, positional_enc, 8, latent_in)
            input_tensor = torch.randn(num_points * bs, input_dim + 3)
            Sigma = torch.randn(bs, num_layers, hidden_dim) * 0.1 - 2.0

            test_cases.append({
                'rank': rank,
                'half': half,
                'U': U,
                'Vt': Vt,
                'bias': bias,
                'input': input_tensor,
                'Sigma': Sigma,
                'bs': bs,
                'positional_enc': False,
                'latent_in': latent_in,
                'description': f'Larger batch: {num_layers} layers, bs={bs}, num_points={num_points}',
            })

        if num_tests > 4:
            # Test 5: More layers
            num_layers = 6
            half = 3
            rank = 2
            bs = 2
            num_points = 10
            positional_enc = False
            latent_in = [4]  # Default latent_in
            U, Vt, bias = self.generate_model_params(num_layers, input_dim, hidden_dim, rank, half, positional_enc, 8, latent_in)
            input_tensor = torch.randn(num_points * bs, input_dim + 3)
            Sigma = torch.randn(bs, num_layers, hidden_dim + rank) * 0.1 - 2.0

            test_cases.append({
                'rank': rank,
                'half': half,
                'U': U,
                'Vt': Vt,
                'bias': bias,
                'input': input_tensor,
                'Sigma': Sigma,
                'bs': bs,
                'positional_enc': False,
                'latent_in': latent_in,
                'description': f'More layers: {num_layers} layers, bs={bs}, rank={rank}',
            })

        if num_tests > 5:
            # Test 6: Different input dimension
            num_layers = 4
            input_dim = 128
            hidden_dim = 256
            rank = 0
            half = 2
            bs = 2
            num_points = 10
            positional_enc = False
            latent_in = [4]  # Default latent_in
            U, Vt, bias = self.generate_model_params(num_layers, input_dim, hidden_dim, rank, half, positional_enc, 8, latent_in)
            input_tensor = torch.randn(num_points * bs, input_dim + 3)
            Sigma = torch.randn(bs, num_layers, hidden_dim) * 0.1 - 2.0

            test_cases.append({
                'rank': rank,
                'half': half,
                'U': U,
                'Vt': Vt,
                'bias': bias,
                'input': input_tensor,
                'Sigma': Sigma,
                'bs': bs,
                'positional_enc': False,
                'latent_in': latent_in,
                'description': f'Larger input: input_dim={input_dim}, hidden_dim={hidden_dim}',
            })

        if num_tests > 6:
            # Test 7: With dropout
            num_layers = 4
            input_dim = 64
            hidden_dim = 128
            rank = 0
            half = 2
            bs = 2
            num_points = 10
            positional_enc = False
            latent_in = [4]  # Default latent_in
            U, Vt, bias = self.generate_model_params(num_layers, input_dim, hidden_dim, rank, half, positional_enc, 8, latent_in)
            input_tensor = torch.randn(num_points * bs, input_dim + 3)
            Sigma = torch.randn(bs, num_layers, hidden_dim) * 0.1 - 2.0

            test_cases.append({
                'rank': rank,
                'half': half,
                'U': U,
                'Vt': Vt,
                'bias': bias,
                'input': input_tensor,
                'Sigma': Sigma,
                'bs': bs,
                'positional_enc': False,
                'dropout_prob': 0.1,
                'latent_in': latent_in,
                'description': f'With dropout: dropout_prob=0.1',
            })

        if num_tests > 7:
            # Test 8: Single point per batch
            num_layers = 4
            input_dim = 64
            hidden_dim = 128
            rank = 0
            half = 2
            bs = 1
            num_points = 1
            positional_enc = False
            latent_in = [4]  # Default latent_in
            U, Vt, bias = self.generate_model_params(num_layers, input_dim, hidden_dim, rank, half, positional_enc, 8, latent_in)
            input_tensor = torch.randn(num_points * bs, input_dim + 3)
            Sigma = torch.randn(bs, num_layers, hidden_dim) * 0.1 - 2.0

            test_cases.append({
                'rank': rank,
                'half': half,
                'U': U,
                'Vt': Vt,
                'bias': bias,
                'input': input_tensor,
                'Sigma': Sigma,
                'bs': bs,
                'positional_enc': False,
                'latent_in': latent_in,
                'description': f'Single point: bs={bs}, num_points={num_points}',
            })

        if num_tests > 8:
            # Test 9: Combined: positional encoding + residual
            num_layers = 4
            input_dim = 64
            hidden_dim = 128
            rank = 2
            half = 2
            bs = 2
            num_points = 10
            positional_enc = True
            n_positional_freqs = 6
            latent_in = [4]  # Default latent_in
            U, Vt, bias = self.generate_model_params(num_layers, input_dim, hidden_dim, rank, half, positional_enc, n_positional_freqs, latent_in)
            input_tensor = torch.randn(num_points * bs, input_dim + 3)
            Sigma = torch.randn(bs, num_layers, hidden_dim + rank) * 0.1 - 2.0

            test_cases.append({
                'rank': rank,
                'half': half,
                'U': U,
                'Vt': Vt,
                'bias': bias,
                'input': input_tensor,
                'Sigma': Sigma,
                'bs': bs,
                'positional_enc': True,
                'n_positional_freqs': 6,
                'n_alpha_epochs': 5,
                'latent_in': latent_in,
                'description': f'Combined: positional encoding + residual, rank={rank}',
            })

        if num_tests > 9:
            # Test 10: Different latent_in (must generate params with same latent_in)
            num_layers = 4
            input_dim = 64
            hidden_dim = 128
            rank = 0
            half = 2
            bs = 2
            num_points = 10
            positional_enc = False
            latent_in = [2, 3]
            U, Vt, bias = self.generate_model_params(num_layers, input_dim, hidden_dim, rank, half, positional_enc, 8, latent_in)
            input_tensor = torch.randn(num_points * bs, input_dim + 3)
            Sigma = torch.randn(bs, num_layers, hidden_dim) * 0.1 - 2.0

            test_cases.append({
                'rank': rank,
                'half': half,
                'U': U,
                'Vt': Vt,
                'bias': bias,
                'input': input_tensor,
                'Sigma': Sigma,
                'bs': bs,
                'positional_enc': False,
                'latent_in': latent_in,
                'description': f'Different latent_in: latent_in=[2, 3]',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            num_layers = 3 + (i % 5)
            input_dim = 32 + (i % 3) * 32
            hidden_dim = 64 + (i % 3) * 64
            rank = (i % 3) * 2
            half = num_layers // 2
            bs = 1 + (i % 4)
            num_points = 5 + (i % 10)
            positional_enc = (i % 2 == 0)
            n_positional_freqs = 6 + (i % 3) * 2

            latent_in = [4]  # Default latent_in
            U, Vt, bias = self.generate_model_params(num_layers, input_dim, hidden_dim, rank, half, positional_enc, n_positional_freqs, latent_in)
            input_tensor = torch.randn(num_points * bs, input_dim + 3)
            if rank > 0:
                Sigma = torch.randn(bs, num_layers, hidden_dim + rank) * 0.1 - 2.0
            else:
                Sigma = torch.randn(bs, num_layers, hidden_dim) * 0.1 - 2.0

            test_cases.append({
                'rank': rank,
                'half': half,
                'U': U,
                'Vt': Vt,
                'bias': bias,
                'input': input_tensor,
                'Sigma': Sigma,
                'bs': bs,
                'positional_enc': (i % 2 == 0),
                'latent_in': latent_in,
                'description': f'Additional test {i+1}: layers={num_layers}, bs={bs}, rank={rank}',
            })

        return test_cases[:num_tests]
