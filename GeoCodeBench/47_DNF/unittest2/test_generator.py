"""
Test Data Generator for Dict_exp.forward() method.
Generates test cases with different configurations.
"""

from __future__ import annotations

import torch

from reference_implementation import Dict_exp as RefDictExp


class TestDataGenerator:
    """Generate test data for Dict_exp.forward() method."""

    def __init__(self, seed=42, device='cpu'):
        self.seed = seed
        self.device = device
        torch.manual_seed(seed)

    def create_dict_exp_instance(self, num_layers=4, input_dim=256, hidden_dims=[128, 64, 32],
                                  output_dim=1, positional_enc=True, n_positional_freqs=8,
                                  latent_in=[4], dropout_prob=0.0, dropout=None):
        """Create a Dict_exp instance with random parameters."""
        bias = []
        U = []
        Sigma = []
        Vt = []

        # Calculate dimensions considering positional encoding
        if positional_enc:
            from utils import embedder
            _, pos_embed_dim = embedder.get_embedder_nerf(n_positional_freqs, input_dims=3, i=0)
            # Input: (input_dim - 3) + pos_embed_dim, where last 3 are xyz
            actual_input_dim = input_dim - 3 + pos_embed_dim
        else:
            actual_input_dim = input_dim

        # First layer input dimension (before any latent_in concatenation)
        layer_input_dim = actual_input_dim

        for i in range(num_layers):
            # Determine output dimension for this layer
            if i < len(hidden_dims):
                layer_output_dim = hidden_dims[i]
            elif i == num_layers - 1:
                layer_output_dim = output_dim
            else:
                layer_output_dim = hidden_dims[-1] if hidden_dims else output_dim

            # If this layer is in latent_in, input will be concatenated
            # So Vt needs to handle the concatenated input
            if i in latent_in:
                if positional_enc:
                    # Will concatenate with input_embed (same as actual_input_dim)
                    vt_input_dim = layer_input_dim + actual_input_dim
                else:
                    # Will concatenate with original input
                    vt_input_dim = layer_input_dim + input_dim
            else:
                vt_input_dim = layer_input_dim

            # Create U, Sigma, Vt for this layer
            # U: (output_dim, rank), Vt: (rank, input_dim)
            rank = min(vt_input_dim, layer_output_dim) // 2
            rank = max(rank, 8)  # Minimum rank

            U_i = torch.randn(layer_output_dim, rank, device=self.device) * 0.1
            Sigma_i = torch.randn(rank, device=self.device) * 0.5 - 1.0  # Negative values for exp
            Vt_i = torch.randn(rank, vt_input_dim, device=self.device) * 0.1
            bias_i = torch.randn(layer_output_dim, device=self.device) * 0.1

            U.append(U_i)
            Sigma.append(Sigma_i)
            Vt.append(Vt_i)
            bias.append(bias_i)

            # Next layer input is current output
            layer_input_dim = layer_output_dim

        # Create instance
        instance = RefDictExp(
            bias=bias,
            U=U,
            Sigma=Sigma,
            Vt=Vt,
            latent_in=latent_in,
            dropout_prob=dropout_prob,
            dropout=dropout,
            positional_enc=positional_enc,
            n_positional_freqs=n_positional_freqs
        )

        instance.eval()  # Set to eval mode
        return instance

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case - small input, with positional encoding
        model1 = self.create_dict_exp_instance(
            num_layers=3, input_dim=64, hidden_dims=[32, 16], output_dim=1,
            positional_enc=True, n_positional_freqs=4, latent_in=[2]
        )
        input1 = torch.randn(10, 64, device=self.device)
        test_cases.append({
            'model': model1,
            'input': input1,
            'description': f'Basic: N=10, dim=64, 3 layers, pos_enc=True, latent_in=[2]',
        })

        if num_tests > 1:
            # Test 2: Without positional encoding
            model2 = self.create_dict_exp_instance(
                num_layers=4, input_dim=128, hidden_dims=[64, 32, 16], output_dim=1,
                positional_enc=False, latent_in=[4]
            )
            input2 = torch.randn(20, 128, device=self.device)
            test_cases.append({
                'model': model2,
                'input': input2,
                'description': f'No pos_enc: N=20, dim=128, 4 layers, latent_in=[4]',
            })

        if num_tests > 2:
            # Test 3: Single point
            model3 = self.create_dict_exp_instance(
                num_layers=2, input_dim=32, hidden_dims=[16], output_dim=1,
                positional_enc=True, n_positional_freqs=6, latent_in=[]
            )
            input3 = torch.randn(1, 32, device=self.device)
            test_cases.append({
                'model': model3,
                'input': input3,
                'description': f'Single point: N=1, dim=32, 2 layers, no latent_in',
            })

        if num_tests > 3:
            # Test 4: Large batch
            model4 = self.create_dict_exp_instance(
                num_layers=5, input_dim=256, hidden_dims=[128, 64, 32, 16], output_dim=1,
                positional_enc=True, n_positional_freqs=8, latent_in=[1, 3]
            )
            input4 = torch.randn(100, 256, device=self.device)
            test_cases.append({
                'model': model4,
                'input': input4,
                'description': f'Large batch: N=100, dim=256, 5 layers, latent_in=[1,3]',
            })

        if num_tests > 4:
            # Test 5: Different positional frequencies
            model5 = self.create_dict_exp_instance(
                num_layers=3, input_dim=96, hidden_dims=[48, 24], output_dim=1,
                positional_enc=True, n_positional_freqs=10, latent_in=[2]
            )
            input5 = torch.randn(15, 96, device=self.device)
            test_cases.append({
                'model': model5,
                'input': input5,
                'description': f'High freq: N=15, dim=96, n_freqs=10, latent_in=[2]',
            })

        if num_tests > 5:
            # Test 6: With dropout
            model6 = self.create_dict_exp_instance(
                num_layers=4, input_dim=128, hidden_dims=[64, 32, 16], output_dim=1,
                positional_enc=True, n_positional_freqs=6, latent_in=[1, 2],
                dropout_prob=0.1, dropout=[1, 2]
            )
            input6 = torch.randn(25, 128, device=self.device)
            test_cases.append({
                'model': model6,
                'input': input6,
                'description': f'With dropout: N=25, dim=128, dropout_prob=0.1',
            })

        if num_tests > 6:
            # Test 7: Small dimensions
            model7 = self.create_dict_exp_instance(
                num_layers=2, input_dim=16, hidden_dims=[8], output_dim=1,
                positional_enc=False, latent_in=[]
            )
            input7 = torch.randn(5, 16, device=self.device)
            test_cases.append({
                'model': model7,
                'input': input7,
                'description': f'Small dims: N=5, dim=16, 2 layers, no pos_enc',
            })

        if num_tests > 7:
            # Test 8: Many layers
            model8 = self.create_dict_exp_instance(
                num_layers=6, input_dim=192, hidden_dims=[96, 48, 24, 12, 6], output_dim=1,
                positional_enc=True, n_positional_freqs=8, latent_in=[3, 5]
            )
            input8 = torch.randn(30, 192, device=self.device)
            test_cases.append({
                'model': model8,
                'input': input8,
                'description': f'Many layers: N=30, dim=192, 6 layers',
            })

        if num_tests > 8:
            # Test 9: All latent inputs
            model9 = self.create_dict_exp_instance(
                num_layers=4, input_dim=128, hidden_dims=[64, 32, 16], output_dim=1,
                positional_enc=True, n_positional_freqs=6, latent_in=[0, 1, 2, 3]
            )
            input9 = torch.randn(18, 128, device=self.device)
            test_cases.append({
                'model': model9,
                'input': input9,
                'description': f'All latent: N=18, all layers in latent_in',
            })

        if num_tests > 9:
            # Test 10: Very large input dimension
            model10 = self.create_dict_exp_instance(
                num_layers=3, input_dim=512, hidden_dims=[256, 128], output_dim=1,
                positional_enc=True, n_positional_freqs=8, latent_in=[1]
            )
            input10 = torch.randn(12, 512, device=self.device)
            test_cases.append({
                'model': model10,
                'input': input10,
                'description': f'Large input: N=12, dim=512',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            num_layers = 3 + (i % 3)
            input_dim = 64 + (i % 5) * 32
            hidden_dims = [input_dim // 2, input_dim // 4][:num_layers-1]
            pos_enc = (i % 2 == 0)
            n_freqs = 4 + (i % 4) * 2

            model = self.create_dict_exp_instance(
                num_layers=num_layers, input_dim=input_dim, hidden_dims=hidden_dims,
                output_dim=1, positional_enc=pos_enc, n_positional_freqs=n_freqs,
                latent_in=[num_layers // 2] if num_layers > 2 else []
            )
            input_tensor = torch.randn(8 + i * 2, input_dim, device=self.device)

            test_cases.append({
                'model': model,
                'input': input_tensor,
                'description': f'Additional test {i+1}: N={8+i*2}, dim={input_dim}, layers={num_layers}',
            })

        return test_cases[:num_tests]
