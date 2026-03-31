"""
Test Data Generator for TwoWayAttentionBlock.forward() function.
Generates test cases with different configurations.
"""

import torch


class TestDataGenerator:
    """Generate test data for TwoWayAttentionBlock.forward() function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Common configurations
        embedding_dims = [64, 128, 256, 512]
        num_heads_list = [4, 8]
        mlp_dims = [128, 256, 512]
        batch_sizes = [1, 2, 4, 8]
        seq_lens_queries = [10, 20, 50, 100]
        seq_lens_keys = [20, 50, 100, 200]

        test_idx = 0

        # Test 1: Basic case with small dimensions
        embedding_dim = 64
        num_heads = 4
        mlp_dim = 128
        batch_size = 2
        seq_len_queries = 10
        seq_len_keys = 20
        skip_first_layer_pe = False

        queries = torch.randn(batch_size, seq_len_queries, embedding_dim)
        keys = torch.randn(batch_size, seq_len_keys, embedding_dim)
        query_pe = torch.randn(batch_size, seq_len_queries, embedding_dim)
        key_pe = torch.randn(batch_size, seq_len_keys, embedding_dim)

        test_cases.append({
            'queries': queries,
            'keys': keys,
            'query_pe': query_pe,
            'key_pe': key_pe,
            'embedding_dim': embedding_dim,
            'num_heads': num_heads,
            'mlp_dim': mlp_dim,
            'skip_first_layer_pe': skip_first_layer_pe,
            'description': f'Basic: batch={batch_size}, seq_q={seq_len_queries}, seq_k={seq_len_keys}, dim={embedding_dim}',
        })
        test_idx += 1

        if num_tests > 1:
            # Test 2: Single sample, skip_first_layer_pe=True
            embedding_dim = 128
            num_heads = 8
            mlp_dim = 256
            batch_size = 1
            seq_len_queries = 5
            seq_len_keys = 10
            skip_first_layer_pe = True

            queries = torch.randn(batch_size, seq_len_queries, embedding_dim)
            keys = torch.randn(batch_size, seq_len_keys, embedding_dim)
            query_pe = torch.randn(batch_size, seq_len_queries, embedding_dim)
            key_pe = torch.randn(batch_size, seq_len_keys, embedding_dim)

            test_cases.append({
                'queries': queries,
                'keys': keys,
                'query_pe': query_pe,
                'key_pe': key_pe,
                'embedding_dim': embedding_dim,
                'num_heads': num_heads,
                'mlp_dim': mlp_dim,
                'skip_first_layer_pe': skip_first_layer_pe,
                'description': f'Single sample: batch={batch_size}, skip_pe=True, dim={embedding_dim}',
            })
            test_idx += 1

        if num_tests > 2:
            # Test 3: Larger batch and dimensions
            embedding_dim = 256
            num_heads = 8
            mlp_dim = 512
            batch_size = 4
            seq_len_queries = 20
            seq_len_keys = 50
            skip_first_layer_pe = False

            queries = torch.randn(batch_size, seq_len_queries, embedding_dim)
            keys = torch.randn(batch_size, seq_len_keys, embedding_dim)
            query_pe = torch.randn(batch_size, seq_len_queries, embedding_dim)
            key_pe = torch.randn(batch_size, seq_len_keys, embedding_dim)

            test_cases.append({
                'queries': queries,
                'keys': keys,
                'query_pe': query_pe,
                'key_pe': key_pe,
                'embedding_dim': embedding_dim,
                'num_heads': num_heads,
                'mlp_dim': mlp_dim,
                'skip_first_layer_pe': skip_first_layer_pe,
                'description': f'Larger: batch={batch_size}, seq_q={seq_len_queries}, seq_k={seq_len_keys}, dim={embedding_dim}',
            })
            test_idx += 1

        if num_tests > 3:
            # Test 4: Different sequence lengths
            embedding_dim = 128
            num_heads = 4
            mlp_dim = 256
            batch_size = 3
            seq_len_queries = 50
            seq_len_keys = 100
            skip_first_layer_pe = False

            queries = torch.randn(batch_size, seq_len_queries, embedding_dim)
            keys = torch.randn(batch_size, seq_len_keys, embedding_dim)
            query_pe = torch.randn(batch_size, seq_len_queries, embedding_dim)
            key_pe = torch.randn(batch_size, seq_len_keys, embedding_dim)

            test_cases.append({
                'queries': queries,
                'keys': keys,
                'query_pe': query_pe,
                'key_pe': key_pe,
                'embedding_dim': embedding_dim,
                'num_heads': num_heads,
                'mlp_dim': mlp_dim,
                'skip_first_layer_pe': skip_first_layer_pe,
                'description': f'Long sequences: batch={batch_size}, seq_q={seq_len_queries}, seq_k={seq_len_keys}',
            })
            test_idx += 1

        if num_tests > 4:
            # Test 5: Large embedding dimension
            embedding_dim = 512
            num_heads = 8
            mlp_dim = 1024
            batch_size = 2
            seq_len_queries = 15
            seq_len_keys = 30
            skip_first_layer_pe = True

            queries = torch.randn(batch_size, seq_len_queries, embedding_dim)
            keys = torch.randn(batch_size, seq_len_keys, embedding_dim)
            query_pe = torch.randn(batch_size, seq_len_queries, embedding_dim)
            key_pe = torch.randn(batch_size, seq_len_keys, embedding_dim)

            test_cases.append({
                'queries': queries,
                'keys': keys,
                'query_pe': query_pe,
                'key_pe': key_pe,
                'embedding_dim': embedding_dim,
                'num_heads': num_heads,
                'mlp_dim': mlp_dim,
                'skip_first_layer_pe': skip_first_layer_pe,
                'description': f'Large dim: batch={batch_size}, dim={embedding_dim}, mlp={mlp_dim}',
            })
            test_idx += 1

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            embedding_dim = embedding_dims[i % len(embedding_dims)]
            num_heads = num_heads_list[i % len(num_heads_list)]
            mlp_dim = mlp_dims[i % len(mlp_dims)]
            batch_size = batch_sizes[i % len(batch_sizes)]
            seq_len_queries = seq_lens_queries[i % len(seq_lens_queries)]
            seq_len_keys = seq_lens_keys[i % len(seq_lens_keys)]
            skip_first_layer_pe = (i % 2 == 0)

            queries = torch.randn(batch_size, seq_len_queries, embedding_dim)
            keys = torch.randn(batch_size, seq_len_keys, embedding_dim)
            query_pe = torch.randn(batch_size, seq_len_queries, embedding_dim)
            key_pe = torch.randn(batch_size, seq_len_keys, embedding_dim)

            test_cases.append({
                'queries': queries,
                'keys': keys,
                'query_pe': query_pe,
                'key_pe': key_pe,
                'embedding_dim': embedding_dim,
                'num_heads': num_heads,
                'mlp_dim': mlp_dim,
                'skip_first_layer_pe': skip_first_layer_pe,
                'description': f'Additional test {i+1}: batch={batch_size}, seq_q={seq_len_queries}, seq_k={seq_len_keys}, dim={embedding_dim}',
            })

        return test_cases[:num_tests]
