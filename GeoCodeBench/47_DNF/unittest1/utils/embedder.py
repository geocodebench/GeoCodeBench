"""
Positional encoding embedder for NeRF-style positional encoding.
"""

import torch
import torch.nn as nn


class Embedder:
    """Positional encoding embedder."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
    
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d
        
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)
        
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d
        
        self.embed_fns = embed_fns
        self.out_dim = out_dim
    
    def embed(self, inputs, alpha=None):
        """Embed inputs with optional alpha parameter."""
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder_nerf(multires, input_dims=3):
    """
    Get NeRF-style positional encoding embedder.
    
    Args:
        multires: Number of frequency bands
        input_dims: Input dimension (default: 3 for xyz)
    
    Returns:
        embed_fn: Embedding function
        out_dim: Output dimension
    """
    if multires == -1:
        return nn.Identity(), input_dims
    
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    
    def embed_fn(x, alpha=None):
        return embedder_obj.embed(x, alpha)
    
    return embed_fn, embedder_obj.out_dim
