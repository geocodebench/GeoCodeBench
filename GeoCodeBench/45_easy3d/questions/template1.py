
"""
LLM Template for TwoWayAttentionBlock
Fill in the implementation below.
"""

from typing import Tuple, Type

import torch
from torch import Tensor, nn


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        # TODO: Implement the MLPBlock initialization here
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the MLPBlock forward pass here
        return x


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        Adapted from Segment Anything (SAM). Uses pytorch's nn.MultiheadAttention and flash attention.
        """
        super().__init__()
        # TODO: Implement the TwoWayAttentionBlock initialization here
        pass

    def forward(self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor):
        # TODO: Implement the TwoWayAttentionBlock forward pass here
        # Self attention block
        
        # Cross attention block, clicks attending to scene embedding
        
        # MLP block
        
        # Cross attention block, scene embedding attending to clicks
        
        return queries, keys
