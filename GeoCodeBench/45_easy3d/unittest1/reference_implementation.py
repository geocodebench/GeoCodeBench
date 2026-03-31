"""
Reference Implementation for TwoWayAttentionBlock.forward()
This serves as the ground truth for testing LLM-generated implementations.
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
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


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
        self.self_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_clicks_to_scene = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_scene_to_clicks = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, batch_first=True
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor):
        # Self attention block
        if self.skip_first_layer_pe:
            queries, _ = self.self_attn(query=queries, key=queries, value=queries, need_weights=False)
        else:
            q = queries + query_pe
            attn_out, _ = self.self_attn(query=q, key=q, value=queries, need_weights=False)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, clicks attending to scene embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out, _ = self.cross_attn_clicks_to_scene(query=q, key=k, value=keys, need_weights=False)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, scene embedding attending to clicks
        q = queries + query_pe
        k = keys + key_pe
        attn_out, _ = self.cross_attn_scene_to_clicks(query=k, key=q, value=queries, need_weights=False)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys
