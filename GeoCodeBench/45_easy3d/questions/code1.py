# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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

        ****EMPTY****

        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        activation=nn.ReLU,
        num_heads=8,
        depth=2,
    ) -> None:
        """
        Adapted from Segment Anything (SAM).
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_clicks_to_scene = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, batch_first=True
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, scene_embedding: Tensor, scene_pe: Tensor, clicks_embedding: Tensor) -> Tuple[Tensor, Tensor]:

        # Define initial queries and keys
        queries = clicks_embedding
        keys = scene_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys, query_pe=clicks_embedding, key_pe=scene_pe)

        # Apply the final attention layer from the clicks to the scene
        q = queries + clicks_embedding
        k = keys + scene_pe
        attn_out, _ = self.final_attn_clicks_to_scene(query=q, key=k, value=keys, need_weights=False)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return keys, queries