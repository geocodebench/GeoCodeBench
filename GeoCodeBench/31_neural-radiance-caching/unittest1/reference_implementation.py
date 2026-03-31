"""
Reference Implementation for shift_direct()
This serves as the ground truth for testing LLM-generated implementations.
"""

import jax.numpy as jnp


def shift_direct(dists, direct_rgbs, weights, n_bins, num_rgb_channels, impulse_response):
    n_rays = direct_rgbs.shape[0]
    rgb = jnp.zeros((n_rays, n_bins, num_rgb_channels))

    dists_low = jnp.maximum(jnp.floor(dists), 0)
    dists_high = jnp.ceil(dists)
    weights_high = dists - dists_low
    weights_low = 1.0 - weights_high

    indices_low = (
        jnp.repeat(jnp.arange(direct_rgbs.shape[0]) * n_bins, direct_rgbs.shape[1])
        + dists_low.reshape(-1).astype(jnp.int32)
    ).astype(jnp.int32)
    indices_high = (
        jnp.repeat(jnp.arange(direct_rgbs.shape[0]) * n_bins, direct_rgbs.shape[1])
        + dists_high.reshape(-1).astype(jnp.int32)
    ).astype(jnp.int32)

    rgb = (
        (rgb.reshape(-1, num_rgb_channels))
        .at[indices_low]
        .add(
            (weights[Ellipsis, None] * direct_rgbs).reshape(-1, num_rgb_channels)
            * weights_low.reshape(-1, 1)
        )
        .reshape(-1, n_bins, num_rgb_channels)
    )

    rgb = (
        (rgb.reshape(-1, num_rgb_channels))
        .at[indices_high]
        .add(
            (weights[Ellipsis, None] * direct_rgbs).reshape(-1, num_rgb_channels)
            * weights_high.reshape(-1, 1)
        )
        .reshape(-1, n_bins, num_rgb_channels)
    )

    return rgb
