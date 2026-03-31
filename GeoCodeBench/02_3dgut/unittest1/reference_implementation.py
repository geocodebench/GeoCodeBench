"""
Reference implementations for _isect_tiles and _isect_offset_encode.
This serves as the ground truth for testing LLM-generated implementations.
"""

import math
import struct
from typing import Tuple

import torch


def reference_isect_tiles(
    means2d: torch.Tensor,  # [..., N, 2]
    radii: torch.Tensor,  # [..., N, 2]
    depths: torch.Tensor,  # [..., N]
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation of _isect_tiles()."""
    image_dims = means2d.shape[:-2]
    n_gauss = means2d.shape[-2]
    device = means2d.device
    num_images = math.prod(image_dims)

    means2d = means2d.reshape(num_images, n_gauss, 2)
    radii = radii.reshape(num_images, n_gauss, 2)
    depths = depths.reshape(num_images, n_gauss)

    tile_means2d = means2d / tile_size
    tile_radii = radii / tile_size
    tile_mins = torch.floor(tile_means2d - tile_radii).int()
    tile_maxs = torch.ceil(tile_means2d + tile_radii).int()
    tile_mins[..., 0] = torch.clamp(tile_mins[..., 0], 0, tile_width)
    tile_mins[..., 1] = torch.clamp(tile_mins[..., 1], 0, tile_height)
    tile_maxs[..., 0] = torch.clamp(tile_maxs[..., 0], 0, tile_width)
    tile_maxs[..., 1] = torch.clamp(tile_maxs[..., 1], 0, tile_height)
    tiles_per_gauss = (tile_maxs - tile_mins).prod(dim=-1)
    tiles_per_gauss *= (radii > 0.0).all(dim=-1)

    n_isects = tiles_per_gauss.sum().item()
    isect_ids_lo = torch.empty(n_isects, dtype=torch.int32, device=device)
    isect_ids_hi = torch.empty(n_isects, dtype=torch.int32, device=device)
    flatten_ids = torch.empty(n_isects, dtype=torch.int32, device=device)

    cum_tiles_per_gauss = torch.cumsum(tiles_per_gauss.flatten(), dim=0)
    tile_n_bits = (tile_width * tile_height).bit_length()

    def kernel(image_id: int, gauss_id: int) -> None:
        if radii[image_id, gauss_id, 0] <= 0.0 or radii[image_id, gauss_id, 1] <= 0.0:
            return

        index = image_id * n_gauss + gauss_id
        curr_idx = cum_tiles_per_gauss[index - 1] if index > 0 else 0

        depth_f32 = depths[image_id, gauss_id]
        depth_id = struct.unpack("i", struct.pack("f", depth_f32))[0]
        depth_id = int(depth_id) & 0xFFFFFFFF

        tile_min = tile_mins[image_id, gauss_id]
        tile_max = tile_maxs[image_id, gauss_id]
        for y in range(tile_min[1], tile_max[1]):
            for x in range(tile_min[0], tile_max[0]):
                tile_id = y * tile_width + x
                isect_ids_lo[curr_idx] = depth_id
                isect_ids_hi[curr_idx] = (image_id << tile_n_bits) | tile_id
                flatten_ids[curr_idx] = index
                curr_idx += 1

    for image_id in range(num_images):
        for gauss_id in range(n_gauss):
            kernel(image_id, gauss_id)

    isect_ids = (isect_ids_hi.to(torch.int64) << 32) | (
        isect_ids_lo.to(torch.int64) & 0xFFFFFFFF
    )

    if sort:
        isect_ids, sort_indices = torch.sort(isect_ids)
        flatten_ids = flatten_ids[sort_indices]

    tiles_per_gauss = tiles_per_gauss.reshape(image_dims + (n_gauss,)).int()
    return tiles_per_gauss, isect_ids, flatten_ids


def reference_isect_offset_encode(
    isect_ids: torch.Tensor, num_images: int, tile_width: int, tile_height: int
) -> torch.Tensor:
    """Reference implementation of _isect_offset_encode()."""
    tile_n_bits = (tile_width * tile_height).bit_length()
    tile_counts = torch.zeros(
        (num_images, tile_height, tile_width), dtype=torch.int64, device=isect_ids.device
    )

    isect_ids_uq, counts = torch.unique_consecutive(isect_ids >> 32, return_counts=True)

    image_ids_uq = isect_ids_uq >> tile_n_bits
    tile_ids_uq = isect_ids_uq & ((1 << tile_n_bits) - 1)
    tile_ids_x_uq = tile_ids_uq % tile_width
    tile_ids_y_uq = tile_ids_uq // tile_width

    tile_counts[image_ids_uq, tile_ids_y_uq, tile_ids_x_uq] = counts

    cum_tile_counts = torch.cumsum(tile_counts.flatten(), dim=0).reshape_as(tile_counts)
    offsets = cum_tile_counts - tile_counts
    return offsets.int()
