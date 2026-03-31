"""
Reference Implementation for _get_subpixel_maximums() function
This serves as the ground truth for testing LLM-generated implementations.
"""

import numpy as np


def _get_subpixel_maximums(heatmaps, locs):
    # Extract integer peak locations
    x_locs = locs[:, 0].astype(np.int32)
    y_locs = locs[:, 1].astype(np.int32)

    # Ensure we are not near the boundaries (avoid boundary issues)
    valid_mask = (x_locs > 0) & (x_locs < heatmaps.shape[2] - 1) & (y_locs > 0) & (y_locs < heatmaps.shape[1] - 1)

    # Initialize the output array with the integer locations
    subpixel_locs = locs.copy()

    if np.any(valid_mask):
        # Extract valid locations
        x_locs_valid = x_locs[valid_mask]
        y_locs_valid = y_locs[valid_mask]

        # Compute gradients (dx, dy) and second derivatives (dxx, dyy)
        dx = (
            heatmaps[valid_mask, y_locs_valid, x_locs_valid + 1] - heatmaps[valid_mask, y_locs_valid, x_locs_valid - 1]
        ) / 2.0
        dy = (
            heatmaps[valid_mask, y_locs_valid + 1, x_locs_valid] - heatmaps[valid_mask, y_locs_valid - 1, x_locs_valid]
        ) / 2.0
        dxx = (
            heatmaps[valid_mask, y_locs_valid, x_locs_valid + 1]
            + heatmaps[valid_mask, y_locs_valid, x_locs_valid - 1]
            - 2 * heatmaps[valid_mask, y_locs_valid, x_locs_valid]
        )
        dyy = (
            heatmaps[valid_mask, y_locs_valid + 1, x_locs_valid]
            + heatmaps[valid_mask, y_locs_valid - 1, x_locs_valid]
            - 2 * heatmaps[valid_mask, y_locs_valid, x_locs_valid]
        )

        # Avoid division by zero by setting a minimum threshold for the second derivatives
        dxx = np.where(dxx != 0, dxx, 1e-6)
        dyy = np.where(dyy != 0, dyy, 1e-6)

        # Calculate the sub-pixel shift
        subpixel_x_shift = -dx / dxx
        subpixel_y_shift = -dy / dyy

        # Update subpixel locations for valid indices
        subpixel_locs[valid_mask, 0] += subpixel_x_shift
        subpixel_locs[valid_mask, 1] += subpixel_y_shift

    return subpixel_locs
