
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
        ****EMPTY****

    return subpixel_locs
