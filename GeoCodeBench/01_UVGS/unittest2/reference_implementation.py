"""
Reference Implementation of equirectangular_unwrap_topK_opacity
This serves as the ground truth for testing LLM-generated implementations.
"""

import numpy as np


def equirectangular_unwrap_topK_opacity(points, opacity, height=512, width=512, K=4):
    """
    Unwrap a point cloud onto an equirectangular (lat-lon) image.

    Args:
        points (np.array): Input point cloud, shape (N,3), where N is #points.
                           Each row is (x, y, z).
        opacity (np.array): Opacity (or intensity) values per point, shape (N,).
        height (int): Vertical resolution of the output image.
        width (int):  Horizontal resolution of the output image.
        K (int): Number of top-opacity points to keep per pixel.

    Returns:
        np.array: An integer image array of shape (height, width, K),
                  containing point indices for the top K opacity values in each pixel.
                  Pixels with fewer than K points have some default fill (0 or no points).
    """
    # 1) Separate coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # 2) Compute spherical radius and angles
    #    r = distance from origin
    #    lon = arctan2(y, x) in [-pi, pi]
    #    lat = arcsin(z / r) in [-pi/2, pi/2]
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-12  # small epsilon to avoid div-by-zero
    lon = np.arctan2(y, x)            # range: [-pi, pi]
    lat = np.arcsin(z / r)            # range: [-pi/2, pi/2]

    # 3) Convert angles to [0,1] range
    #    u in [0,1] for lon in [-pi, +pi]
    #    v in [0,1] for lat in [-pi/2, +pi/2]
    #    Then map to pixel coordinates:
    #      col = round(u * (width - 1))
    #      row = round(v * (height - 1))
    u = (lon + np.pi) / (2.0 * np.pi)          # map lon: [-pi, pi]   -> [0, 1]
    v = (lat + np.pi/2.0) / np.pi              # map lat: [-pi/2, pi/2] -> [0, 1]

    col = (u * (width  - 1)).round().astype(int)
    row = (v * (height - 1)).round().astype(int)

    # 4) Prepare output arrays
    #    image[p_row, p_col, :]   = top K indices
    #    opacity_image[p_row, p_col, :] = top K opacities
    image = np.zeros((height, width, K), dtype=int)
    opacity_image = np.full((height, width, K), -np.inf, dtype=float)

    # 5) Populate each pixel with up to K highest-opacity points
    for pt_idx, (r_i, c_i, opac) in enumerate(zip(row, col, opacity)):
        if 0 <= r_i < height and 0 <= c_i < width:
            current_opacities = opacity_image[r_i, c_i, :]
            # If there's space or this point has higher opacity than the lowest in top-K
            if -np.inf in current_opacities or opac > current_opacities.min():
                # Replace the lowest opacity
                min_index = current_opacities.argmin()
                opacity_image[r_i, c_i, min_index] = opac
                image[r_i, c_i, min_index] = pt_idx

    return image

