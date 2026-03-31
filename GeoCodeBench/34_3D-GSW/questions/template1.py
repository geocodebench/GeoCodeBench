
import numpy as np
from numpy.fft import fft2, fftshift


def high_frequency_strength(patch):
    """
    Scalar high-frequency strength from a 2D patch (FFT magnitude with axis weights).

    Args:
        patch: 2D ndarray.

    Returns:
        float: mean of weighted |fftshift(fft2(patch))| (weights: linspace(-1,1)^2 per axis).
    """

    ****EMPTY****
    return f_weighted.sum() / f_weighted.size


def patchify_and_get_fdomain(image, patch_size):
    """
    Tile image into patches; FFT each full patch and score high-frequency strength.

    Args:
        image: 2D ndarray.
        patch_size: (h, w); stride (h, w). Skip incomplete edge patches.

    Returns:
        (frequency_patches, high_frequency_score_list): lists of fftshift(fft2(patch))
        and high_frequency_strength(patch), same length and order.
    """
    
    ****EMPTY****
    return frequency_patches, high_frequency_score_list
