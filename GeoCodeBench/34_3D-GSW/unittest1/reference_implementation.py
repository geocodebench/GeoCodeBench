"""
Reference Implementation for high_frequency_strength() and patchify_and_get_fdomain()
This serves as the ground truth for testing LLM-generated implementations.
"""

import numpy as np
from numpy.fft import fft2, fftshift


def high_frequency_strength(patch):
    # Compute the 2D FFT of the patch
    f = np.fft.fft2(patch)
    # Center the zero-frequency component
    fshift = np.fft.fftshift(f)
    # Compute the magnitude spectrum
    magnitude_spectrum = np.abs(fshift)
    
    # Define high-frequency region (corners of the array)
    # For simplicity, consider a square around the corners to be the high-frequency region
    # Adjust the size of the region as needed
    size = patch.shape[0]
    corner_size = size // 4  # Example: consider outer 25% of the array as high-frequency region
    high_freq_region = np.r_[
        magnitude_spectrum[:corner_size, :corner_size].flat,
        magnitude_spectrum[-corner_size:, :corner_size].flat,
        magnitude_spectrum[:corner_size, -corner_size:].flat,
        magnitude_spectrum[-corner_size:, -corner_size:].flat,
    ]
    
    # Calculate the strength of the high-frequency signal (you can also use other metrics)
    
    hor_weight = np.linspace(-1, 1, f.shape[1]).reshape(1, -1, 1) ** 2
    ver_weight = np.linspace(-1, 1, f.shape[0]).reshape(-1, 1, 1) ** 2
    
    f_weighted = np.abs(fshift) * hor_weight * ver_weight
    return f_weighted.sum() / f_weighted.size


def patchify_and_get_fdomain(image, patch_size):
    frequency_patches = []
    high_frequency_score_list = []

    for i in range(0, image.shape[0], patch_size[0]):
        for j in range(0, image.shape[1], patch_size[1]):
            # Extract the patch
            patch = image[i:i + patch_size[0], j:j + patch_size[1]]

            # Check if the patch size is as expected (it might not be at the edges)
            if patch.shape[0] == patch_size[0] and patch.shape[1] == patch_size[1]:
                # Compute the 2D Fourier Transform of the patch
                fft_patch = fft2(patch)
                # Shift the zero frequency component to the center
                fft_patch_shifted = fftshift(fft_patch)

                # Save the transformed patch
                frequency_patches.append(fft_patch_shifted)

                strength = high_frequency_strength(patch)
                high_frequency_score_list.append(strength)
    return frequency_patches, high_frequency_score_list
