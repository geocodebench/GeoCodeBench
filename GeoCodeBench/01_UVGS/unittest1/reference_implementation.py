"""
Reference Implementation of spherical_unwrap_opacity
This serves as the ground truth for testing LLM-generated implementations.
"""

import numpy as np


def spherical_unwrap_opacity(points, opacity, height=256, width=256):
    """
    Unwrap a point cloud onto a spherical surface.

    Args:
    points (np.array): Input point cloud array of shape (N, 3), where N is the number of points.
    opacity (np.array): Opacity values for each point, shape (N,).
    height (int): The vertical resolution of the output image.
    width (int): The horizontal resolution of the output image.

    Returns:
    np.array: Unwrapped image of the spherical projection with shape (height, width, 3).
              Each pixel contains the index of the point with highest opacity at that location.
    """
    # Extract x, y, z coordinates
    x1, y1, z1 = points[:, 0], points[:, 1], points[:, 2]
    
    x = z1
    y = y1
    z = x1

    # Compute spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2)  # Spherical radius
    theta = np.arctan2(y, x)  # Azimuthal angle
    phi = np.arccos(z / r)  # Polar angle, assuming r is never zero

    # Normalize and scale theta and phi to image coordinates
    theta = np.degrees(theta) + 180  # Convert angle to [0, 360] degrees
    phi = np.degrees(phi)  # Convert polar angle to degrees
    theta_scaled = np.round((theta / 360) * width).astype(int)  # Map to image width
    phi_scaled = np.round((phi / 180) * height).astype(int)  # Map to image height

    # Create an empty image
    image = np.zeros((height, width, 3))
    opacity_image = np.zeros((height, width, 1))

    # Fill image with original xyz values
    for ind, t, P, xyz, opac in zip(range(len(opacity)), theta_scaled, phi_scaled, points, opacity):
        if 0 <= P < height and 0 <= t < width:
            if image[P, t, :].all() == 0:
                image[P, t, :] = ind
                opacity_image[P, t, :] = opac
            else:
                if opac > opacity_image[P, t, :]:
                    image[P, t, :] = ind

    return image

