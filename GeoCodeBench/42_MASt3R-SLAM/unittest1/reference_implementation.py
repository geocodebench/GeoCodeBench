"""
Reference Implementation for project_calib() function
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch


def decompose_K(K):
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    return fx, fy, cx, cy


def project_calib(P, K, img_size, jacobian=False, border=0, z_eps=0.0):
    b = P.shape[:-1]
    # print(K.shape)
    # print(K.view(1, 1, 3, 3))
    K_rep = K.repeat(*b, 1, 1)

    p = (K_rep @ P[..., None]).squeeze(-1)
    p = p / p[..., 2:3]
    p = p[..., :2]

    u, v = p.split([1, 1], dim=-1)
    x, y, z = P.split([1, 1, 1], dim=-1)

    # Check if pixel falls in image
    valid_u = (u > border) & (u < img_size[1] - 1 - border)
    valid_v = (v > border) & (v < img_size[0] - 1 - border)
    # Check if in front of camera
    valid_z = z > z_eps
    # Get total valid
    valid = valid_u & valid_v & valid_z

    # Depth transformation
    logz = torch.log(z)
    invalid_z = torch.logical_not(valid_z)
    logz[invalid_z] = 0.0  # Need to avoid nans

    # Output
    pz = torch.cat((p, logz), dim=-1)

    if not jacobian:
        return pz, valid
    else:
        fx, fy, cx, cy = decompose_K(K)
        z_inv = 1.0 / z[..., 0]
        dpz_dP = torch.zeros(*b + (3, 3), device=P.device, dtype=P.dtype)
        dpz_dP[..., 0, 0] = fx
        dpz_dP[..., 1, 1] = fy
        dpz_dP[..., 0, 2] = -fx * x[..., 0] * z_inv
        dpz_dP[..., 1, 2] = -fy * y[..., 0] * z_inv
        dpz_dP *= z_inv[..., None, None]
        dpz_dP[..., 2, 2] = z_inv  # Only z itself in bottom row
        return pz, dpz_dP, valid
