"""
LLM Template for project_calib() function
This template shows how to provide the fill-in-the-blank question to LLMs.
The template should preserve the context from the original question and only
provide input/output without any hints.
"""

import torch


def decompose_K(K):
    """
    Extract intrinsic parameters from 3x3 calibration matrix K.
    Input:  K: torch.Tensor, shape (..., 3, 3).
    Output: fx, fy, cx, cy: each shape (...), scalars per batch.
    """
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    return fx, fy, cx, cy


def project_calib(P, K, img_size, jacobian=False, border=0, z_eps=0.0):
    """
    Project 3D points (camera frame) to pixel coords and log-depth.
    Input:
        P: torch.Tensor, shape (..., 3) — 3D points (x, y, z) in camera frame.
        K: torch.Tensor, shape (..., 3, 3) — camera intrinsics (will be repeated to batch).
        img_size: (H, W) — image height and width; valid uses img_size[0] for v, img_size[1] for u.
        jacobian: bool — if True, also return Jacobian dpz_dP.
        border, z_eps: scalar — validity margin and minimum z.
    Output:
        If jacobian=False: (pz, valid)
            pz: shape (..., 3) — (u, v, log_z) pixel and log-depth.
            valid: shape (..., 1) or broadcast — bool, True where in image and z > z_eps.
        If jacobian=True: (pz, dpz_dP, valid)
            dpz_dP: shape (..., 3, 3) — Jacobian of (u, v, log_z) w.r.t. P.
    """
    b = P.shape[:-1]
    # print(K.shape)
    # print(K.view(1, 1, 3, 3))
    K_rep = K.repeat(*b, 1, 1)

    # First EMPTY: compute p (pixel xy), u, v, and x, y, z from P and K_rep.
    # Required after this block:
    #   p: shape (*b, 2) — pixel (u, v) in image coords.
    #   u, v: shape (*b, 1) each — same as p.split([1,1], dim=-1).
    #   x, y, z: shape (*b, 1) each — from P.split([1,1,1], dim=-1).
    ****EMPTY****

    # Check if pixel falls in image
    valid_u = (u > border) & (u < img_size[1] - 1 - border)
    valid_v = (v > border) & (v < img_size[0] - 1 - border)
    # Check if in front of camera
    valid_z = z > z_eps
    # Get total valid
    valid = valid_u & valid_v & valid_z

    # Depth transformation: compute logz from z, mask invalid to 0 to avoid nans.
    # Required: logz: shape (*b, 1), same as z; set logz[~valid_z] = 0.
    ****EMPTY****

    # Output
    pz = torch.cat((p, logz), dim=-1)

    if not jacobian:
        return pz, valid
    else:
        # Jacobian branch: compute dpz_dP shape (*b, 3, 3) using fx, fy, cx, cy, x, y, z.
        ****EMPTY****
        return pz, dpz_dP, valid
