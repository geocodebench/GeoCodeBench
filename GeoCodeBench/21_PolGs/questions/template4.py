
"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
"""

import numpy as np
import torch

pixel_camera = None

def sample_camera_rays(HWK, R, T):
    H, W, K = HWK
    R = R.T  # NOTE!!! the R rot matrix is transposed save in 3DGS
    global pixel_camera
    # TODO: Fill in LLM-generated code here
    raise NotImplementedError("Please implement this function")


def reflection(rayd, normal):
    # TODO: Fill in LLM-generated code here
    raise NotImplementedError("Please implement this function")
    return refl


def sample_cubemap_color(rays_d, env_map):
    # TODO: Fill in LLM-generated code here
    raise NotImplementedError("Please implement this function")
    return outcolor


def get_refl_color(envmap, HWK, R, T, normal_map):  # RT W2C
    # TODO: Fill in LLM-generated code here
    raise NotImplementedError("Please implement this function")
    return sample_cubemap_color(rays_d, envmap)
