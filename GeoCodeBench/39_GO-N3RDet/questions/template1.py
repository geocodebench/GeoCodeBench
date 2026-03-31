
"""
LLM Template - This file shows the format expected for LLM implementations.
LLMs should fill in the ****EMPTY**** sections based on the context.
No hints are provided - only input/output format.
"""

from collections import OrderedDict
import torch


def raw2outputs(raw, z_vals, mask, white_bkgd=False):
    """Transform raw data to outputs:

    Args:
        raw(tensor):Raw network output.Tensor of shape [N_rays, N_samples, 4]
        z_vals(tensor):Depth of point samples along rays.
            Tensor of shape [N_rays, N_samples]
        ray_d(tensor):[N_rays, 3]

    Returns:
        ret(dict):
            -rgb(tensor):[N_rays, 3]
            -depth(tensor):[N_rays,]
            -weights(tensor):[N_rays,]
            -depth_std(tensor):[N_rays,]
    """
    rgb = raw[:, :, :3]  # [N_rays, N_samples, 3]
    sigma = raw[:, :, 3]  # [N_rays, N_samples]

    # note: we did not use the intervals here,
    # because in practice different scenes from COLMAP can have
    # very different scales, and using interval can affect
    # the model's generalization ability.
    # Therefore we don't use the intervals for both training and evaluation.
    sigma2alpha = lambda sigma, dists: 1. - torch.exp(-sigma)  # noqa

    ****EMPTY****

    if white_bkgd:
        ****EMPTY****

    if mask is not None:
        mask = mask.float().sum(dim=1) > 8

    ****EMPTY****

    ret = OrderedDict([('rgb', rgb_map), ('depth', depth_map),
                       ('weights', weights), ('mask', mask), ('alpha', alpha),
                       ('z_vals', z_vals), ('transparency', T)])

    return ret
