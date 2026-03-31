"""
Reference Implementation for raw2outputs() function
This serves as the ground truth for testing LLM-generated implementations.
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

    # point samples are ordered with increasing depth
    # interval between samples
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat((dists, dists[:, -1:]), dim=-1)

    alpha = sigma2alpha(sigma, dists)

    T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[:, :-1]
    T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)

    # maths show weights, and summation of weights along a ray,
    # are always inside [0, 1]
    weights = alpha * T
    rgb_map = torch.sum(weights.unsqueeze(2) * rgb, dim=1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - torch.sum(weights, dim=-1, keepdim=True))

    if mask is not None:
        mask = mask.float().sum(dim=1) > 8

    depth_map = torch.sum(
        weights * z_vals, dim=-1) / (
            torch.sum(weights, dim=-1) + 1e-8)
    depth_map = torch.clamp(depth_map, z_vals.min(), z_vals.max())

    ret = OrderedDict([('rgb', rgb_map), ('depth', depth_map),
                       ('weights', weights), ('mask', mask), ('alpha', alpha),
                       ('z_vals', z_vals), ('transparency', T)])

    return ret
