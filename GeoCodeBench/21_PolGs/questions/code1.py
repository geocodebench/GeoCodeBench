
import numpy as np



def normalize(v):
    import torch
    return torch.nn.functional.normalize(v, dim=-1)

def normalize_np(v):
    return  v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),
                     1e-12)

def stokes_fac_from_normal(rays_o, rays_d, normal, 
                           train_mode=False,
                           ret_spec=False,
                           clip_spec=False):
    import torch

    # Add singleton dimension for Num_lights
    rays_d = rays_d[..., None, :]
    normal = normal[..., None, :]

    ****EMPTY****

    stokes_diff_fac = stack([torch.ones_like(T_o__min), T_o__min / T_o__plus * cos(2 * phi_o), -T_o__min / T_o__plus * sin(2 * phi_o)], -1)
    stokes_spec_fac = stack([torch.ones_like(R__plus), R__min / R__plus * cos(2 * psi_o), -R__min / R__plus * sin(2 * psi_o)], -1)

    if clip_spec:
        spec_mask = dotno > 1e-7
        stokes_spec_fac = mask_fn(stokes_spec_fac, spec_mask[..., None])
        R__plus = mask_fn(R__plus, spec_mask)

    return stokes_diff_fac, stokes_spec_fac, R__plus[..., None]
