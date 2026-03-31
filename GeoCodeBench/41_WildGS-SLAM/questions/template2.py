
"""
Template for LLM Implementation
Copy this file and fill in the EMPTY parts with LLM-generated code.
This template matches the question format - no hints provided.
"""

import torch
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from reference_implementation import MockProjectiveOps, MockPoses, schur_solve, scatter_sum


# utility functions for scattering ops (from ba.py lines 24-45)
def safe_scatter_add_mat(A, ii, jj, n, m):
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)

def safe_scatter_add_vec(b, ii, n):
    v = (ii >= 0) & (ii < n)
    return scatter_sum(b[:,v], ii[v], dim=1, dim_size=n)

# apply retraction operator to inv-depth maps
def disp_retr(disps, dz, ii):
    ii = ii.to(device=dz.device)
    return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])

def wq_retr(wqs, dwq, ii):
    ii = ii.to(device=dwq.device)
    return wqs + scatter_sum(dwq, ii, dim=1, dim_size=wqs.shape[1])

# apply retraction operator to poses
def pose_retr(poses, dx, ii):
    ii = ii.to(device=dx.device)
    return poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))


@torch.no_grad()
def BA_with_scale_shift(target, weight, eta, poses, disps, intrinsics, ii, jj, 
       mono_disps, scales=None, shifts=None, 
       valid_depth_mask=None, ignore_frames=0,
       lm=0.0001, ep=0.1, alpha=1.0, fixedp=1, rig=1):
    """ optimize disparities (disp), scales (w) and shifts (q) together, eq.17 in the paper,
        math details can be found in the supplementary
    """
    device = ii.device
    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim
    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]
    sqrt_alpha = torch.tensor(alpha).sqrt().to(device)
    ll = torch.arange(M,device=device)
    wqs = torch.stack([scales,shifts],dim=2)         #[B,P,2]

    ignore_mask = kx<ignore_frames
    invalid_mask = (mono_disps[:,kx]<1e-6).view(B,M,ht*wd)    #[B,M,ht*wd]
    invalid_mask[:,ignore_mask] = True

    valid_depth_mask = valid_depth_mask[:,kx].view(B,M,ht*wd)
    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = MockProjectiveOps.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)

    r = (target - coords).view(B, N, -1, 1)           #[B,N,ht*wq*2,1]
    r_depth = sqrt_alpha * (disps[:,kx]-(scales[:,kx,None,None]*mono_disps[:,kx]+shifts[:,kx,None,None])).view(B,M,ht*wd,1)

    w = .001 * (valid * weight).view(B, N, -1, 1)

    ****EMPTY****
    
    disps = disps.clamp(min=0.0)

    return poses, disps , wqs
