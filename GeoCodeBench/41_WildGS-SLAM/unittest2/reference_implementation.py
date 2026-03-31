"""
Reference Implementation for BA_with_scale_shift()
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch
import torch.nn.functional as F

# Try to import torch_scatter, fallback to simple implementation
try:
    from torch_scatter import scatter_sum
except ImportError:
    # Fallback implementation of scatter_sum using torch operations
    def scatter_sum(src, index, dim=0, dim_size=None):
        """
        Fallback scatter_sum implementation for testing.
        This is a simplified version that works for our use cases.
        """
        if dim_size is None:
            dim_size = int(index.max().item()) + 1
        
        # Get device and dtype
        device = src.device
        dtype = src.dtype
        
        # Create output tensor
        out_shape = list(src.shape)
        out_shape[dim] = dim_size
        out = torch.zeros(out_shape, dtype=dtype, device=device)
        
        # Use index_add for efficient scatter
        if dim == 1:  # Most common case in our code
            # src: [B, N, ...], index: [N], dim=1
            B = src.shape[0]
            for b in range(B):
                for i in range(index.shape[0]):
                    idx = int(index[i].item())
                    if 0 <= idx < dim_size:
                        out[b, idx] += src[b, i]
        elif dim == 0:
            # src: [N, ...], index: [N], dim=0
            for i in range(index.shape[0]):
                idx = int(index[i].item())
                if 0 <= idx < dim_size:
                    out[idx] += src[i]
        else:
            # General case - use loop (slower but works)
            indices = [slice(None)] * src.ndim
            for i in range(index.shape[0]):
                idx = int(index[i].item())
                if 0 <= idx < dim_size:
                    indices[dim] = idx
                    src_indices = [slice(None)] * src.ndim
                    src_indices[dim] = i
                    out[tuple(indices)] += src[tuple(src_indices)]
        
        return out


# Mock projective_ops module
class MockProjectiveOps:
    @staticmethod
    def projective_transform(poses, disps, intrinsics, ii, jj, jacobian=True):
        """
        Mock projective_transform function.
        Returns: coords, valid, (Ji, Jj, Jz)
        Uses deterministic values based on input shapes for reproducibility.
        """
        B, P, ht, wd = disps.shape
        N = ii.shape[0]
        D = poses.manifold_dim
        
        # Use deterministic values based on input shapes and indices
        # This ensures same inputs produce same outputs
        device = poses.device
        
        # Create deterministic coords based on disps and indices
        coords = torch.zeros(B, N, ht, wd, 2, device=device)
        for b in range(B):
            for n in range(N):
                i_idx = ii[n].item()
                j_idx = jj[n].item()
                # Use disps to create deterministic coords
                coords[b, n, :, :, 0] = disps[b, i_idx, :, :] * 0.5 + disps[b, j_idx, :, :] * 0.3
                coords[b, n, :, :, 1] = disps[b, i_idx, :, :] * 0.4 + disps[b, j_idx, :, :] * 0.2
        
        valid = torch.ones(B, N, ht, wd, device=device, dtype=torch.bool)
        
        if jacobian:
            # Create deterministic jacobians
            Ji = torch.zeros(B, N, ht, wd, D, device=device)
            Jj = torch.zeros(B, N, ht, wd, D, device=device)
            Jz = torch.zeros(B, N, ht, wd, 2, device=device)
            
            # Fill with deterministic values based on disps
            for b in range(B):
                for n in range(N):
                    i_idx = ii[n].item()
                    j_idx = jj[n].item()
                    # Use disps to create deterministic jacobians
                    Ji[b, n, :, :, :] = disps[b, i_idx, :, :, None].expand(-1, -1, D) * 0.1
                    Jj[b, n, :, :, :] = disps[b, j_idx, :, :, None].expand(-1, -1, D) * 0.1
                    Jz[b, n, :, :, 0] = disps[b, i_idx, :, :] * 0.05
                    Jz[b, n, :, :, 1] = disps[b, j_idx, :, :] * 0.05
            
            return coords, valid, (Ji, Jj, Jz)
        else:
            return coords, valid


# Mock lietorch poses
class MockPoses:
    """Mock poses object with manifold_dim and retr method."""
    def __init__(self, B, P, device='cpu'):
        self.B = B
        self.P = P
        self.manifold_dim = 6  # SE(3) has 6 dimensions
        self.device = device
        # Store pose data as a tensor
        self.data = torch.randn(B, P, 4, 4, device=device)
    
    def retr(self, dx):
        """Retraction operator for poses."""
        # dx: [B, P, D] where D=6
        # Simple mock: just add a small perturbation
        return self


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


# Mock schur_solve from chol.py
def schur_solve(H, E, C, v, w, ep=0.1, lm=0.0001, sless=False):
    """ solve using shur complement """
    B, P, M, D, HW = E.shape
    H = H.permute(0,1,3,2,4).reshape(B, P*D, P*D)
    E = E.permute(0,1,3,2,4).reshape(B, P*D, M*HW)
    Q = (1.0 / C).view(B, M*HW, 1)

    # damping
    I = torch.eye(P*D).to(H.device)
    H = H + (ep + lm*H) * I
    
    v = v.reshape(B, P*D, 1)
    w = w.reshape(B, M*HW, 1)

    Et = E.transpose(1,2)
    S = H - torch.matmul(E, Q*Et)
    v = v - torch.matmul(E, Q*w)

    # Use Cholesky solve
    try:
        U = torch.linalg.cholesky(S)
        dx = torch.cholesky_solve(v, U)
    except:
        dx = torch.zeros_like(v)
    
    if sless:
        return dx.reshape(B, P, D)

    dz = Q * (w - Et @ dx)    
    dx = dx.reshape(B, P, D)
    dz = dz.reshape(B, M, HW)

    return dx, dz


# Reference implementation of BA_with_scale_shift
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

    sqrt_alpha = torch.ones(B,M,ht*wd,1).float().to(device) * sqrt_alpha
    sqrt_alpha[valid_depth_mask] *= 10

    J_d = torch.ones(B,M,ht*wd,1).float().to(device) * sqrt_alpha
    J_scale = -mono_disps[:,kx].clone().view(B,M,ht*wd,1) * sqrt_alpha#[B,M,ht*wd,1]
    J_shift = -torch.ones(B,M,ht*wd,1).float().to(device) * sqrt_alpha#[B,M,ht*wd,1]

    J_d[invalid_mask*valid_depth_mask] = 0
    J_scale[invalid_mask] = 0
    J_shift[invalid_mask] = 0

    J_wq = torch.cat([J_scale,J_shift],dim=3)         #[B,M,ht*wd,2]
    J_wq_T = J_wq.transpose(2,3)         #[B,M,2,ht*wd]
    H_wq = torch.matmul(J_wq_T, J_wq)   #[B,M,2,2]
    u = - torch.matmul(J_wq_T, r_depth).squeeze(-1) #[B,M,2]
    ### 2: construct linear system ###

    Jz = Jz.reshape(B, N, ht*wd, -1)    #[B,N,ht*wd,2] 
    # here Jz does not contain the negative sign in the residual term 

    E_wq_d = (J_wq_T.view(B,M,2,ht*wd,-1) * J_d[:,:,None]).sum(dim=-1) #[B,M,2,ht*wd]

    w = w.view(B, N, ht*wd, -1) #[B,N,ht*wd,2]
    r = r.view(B, N, ht*wd, -1) #[B,N,ht*wd,2]
    wk = torch.sum(-w*r*Jz, dim=-1) #[B,N,ht*wd]
    Ck = torch.sum(w*(-Jz)*(-Jz), dim=-1) #[B,N,ht*wd]

    # only optimize keyframe poses
    P = torch.div(P,rig,rounding_mode="trunc")-fixedp
    ii = torch.div(ii,rig,rounding_mode="trunc")-fixedp
    jj = torch.div(jj,rig,rounding_mode="trunc")-fixedp

    H_wq = safe_scatter_add_mat(H_wq,ll,ll,M,M)       #[B,M*M,2,2]
    E_wq_d = safe_scatter_add_mat(E_wq_d,ll,ll,M,M)      #[B,M*M,2,ht*wd]
    C_proj = safe_scatter_add_vec(Ck, kk, M)                #[B,M,ht*wd]
    u = safe_scatter_add_vec(u, ll, M)                      #[B,M,2]

    # C = C + eta.view(*C.shape) #+ 1e-7
    C_depth = (J_d*J_d).view(B,M,ht*wd)
    # C = C_proj + C_depth + (1-C_depth)*eta.view(*C_proj.shape)             #[B,M,ht*wd]
    # eta is [B, P, ht, wd], need to index with kx and reshape to [B, M, ht*wd]
    eta_kx = eta[:, kx].view(B, M, ht*wd)  # [B, M, ht*wd]
    C = C_proj + C_depth + eta_kx #+ 1e-7

    w_proj = safe_scatter_add_vec(wk, kk, M)                               #[B,M,ht*wd]
    w = -w_proj - (J_d*r_depth).view(B,M,ht*wd)  #[B,M,ht*wd]
    H = H_wq.view(B, M, M, 2, 2)
    E = E_wq_d.view(B, M, M, 2, ht*wd)
    ### 3: solve the system ###    
    dwq, dz = schur_solve(H, E, C, u, w, ep, lm)
    # dwq [B,M,2]
    # dz [B,M,ht*wd]
    ### 4: apply retraction ###
    # poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)
    wqs = wq_retr(wqs,dwq,kx)
    # disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.0)

    return poses, disps , wqs
