"""
Reference Implementation for update_pose
This serves as the ground truth for testing LLM-generated implementations.
"""

import numpy as np
import torch


class MockCamera:
    """Mock camera class for testing update_pose function."""
    
    def __init__(self, R, T, cam_trans_delta=None, cam_rot_delta=None, device='cpu'):
        self.R = R.clone() if isinstance(R, torch.Tensor) else torch.tensor(R, dtype=torch.float32, device=device)
        self.T = T.clone() if isinstance(T, torch.Tensor) else torch.tensor(T, dtype=torch.float32, device=device)
        
        if cam_trans_delta is None:
            self.cam_trans_delta = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)
        else:
            self.cam_trans_delta = cam_trans_delta.clone() if isinstance(cam_trans_delta, torch.Tensor) else torch.tensor(cam_trans_delta, dtype=torch.float32, device=device, requires_grad=True)
        
        if cam_rot_delta is None:
            self.cam_rot_delta = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)
        else:
            self.cam_rot_delta = cam_rot_delta.clone() if isinstance(cam_rot_delta, torch.Tensor) else torch.tensor(cam_rot_delta, dtype=torch.float32, device=device, requires_grad=True)
    
    def update_RT(self, new_R, new_T):
        """Update rotation and translation matrices."""
        self.R = new_R.clone()
        self.T = new_T.clone()


def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    if torch.isnan(W).any():
        print("Matrix W contains NAN")
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device
    if torch.isnan(tau).any():
        print("Matrix W contains NAN")
        return torch.eye(4, device=device, dtype=dtype)
    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def update_pose(camera, converged_threshold=1e-4):
    """Reference implementation of update_pose function."""
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)

    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = camera.R
    T_w2c[0:3, 3] = camera.T

    new_w2c = SE3_exp(tau) @ T_w2c

    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]

    converged = (tau.norm() < converged_threshold).item()
    camera.update_RT(new_R, new_T)

    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged
