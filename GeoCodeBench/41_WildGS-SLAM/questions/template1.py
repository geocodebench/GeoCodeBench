
# Copyright 2024 The GlORIE-SLAM Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
try:
    import src.geom.projective_ops as pops
except ImportError:
    pass  # Optional import, may not be available in test environment

# class CholeskySolver(torch.autograd.Function):
class CholeskySolver():
    @staticmethod
    def apply(H, b):
        """
        Solve H @ x = b via Cholesky decomposition (forward only, no backward).
        Input:
            H: torch.Tensor, shape (..., n, n), positive definite.
            b: torch.Tensor, shape (..., n, 1) or (..., n), same batch dims as H.
        Output:
            xs: torch.Tensor, same shape as b. On Cholesky failure, return zeros_like(b).
        """
        ****EMPTY****

        return xs

    def __call__(ctx, H, b):
        """
        Solve H @ x = b via Cholesky (differentiable). Save U, xs for backward.
        Input:
            H: torch.Tensor, shape (..., n, n), positive definite.
            b: torch.Tensor, shape (..., n, 1) or (..., n).
        Output:
            xs: torch.Tensor, same shape as b. On failure set ctx.failed=True, return zeros_like(b).
        """
        # don't crash training if cholesky decomp fails
        ****EMPTY****

        return xs

    @staticmethod
    def backward(ctx, grad_x):
        """
        Backward of __call__. grad_x has same shape as xs.
        Input: ctx (with saved U, xs; ctx.failed), grad_x.
        Output: (dH, dz) - gradients for H and b; same shapes as H and b. Return (None, None) if ctx.failed.
        """
        if ctx.failed:
            return None, None

        ****EMPTY****

        return dH, dz


def block_solve(H, b, ep=0.1, lm=0.0001):
    """
    Solve block normal equations (H + damping) @ x = b.
    Input:
        H: torch.Tensor, shape (B, N, N, D, D) - block matrix, N×N blocks each D×D.
        b: torch.Tensor, shape (B, N, D).
        ep, lm: float, damping (default 0.1, 0.0001).
    Output:
        x: torch.Tensor, shape (B, N, D).
    """
    ****EMPTY****
    return x.reshape(B, N, D)


def schur_solve(H, E, C, v, w, ep=0.1, lm=0.0001, sless=False):
    """
    Solve Schur complement system (camera-point split).
    Input:
        H: torch.Tensor, shape (B, P, P, D, D) - camera block Hessian.
        E: torch.Tensor, shape (B, P, M, D, HW) - cross camera-point blocks.
        C: torch.Tensor, shape (B, M, HW) - point diagonal (positive), used as 1/C.
        v: torch.Tensor, shape (B, P, D) - camera RHS.
        w: torch.Tensor, shape (B, M, HW) - point RHS.
        ep, lm: float, damping.
        sless: bool. If True, return only dx; else return (dx, dz).
    Output:
        If sless=True: dx, shape (B, P, D).
        If sless=False: (dx, dz), dx shape (B, P, D), dz shape (B, M, HW).
    """
    ****EMPTY****
    # if sless: return dx (shape (B, P, D)); else: return (dx, dz) with dz shape (B, M, HW)
    return dx, dz
