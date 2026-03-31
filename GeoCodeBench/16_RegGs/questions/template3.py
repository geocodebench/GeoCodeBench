
from __future__ import annotations

import torch
import torch.nn.functional as F


def build_rotation(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternions to rotation matrices.

    Args:
        quaternions: Tensor of shape [B, 4] in (w, x, y, z) order.

    Returns:
        Rotation matrices of shape [B, 3, 3].
    """
    assert quaternions.dim() == 2 and quaternions.size(1) == 4
    q = F.normalize(quaternions, dim=1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    r00 = ww + xx - yy - zz
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)

    r10 = 2 * (xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2 * (yz - wx)

    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = ww - xx - yy + zz

    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)
    return R


class SinkhornDistance(torch.nn.Module):
    """
    Lightweight 2-Wasserstein cost aggregation between two Gaussian mixtures.
    Uses pairwise W2 cost and aggregates with outer-product of weights (CPU-friendly).
    """

    def __init__(self, epsilon: float = 0.1, max_iter: int = 100):
        super().__init__()
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)

    @staticmethod
    def _matrix_sqrt_eigh(cov: torch.Tensor) -> torch.Tensor:
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = torch.clamp(eigvals, min=1e-12)
        sqrt_eigvals = torch.sqrt(eigvals)
        return eigvecs @ torch.diag_embed(sqrt_eigvals) @ eigvecs.transpose(-1, -2)

    def _pairwise_w2_cost(self, mu_a: torch.Tensor, cov_a: torch.Tensor,
                           mu_b: torch.Tensor, cov_b: torch.Tensor) -> torch.Tensor:
        diff = mu_a.unsqueeze(1) - mu_b.unsqueeze(0)  # [Na, Nb, 3]
        mean_term = torch.sum(diff * diff, dim=-1)   # [Na, Nb]

        sqrt_a = self._matrix_sqrt_eigh(cov_a)       # [Na, 3, 3]
        a_exp = sqrt_a.unsqueeze(1)                  # [Na, 1, 3, 3]
        b_exp = cov_b.unsqueeze(0)                   # [1, Nb, 3, 3]
        mid = a_exp @ b_exp @ a_exp.transpose(-1, -2)  # [Na, Nb, 3, 3]

        Na, Nb = mean_term.shape
        mid_flat = mid.reshape(Na * Nb, 3, 3)
        sqrt_mid_flat = self._matrix_sqrt_eigh(mid_flat)
        sqrt_mid = sqrt_mid_flat.reshape(Na, Nb, 3, 3)

        trace_a = torch.einsum('bii->b', cov_a).unsqueeze(1)  # [Na, 1]
        trace_b = torch.einsum('bii->b', cov_b).unsqueeze(0)  # [1, Nb]
        trace_term = trace_a + trace_b - 2.0 * torch.einsum('abii->ab', sqrt_mid)
        return mean_term + trace_term

    def forward(self,
                mu_a: torch.Tensor, cov_a: torch.Tensor, w_a: torch.Tensor,
                mu_b: torch.Tensor, cov_b: torch.Tensor, w_b: torch.Tensor) -> torch.Tensor:
        w_a = w_a / (w_a.sum() + 1e-12)
        w_b = w_b / (w_b.sum() + 1e-12)
        C = self._pairwise_w2_cost(mu_a, cov_a, mu_b, cov_b)  # [Na, Nb]
        loss = torch.sum(C * (w_a.unsqueeze(1) * w_b.unsqueeze(0)))
        return loss


class TemplateAligner:
    def compute_mw2_loss(self, main_component: tuple, mini_component: tuple,
                         w2c_i: torch.Tensor,
                         opt_scale: torch.Tensor,
                         opt_cam_rot: torch.Tensor,
                         opt_cam_trans: torch.Tensor) -> torch.Tensor:

        w2c_r = w2c_i[:3, :3]
        w2c_t = w2c_i[:3, 3]

        rel_R = build_rotation(F.normalize(opt_cam_rot[None])).squeeze(0)
        rel_T = opt_cam_trans

        # hint
        # sinkhorn = SinkhornDistance(epsilon=0.1)

        ****EMPTY****
        
        return loss
