"""
Test data generator for compute_mw2_loss (TemplateAligner).
"""

from __future__ import annotations

import torch
from typing import Any, Dict, List

from reference_implementation import build_rotation


class TestDataGenerator:
    def __init__(self, seed: int = 1234):
        self.seed = seed
        torch.manual_seed(seed)

    @staticmethod
    def _make_spd(batch: int) -> torch.Tensor:
        A = torch.randn(batch, 3, 3)
        cov = A @ A.transpose(1, 2)
        cov = cov + 1e-3 * torch.eye(3).unsqueeze(0)
        return cov

    @staticmethod
    def _rand_quat() -> torch.Tensor:
        q = torch.randn(4)
        q = q / (q.norm() + 1e-12)
        return q

    def _rand_w2c(self) -> torch.Tensor:
        R = build_rotation(self._rand_quat().unsqueeze(0)).squeeze(0)
        t = torch.randn(3)
        w2c = torch.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        return w2c

    def generate(self, num_tests: int) -> List[Dict[str, Any]]:
        tests: List[Dict[str, Any]] = []
        for i in range(num_tests):
            torch.manual_seed(self.seed + i)
            na = 60 + (i % 5) * 10
            nb = 55 + (i % 4) * 15
            main_mu = torch.randn(na, 3)
            main_cov = self._make_spd(na)
            main_w = torch.softmax(torch.randn(na), dim=0)

            mini_mu = torch.randn(nb, 3)
            mini_cov = self._make_spd(nb)
            mini_w = torch.softmax(torch.randn(nb), dim=0)

            w2c = self._rand_w2c()
            scale = torch.exp(torch.randn(()) / 3.0)
            cam_rot = self._rand_quat()
            cam_trans = torch.randn(3)

            tests.append({
                'main_component': (main_mu, main_cov, main_w),
                'mini_component': (mini_mu, mini_cov, mini_w),
                'w2c': w2c,
                'scale': scale,
                'cam_rot': cam_rot,
                'cam_trans': cam_trans,
                'description': f'Na={na}, Nb={nb}'
            })
        return tests
