
"""
Template for LLM Implementation (no hints)
Copy and let the LLM fill ONLY the ****EMPTY**** section.
"""

import torch
import torch.nn as nn


class SinkhornDistance(nn.Module):
    def __init__(self, epsilon=1.0, max_iter=100):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter

    def matrix_sqrt_eigh(self, M):
        epsilon = 1e-6 * torch.eye(M.size(-1), device=M.device, dtype=M.dtype)
        M_reg = M + epsilon
        eigenvalues, eigenvectors = torch.linalg.eigh(M_reg)
        sqrt_eigenvalues = torch.sqrt(eigenvalues.clamp(min=0.0))
        sqrt_M = eigenvectors @ (sqrt_eigenvalues.unsqueeze(-1) * eigenvectors.transpose(-2, -1))
        return sqrt_M

    def compute_cost_matrix(self, mu_A, cov_A, mu_B, cov_B):
        """
        W2距离矩阵的优化计算（GPU并行版，分块处理）
        """
        ****EMPTY****
        return mean_term + covariance_term
