"""
Reference implementation exposing SinkhornDistance.compute_cost_matrix
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
        # Mean term
        mu_diff = mu_A.unsqueeze(1) - mu_B.unsqueeze(0)
        mean_term = torch.sum(mu_diff**2, dim=-1)

        # Covariance term
        tr_cov_A = torch.einsum('...ii->...', cov_A)
        tr_cov_B = torch.einsum('...ii->...', cov_B)

        sqrt_cov_A = self.matrix_sqrt_eigh(cov_A)

        batch_A, d = mu_A.shape
        batch_B = mu_B.shape[0]
        chunk_size = 256
        device = mu_A.device
        dtype = mu_A.dtype

        covariance_term = torch.zeros((batch_A, batch_B), device=device, dtype=dtype)

        for i in range(0, batch_A, chunk_size):
            for j in range(0, batch_B, chunk_size):
                i_end = min(i + chunk_size, batch_A)
                j_end = min(j + chunk_size, batch_B)

                sqrt_cov_A_chunk = sqrt_cov_A[i:i_end]
                cov_B_chunk = cov_B[j:j_end]
                chunk_A_size = sqrt_cov_A_chunk.size(0)
                chunk_B_size = cov_B_chunk.size(0)

                sqrt_expanded = sqrt_cov_A_chunk.unsqueeze(1)
                cov_expanded = cov_B_chunk.unsqueeze(0)
                temp = torch.matmul(sqrt_expanded, cov_expanded)
                M_chunk = torch.matmul(temp, sqrt_expanded)

                M_flatten = M_chunk.view(-1, d, d)
                sqrt_M_flatten = self.matrix_sqrt_eigh(M_flatten)
                sqrt_M_chunk = sqrt_M_flatten.view(chunk_A_size, chunk_B_size, d, d)

                trace_sqrtM = torch.einsum('abii->ab', sqrt_M_chunk)

                tr_A_chunk = tr_cov_A[i:i_end].unsqueeze(1)
                tr_B_chunk = tr_cov_B[j:j_end].unsqueeze(0)
                covariance_term[i:i_end, j:j_end] = tr_A_chunk + tr_B_chunk - 2 * trace_sqrtM

        return mean_term + covariance_term


