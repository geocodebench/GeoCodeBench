"""
Test Data Generator for chol.py functions.
Generates test cases for CholeskySolver.apply, block_solve, schur_solve.
"""

from __future__ import annotations

import torch


class TestDataGenerator:
    """Generate test data for chol.py functions."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)
        # Ensure CPU device (no CUDA)
        self.device = torch.device('cpu')

    def generate_cholesky_test_cases(self, num_tests=5):
        """Generate test cases for CholeskySolver.apply(H, b)."""
        test_cases = []

        for i in range(num_tests):
            # Generate positive definite matrix H
            n = 3 + (i % 5)  # 3 to 7
            A = torch.randn(n, n, device=self.device)
            H = A @ A.transpose(-1, -2) + 0.1 * torch.eye(n, device=self.device)  # Make it positive definite
            b = torch.randn(n, 1, device=self.device)

            test_cases.append({
                'H': H,
                'b': b,
                'description': f'CholeskySolver.apply: n={n}',
            })

        return test_cases

    def generate_block_solve_test_cases(self, num_tests=5):
        """Generate test cases for block_solve(H, b, ep, lm)."""
        test_cases = []

        for i in range(num_tests):
            B = 1 + (i % 3)  # 1 to 3
            N = 2 + (i % 4)  # 2 to 5
            D = 3 + (i % 3)  # 3 to 5

            # Generate positive definite block matrices
            # H shape: [B, N, N, D, D]
            # After permute(0,1,3,2,4): [B, N, D, N, D]
            # After reshape: [B, N*D, N*D]
            # So we generate [B, N*D, N*D] positive definite, then reverse the operations
            H = torch.zeros(B, N, N, D, D, device=self.device)
            for b_idx in range(B):
                # Generate a random matrix and make it positive definite
                A_full = torch.randn(N*D, N*D, device=self.device)
                H_full = A_full @ A_full.transpose(-1, -2) + 0.5 * torch.eye(N*D, device=self.device)
                # Reshape to [N, D, N, D], then permute to [N, N, D, D]
                # The reverse of permute(0,1,3,2,4) is permute(0,1,3,2,4) (self-inverse)
                H_reshaped = H_full.view(N, D, N, D)
                H[b_idx] = H_reshaped.permute(0, 2, 1, 3)  # [N, N, D, D]

            b = torch.randn(B, N, D, device=self.device)
            ep = 0.1
            lm = 0.0001

            test_cases.append({
                'H': H,
                'b': b,
                'ep': ep,
                'lm': lm,
                'description': f'block_solve: B={B}, N={N}, D={D}',
            })

        return test_cases

    def generate_schur_solve_test_cases(self, num_tests=5):
        """Generate test cases for schur_solve(H, E, C, v, w, ep, lm, sless)."""
        test_cases = []

        for i in range(num_tests):
            B = 1 + (i % 2)  # 1 to 2
            P = 2 + (i % 3)  # 2 to 4
            M = 2 + (i % 3)  # 2 to 4
            D = 3 + (i % 2)  # 3 to 4
            HW = 4 + (i % 3)  # 4 to 6

            # Generate positive definite H
            # H shape: [B, P, P, D, D]
            # After permute(0,1,3,2,4): [B, P, D, P, D]
            # After reshape: [B, P*D, P*D]
            H = torch.zeros(B, P, P, D, D, device=self.device)
            for b_idx in range(B):
                # Generate a random matrix and make it positive definite
                A_full = torch.randn(P*D, P*D, device=self.device)
                H_full = A_full @ A_full.transpose(-1, -2) + 0.5 * torch.eye(P*D, device=self.device)
                # Reshape to [P, D, P, D], then permute to [P, P, D, D]
                H_reshaped = H_full.view(P, D, P, D)
                H[b_idx] = H_reshaped.permute(0, 2, 1, 3)  # [P, P, D, D]

            E = torch.randn(B, P, M, D, HW, device=self.device)
            C = torch.rand(B, M, HW, device=self.device) + 0.1  # Positive values
            v = torch.randn(B, P, D, device=self.device)
            w = torch.randn(B, M, HW, device=self.device)
            ep = 0.1
            lm = 0.0001
            sless = (i % 2 == 0)  # Alternate between True and False

            test_cases.append({
                'H': H,
                'E': E,
                'C': C,
                'v': v,
                'w': w,
                'ep': ep,
                'lm': lm,
                'sless': sless,
                'description': f'schur_solve: B={B}, P={P}, M={M}, D={D}, HW={HW}, sless={sless}',
            })

        return test_cases
