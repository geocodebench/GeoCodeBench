"""
Reference Implementation for shape_recovery_from_pc_cvxpy()
This serves as the ground truth for testing LLM-generated implementations.
"""

import numpy as np
import cvxpy as cp
import torch
from torch import nn as nn


def project_simplex(V, z=1):
    """
    V: (num_samples, feature_dim)
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V

    Credit: https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89
    Paper: https://www.jmlr.org/papers/volume7/shalev-shwartz06a/shalev-shwartz06a.pdf
    """
    n_features = V.shape[1]
    U = np.sort(V, axis=1)[:, ::-1]
    z = np.ones(len(V)) * z
    cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
    ind = np.arange(n_features) + 1
    cond = U - cssv / ind > 0
    rho = np.count_nonzero(cond, axis=1)
    theta = cssv[np.arange(len(V)), rho - 1] / rho
    return np.maximum(V - theta[:, np.newaxis], 0)


def create_F_matrix(sdf_model, shape_code_mat, query_points, normalize_by_extent=False):
    """Helper function to create F matrix (LSQ formulation)"""

    def f_sdf_conditioned(shape_code, x):
        return sdf_model.forward(shape_code=shape_code, coords=x)

    B = shape_code_mat.shape[0]
    K = shape_code_mat.shape[2]
    N = query_points.shape[1]

    if normalize_by_extent:
        # TODO: Potentially use quantile to get the 90% max / 10% min
        delta_xyz = (
            torch.max(query_points, dim=1, keepdim=True).values - torch.min(query_points, dim=1, keepdim=True).values
        )
        scales = torch.norm(delta_xyz, dim=2, keepdim=True)
        # set bounds on scales between 1e-4 and 1e4 to avoid numerical issues
        scales = torch.max(torch.ones_like(scales) * 1e-4, torch.min(scales, torch.ones_like(scales) * 1e4))
    else:
        scales = torch.ones((B, 1, 1), device=query_points.device)

    # create F matrix
    F_all = np.zeros((B, N, K))
    with torch.no_grad():
        for k in range(K):
            F_all[:, :, k] = (
                (f_sdf_conditioned(shape_code_mat[:, :, k], query_points) / scales).squeeze(-1).cpu().numpy()
            )
    return F_all


def shape_recovery_from_pc_cvxpy(
    sdf_model,
    initial_shape_code,
    nocs,
    masks,
    shape_code_library,
    use_L1_reg=False,
    use_onehot=False,
    use_initial_shape_code_basis=False,
    normalize_F_matrix=False,
    L1_weight=5,
):
    # shape coefficients to be optimized
    # note: this is not in simplex
    N = nocs.shape[1]
    og_K, latent_dim = shape_code_library.shape[1], shape_code_library.shape[0]
    if use_initial_shape_code_basis:
        K = og_K + 1
    else:
        K = og_K
    B = nocs.shape[0]

    # create shape code mat
    shape_code_mat = np.zeros((B, latent_dim, K), dtype=np.float32)
    for b in range(B):
        if use_initial_shape_code_basis:
            shape_code_mat[b, :, :og_K] = shape_code_library
            shape_code_mat[b, :, -1] = initial_shape_code[b, ...].detach().cpu().flatten().numpy()
        else:
            shape_code_mat[b, ...] = shape_code_library
    shape_code_mat = torch.tensor(shape_code_mat).to(nocs.device)

    # create F matrix
    F_all = create_F_matrix(sdf_model, shape_code_mat, nocs, normalize_by_extent=normalize_F_matrix)

    with torch.no_grad():
        postcrt_shape_codes = torch.zeros((B, latent_dim), device=nocs.device)
        best_shape_coeffs = np.zeros((B, og_K))
        solver_statuses = []
        for b in range(B):
            # formulate the problem
            F = F_all[b, ...]
            c = cp.Variable(K)
            if use_L1_reg:
                cost = cp.sum_squares(F @ c) + L1_weight * cp.norm1(c)
            else:
                cost = cp.sum_squares(F @ c)
            prob = cp.Problem(cp.Minimize(cost), [cp.sum(c) == 1, c >= 0])
            prob.solve(solver="CLARABEL")
            coeffs_sol = c.value.flatten()
            c_shape_code_mat = shape_code_mat[b, ...].cpu().numpy()
            postcrt_shp = c_shape_code_mat @ coeffs_sol
            if use_initial_shape_code_basis:
                # get the coefficients if we are using the original shape code bases
                temp_coeffs = np.linalg.lstsq(shape_code_library, postcrt_shp)[0]
                # project to original simplex
                projected_coeffs = project_simplex(temp_coeffs.reshape((1, -1)))
                # compute newly projected shape code
                postcrt_shp = shape_code_library @ projected_coeffs.flatten()
                coeffs_sol = projected_coeffs.flatten()
            postcrt_shape_codes[b, ...] = torch.tensor(postcrt_shp).float()

            if use_onehot:
                # project to onehot vector and recompute post correction shape code
                max_index = np.argmax(coeffs_sol)
                best_shape_coeffs[b, max_index] = 1
                postcrt_shape_codes[b, ...] = torch.tensor(shape_code_library @ best_shape_coeffs[b, ...]).float()
            else:
                best_shape_coeffs[b, :] = coeffs_sol

            solver_statuses.append((prob.status, prob.solver_stats))

    return {
        "postcrt_shape_code": postcrt_shape_codes,
        "postcrt_shape_coeffs": torch.tensor(best_shape_coeffs),
        "solver_statuss": solver_statuses,
    }
