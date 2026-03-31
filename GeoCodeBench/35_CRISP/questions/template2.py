
"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
"""

import torch


def project_SO3(A):
    """Project a batched matrix to SO(3)"""
    U, S, Vh = torch.linalg.svd(A)
    d = torch.linalg.det(U) * torch.linalg.det(Vh)
    temp = U.clone()
    temp[:, :, -1] *= d[:, None]
    R = temp @ Vh
    return R, U, S, Vh, d


def arun_batched(source_points, target_points):
    """Run Arun's algorithm to estimate R, t (unweighted)
    target = R*source + t

    Note: this is a batched version
    """
    ****EMPTY****

    return R.to(og_dtype), t.to(og_dtype)
