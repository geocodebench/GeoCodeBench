"""
Reference Implementation for project_simplex()
This serves as the ground truth for testing LLM-generated implementations.
"""

import numpy as np


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
