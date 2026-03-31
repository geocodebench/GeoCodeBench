
"""
LLM Template for project_simplex() function
This template shows the expected format for LLM implementations.
Only input and output are provided, no hints.
"""

import numpy as np


def project_simplex(V, z=1):
    """
    V: (num_samples, feature_dim)
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    """
    ****EMPTY****
