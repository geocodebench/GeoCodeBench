
"""
Template for LLM Implementation
Copy this file and fill in the EMPTY part with LLM-generated code.
"""

import numpy as np


def solve_shift_and_scale_two_focal(x1_, x2_, d1, d2):
    # Normalize focal length
    x1 = x1_.copy()
    x2 = x2_.copy()

    f1_0 = np.abs(x1[:, 0:2].flatten()).mean()
    f2_0 = np.abs(x2[:, 0:2].flatten()).mean()
    x1[:, 0:2] /= f1_0
    x2[:, 0:2] /= f2_0

    # Compute coefficients
    ****EMPTY****

    # Extract solutions
    solutions = []

    for s in sols:
        s = np.real(s)
        if s[3] < 0 or s[4] < 0:
            continue
        solutions.append(
            (
                1.0,
                s[1],
                np.sqrt(s[0]),
                s[2] * np.sqrt(s[0]),
                f1_0 / np.sqrt(s[3]),
                f2_0 / np.sqrt(s[4]),
            )
        )

    return solutions
