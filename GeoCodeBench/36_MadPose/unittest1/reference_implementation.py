"""
Reference Implementation for solve_shift_and_scale()
This serves as the ground truth for testing LLM-generated implementations.
"""

import numpy as np


def solve_shift_and_scale(x1, x2, d1, d2):
    # Compute coefficients
    coeffs = np.zeros((18,))
    coeffs[0] = 2 * x2[0].dot(x2[1]) - x2[0].dot(x2[0]) - x2[1].dot(x2[1])
    coeffs[1] = x1[0].dot(x1[0]) + x1[1].dot(x1[1]) - 2 * x1[0].dot(x1[1])
    coeffs[2] = (
        2 * (d2[0] + d2[1]) * x2[0].dot(x2[1])
        - 2 * d2[0] * x2[0].dot(x2[0])
        - 2 * d2[1] * x2[1].dot(x2[1])
    )
    coeffs[3] = (
        2 * d2[0] * d2[1] * x2[0].dot(x2[1])
        - d2[0] * d2[0] * x2[0].dot(x2[0])
        - d2[1] * d2[1] * x2[1].dot(x2[1])
    )
    coeffs[4] = (
        2 * d1[0] * x1[0].dot(x1[0])
        + 2 * d1[1] * x1[1].dot(x1[1])
        - 2 * (d1[0] + d1[1]) * x1[0].dot(x1[1])
    )
    coeffs[5] = (
        d1[0] * d1[0] * x1[0].dot(x1[0])
        + d1[1] * d1[1] * x1[1].dot(x1[1])
        - 2 * d1[0] * d1[1] * x1[0].dot(x1[1])
    )
    coeffs[6] = 2 * x2[0].dot(x2[2]) - x2[0].dot(x2[0]) - x2[2].dot(x2[2])
    coeffs[7] = x1[0].dot(x1[0]) + x1[2].dot(x1[2]) - 2 * x1[0].dot(x1[2])
    coeffs[8] = (
        2 * (d2[0] + d2[2]) * x2[0].dot(x2[2])
        - 2 * d2[0] * x2[0].dot(x2[0])
        - 2 * d2[2] * x2[2].dot(x2[2])
    )
    coeffs[9] = (
        2 * d2[0] * d2[2] * x2[0].dot(x2[2])
        - d2[0] * d2[0] * x2[0].dot(x2[0])
        - d2[2] * d2[2] * x2[2].dot(x2[2])
    )
    coeffs[10] = (
        2 * d1[0] * x1[0].dot(x1[0])
        + 2 * d1[2] * x1[2].dot(x1[2])
        - 2 * (d1[0] + d1[2]) * x1[0].dot(x1[2])
    )
    coeffs[11] = (
        d1[0] * d1[0] * x1[0].dot(x1[0])
        + d1[2] * d1[2] * x1[2].dot(x1[2])
        - 2 * d1[0] * d1[2] * x1[0].dot(x1[2])
    )
    coeffs[12] = 2 * x2[1].dot(x2[2]) - x2[1].dot(x2[1]) - x2[2].dot(x2[2])
    coeffs[13] = x1[1].dot(x1[1]) + x1[2].dot(x1[2]) - 2 * x1[1].dot(x1[2])
    coeffs[14] = (
        2 * (d2[1] + d2[2]) * x2[1].dot(x2[2])
        - 2 * d2[1] * x2[1].dot(x2[1])
        - 2 * d2[2] * x2[2].dot(x2[2])
    )
    coeffs[15] = (
        2 * d2[1] * d2[2] * x2[1].dot(x2[2])
        - (d2[1] ** 2) * x2[1].dot(x2[1])
        - (d2[2] ** 2) * x2[2].dot(x2[2])
    )
    coeffs[16] = (
        2 * d1[1] * x1[1].dot(x1[1])
        + 2 * d1[2] * x1[2].dot(x1[2])
        - 2 * (d1[1] + d1[2]) * x1[1].dot(x1[2])
    )
    coeffs[17] = (
        d1[1] * d1[1] * x1[1].dot(x1[1])
        + d1[2] * d1[2] * x1[2].dot(x1[2])
        - 2 * d1[1] * d1[2] * x1[1].dot(x1[2])
    )

    # fmt: off
    # Setup expanded equation system
    coeff_ind0 = [0, 6, 12, 1, 7, 13, 2, 8, 0, 6, 12, 14, 6, 0, 12, 1, 7, 13, 3, 9, 2, 8, 14, 15, 4, 10, 7,
                  1, 16, 13, 8, 2, 6, 12, 0, 14, 9, 3, 8, 14, 2, 15, 3, 9, 15, 4, 10, 16, 7, 13, 1, 5, 11, 10, 4, 17, 16]
    coeff_ind1 = [11, 17, 5, 9, 15, 3, 5, 11, 17, 10, 16, 4, 11, 5, 17]
    ind0 = [0, 1, 9, 12, 13, 21, 24, 25, 26, 28, 29, 33, 39, 42, 47, 50, 52, 53, 60, 61, 62, 64, 65, 69, 72, 73, 75, 78, 81, 83, 87,
            90, 91, 92, 94, 95, 99, 102, 103, 104, 106, 107, 110, 112, 113, 122, 124, 125, 127, 128, 130, 132, 133, 135, 138, 141, 143]
    ind1 = [7, 8, 10, 19, 20, 22, 26, 28, 29, 31, 32, 34, 39, 42, 47]

    # fmt: on
    C0 = np.zeros((12, 12))
    C1 = np.zeros((12, 4))
    C0[np.unravel_index(ind0, (12, 12), "F")] = coeffs[coeff_ind0]
    C1[np.unravel_index(ind1, (12, 4), "F")] = coeffs[coeff_ind1]

    # Linear elimination
    C2 = np.linalg.solve(C0, C1)

    # Setup action matrix
    AM = np.array([[0.0, 0.0, 1.0, 0.0], -C2[9, :], -C2[10, :], -C2[11, :]])

    # Solve eigenvalue problem and get real solutions
    D, V = np.linalg.eig(AM)
    sols = np.array([V[1, :] / V[0, :], D, V[3, :] / V[0, :]]).T
    sols = sols[np.isreal(D), :]

    # Extract solutions
    solutions = []

    for s in sols:
        s = np.real(s)
        solutions.append((1.0, s[1], np.sqrt(s[0]), s[2] * np.sqrt(s[0])))

    return solutions
