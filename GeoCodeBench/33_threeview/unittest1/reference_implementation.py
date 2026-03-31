"""
Reference Implementation for run_test_dE23dr12, run_test_dE23dt12, run_test_dE23dr13
This serves as the ground truth for testing LLM-generated implementations.
"""

import numpy as np
from scipy.spatial.transform import Rotation

def quat_multiply(qa, qb):
    qa1, qa2, qa3, qa4 = qa
    qb1, qb2, qb3, qb4 = qb

    return np.array([qa1 * qb1 - qa2 * qb2 - qa3 * qb3 - qa4 * qb4, qa1 * qb2 + qa2 * qb1 + qa3 * qb4 - qa4 * qb3,
                           qa1 * qb3 + qa3 * qb1 - qa2 * qb4 + qa4 * qb2,
                           qa1 * qb4 + qa2 * qb3 - qa3 * qb2 + qa4 * qb1])

def quat_exp(w):
    theta = np.linalg.norm(w)
    theta_half = theta / 2
    if theta > 1e-6:
        re = np.cos(theta_half)
        im = np.sin(theta_half) / theta
    else:
        theta2 = theta * theta
        theta4 = theta2 * theta2
        re = 1.0 - (1.0 / 8.0) * theta2 + (1.0 / 384.0) * theta4
        im = 0.5 - (1.0 / 48.0) * theta2 + (1.0 / 3840.0) * theta4

        s = np.sqrt(re * re + im * im * theta2)
        re /= s
        im /= s
    return np.array([re, im * w[0], im * w[1], im * w[2]])

def rotmat_to_quat(R):
    q = Rotation.from_matrix(R).as_quat(canonical=True)
    return np.array([q[3], q[0], q[1], q[2]])

def quat_to_rotmat(q):
    return Rotation.from_quat(np.array([q[1], q[2], q[3], q[0]])).as_matrix()

def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

def get_E(R, t):
    return skew(t) @ R

def get_random_Rt(normalized=False):
    R, _ = np.linalg.qr(np.random.randn(3, 3))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1

    t = np.random.randn(3)
    if normalized:
        t /= np.linalg.norm(t)

    return R, t

def apply(R, w_delta):
    q = rotmat_to_quat(R)
    q_new = quat_multiply(q, quat_exp(w_delta))
    return quat_to_rotmat(q_new)

def run_test_dE23dr12(i=0, eps=1e-8):
    e = np.eye(3)[i]

    R12, t12 = get_random_Rt()
    R13, t13 = get_random_Rt()

    R23 = R13 @ R12.T
    t23 = -R23 @ t12 + t13
    E23 = skew(t23) @ R23

    r = e * eps

    R12_new = apply(R12, r)
    R23_new = R13 @ R12_new.T
    t23_new = -R23_new @ t12 + t13
    E23_new = skew(t23_new) @ R23_new

    dE23dr12 = (E23_new - E23) / eps

    dE23dr12_direct = skew(R13 @ skew(e) @ R12.T @ t12) @ R23 \
                    - skew(t23) @ R13 @ skew(e) @ R12.T

    return np.linalg.norm(dE23dr12 - dE23dr12_direct)

def run_test_dE23dt12(i=0, eps=1e-8):
    e = np.eye(3)[i]

    R12, t12 = get_random_Rt()
    R13, t13 = get_random_Rt()

    R23 = R13 @ R12.T
    t23 = -R23 @ t12 + t13
    E23 = skew(t23) @ R23

    t12_new = t12 + e * eps

    t23_new = -R23 @ t12_new + t13
    E23_new = skew(t23_new) @ R23

    dE23dt12 = (E23_new - E23) / eps

    dE23dt12_direct = - skew(R23 @ e) @ R23

    return np.linalg.norm(dE23dt12 - dE23dt12_direct)

def run_test_dE23dr13(i=0, eps=1e-8):
    e = np.eye(3)[i]

    R12, t12 = get_random_Rt()
    R13, t13 = get_random_Rt()

    R23 = R13 @ R12.T
    t23 = -R23 @ t12 + t13
    E23 = skew(t23) @ R23

    r = e * eps

    R13_new = apply(R13, r)
    R23_new = R13_new @ R12.T
    t23_new = -R23_new @ t12 + t13
    E23_new = skew(t23_new) @ R23_new

    dE23dr13 = (E23_new - E23) / eps

    dE23dr12_direct = skew(R13 @ skew(e) @ R12.T @ t12) @ R23 \
                    - skew(t23) @ R13 @ skew(e) @ R12.T

    dE23dr13_direct = - dE23dr12_direct

    return np.linalg.norm(dE23dr13 - dE23dr13_direct)
