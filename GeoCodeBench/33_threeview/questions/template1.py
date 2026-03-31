
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

def run_test_dE23dr12(i = 0, eps = 1e-8):
    ****EMPTY****

    return np.linalg.norm(dE23dr12 - dE23dr12_direct)

def run_test_dE23dt12(i = 0, eps = 1e-8):
    ****EMPTY****

    return np.linalg.norm(dE23dt12 - dE23dt12_direct)

def run_test_dE23dr13(i = 0, eps = 1e-8):
    ****EMPTY****

    return np.linalg.norm(dE23dr13 - dE23dr13_direct)
