import numpy as np

import madpose


def solve_shift_and_scale_shared_focal(x1_, x2_, d1, d2):
    # Normalize focal length
    x1 = x1_.copy()
    x2 = x2_.copy()

    f1_0 = np.abs(x1[:, 0:2].flatten()).mean()
    f2_0 = np.abs(x2[:, 0:2].flatten()).mean()
    f0 = 0.5 * (f1_0 + f2_0)
    x1[:, 0:2] /= f0
    x2[:, 0:2] /= f0

    # Compute coefficients
    ****EMPTY****

    # Extract solutions
    solutions = []

    for s in sols:
        s = np.real(s)
        if s[3] < 0:
            continue
        solutions.append((1.0, s[1], np.sqrt(s[0]), s[2] * np.sqrt(s[0]), f0 / np.sqrt(s[3])))

    return solutions


def find_transform(X1, X2):
    m1 = np.mean(X1, axis=0)
    m2 = np.mean(X2, axis=0)
    X1m = X1 - m1
    X2m = X2 - m2
    u, s, vt = np.linalg.svd(X2m.T @ X1m)
    R = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    t = m2 - R @ m1
    return R, t


def test_solver():
    # Setup instance (with positive depths)
    while True:
        x1 = np.c_[np.random.randn(4, 2), np.ones((4,))]
        f_gt = 1000.0 + 2000.0 * np.random.rand(1)
        d1_gt = 1.0 + 5 * np.random.rand(4)
        X = x1 * d1_gt[:, None]
        R = np.linalg.qr(np.random.randn(3, 3))[0]
        R = R * np.linalg.det(R)
        t = np.random.randn(3)
        X2 = X @ R.T + t
        d2_gt = X2[:, 2]
        x2 = X2 / d2_gt[:, None]

        # Add shared focal length
        x1[:, 0:2] *= f_gt
        x2[:, 0:2] *= f_gt
        x2[:, 0:2] += 0.5 * np.random.randn(4, 2) - 0.25

        if np.all(d2_gt > 0):
            break

    # Shift and scale gt depths
    a1_gt = np.random.rand(1)
    b1_gt = np.random.randn(1)
    a2_gt = np.random.rand(1)
    b2_gt = np.random.randn(1)

    # d1_gt = a1 * d1 + b1
    d1 = (d1_gt - b1_gt) / a1_gt
    d2 = (d2_gt - b2_gt) / a2_gt

    sols = solve_shift_and_scale_shared_focal(x1, x2, d1, d2)
    sols_madpose = madpose.solve_scale_and_shift_shared_focal(x1.T, x2.T, d1, d2)
    posescaleoffsetsfs = madpose.solve_scale_shift_pose_shared_focal(x1.T, x2.T, d1, d2)

    for p in posescaleoffsetsfs:
        R_est, t_est = p.R(), p.t()
        a, b1, b2, f = p.scale, p.offset0, p.offset1, p.focal
        err_a = np.abs(a - a2_gt / a1_gt)

        d1_corr = d1 + b1
        d2_corr = a * d2 + b2

        K_ = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]])
        K_inv = np.linalg.inv(K_)
        x1u = x1.copy()
        x1u = x1u @ K_inv.T
        x2u = x2.copy()
        x2u = x2u @ K_inv.T

        X1 = x1u * d1_corr[:, None]
        X2 = x2u * d2_corr[:, None]

        err_R = np.linalg.norm(R - R_est)
        err_t = np.linalg.norm(t / np.linalg.norm(t) - t_est / np.linalg.norm(t_est))
        print(f"posescaleoffsetsfs, residual={err_a}, rotation={err_R}, translation={err_t}")

    print(len(sols), len(sols_madpose))
    for k, (a1, b1, a2, b2, f) in enumerate(sols + sols_madpose):
        err = (
            np.abs(a2 - a2_gt / a1_gt)
            + np.abs(b1 - b1_gt / a1_gt)
            + np.abs(b2 - b2_gt / a1_gt)
            + np.abs(f - f_gt)
        )
        focal_err = np.abs(f - f_gt)
        print(f"focal_err={focal_err}")

        d1_corr = a1 * d1 + b1
        d2_corr = a2 * d2 + b2

        x1u = x1.copy()
        x1u[:, 0:2] /= f
        x2u = x2.copy()
        x2u[:, 0:2] /= f

        X1 = x1u * d1_corr[:, None]
        X2 = x2u * d2_corr[:, None]
        R_est, t_est = find_transform(X1[0:4, :], X2[0:4, :])

        err_R = np.linalg.norm(R - R_est)
        err_t = np.linalg.norm(t / np.linalg.norm(t) - t_est / np.linalg.norm(t_est))
        print(f"solution={k}, residual={err}, rotation={err_R}, translation={err_t}")


if __name__ == "__main__":
    test_solver()