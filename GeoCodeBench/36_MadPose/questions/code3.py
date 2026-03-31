import numpy as np

import madpose


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
        f1_gt = 1000.0 + 2000.0 * np.random.rand(1)
        f2_gt = 1000.0 + 2000.0 * np.random.rand(1)

        d1_gt = 1.0 + 5 * np.random.rand(4)
        X = x1 * d1_gt[:, None]
        R = np.linalg.qr(np.random.randn(3, 3))[0]
        R = R * np.linalg.det(R)
        t = np.random.randn(3)
        X2 = X @ R.T + t
        d2_gt = X2[:, 2]
        x2 = X2 / d2_gt[:, None]

        # Add shared focal length
        x1[:, 0:2] *= f1_gt
        x2[:, 0:2] *= f2_gt
        x2[:, 0:2] += np.random.randn(4, 2) * 1 + 0.5
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

    sols = solve_shift_and_scale_two_focal(x1, x2, d1, d2)
    sols_madpose = madpose.solve_scale_and_shift_two_focal(x1.T, x2.T, d1, d2)
    # poses = madpose.solve_scale_shift_pose_two_focal(x1.T, x2.T, d1, d2)

    print(len(sols), len(sols_madpose))
    for k, (a1, b1, a2, b2, f1, f2) in enumerate(sols + sols_madpose):
        err = (
            np.abs(a2 - a2_gt / a1_gt)
            + np.abs(b1 - b1_gt / a1_gt)
            + np.abs(b2 - b2_gt / a1_gt)
            + np.abs(f1 - f1_gt)
            + np.abs(f2 - f2_gt)
        )
        focal_err = np.abs(f1 - f1_gt) + np.abs(f2 - f2_gt)
        print(f"focal_error={focal_err}")

        t_skew = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        E = t_skew @ R
        K2_ = np.array([[f2, 0, 0], [0, f2, 0], [0, 0, 1]])
        K1_ = np.array([[f1, 0, 0], [0, f1, 0], [0, 0, 1]])
        F = np.linalg.inv(K2_.T) @ E @ K1_

        d1_corr = a1 * d1 + b1
        d2_corr = a2 * d2 + b2

        x1u = x1.copy()
        x1u[:, 0:2] /= f1
        x2u = x2.copy()
        x2u[:, 0:2] /= f2

        X1 = x1u * d1_corr[:, None]
        X2 = x2u * d2_corr[:, None]
        R_est, t_est = find_transform(X1[0:3, :], X2[0:3, :])

        t_est_skew = np.array(
            [
                [0, -t_est[2], t_est[1]],
                [t_est[2], 0, -t_est[0]],
                [-t_est[1], t_est[0], 0],
            ]
        )
        E_est = t_est_skew @ R_est
        K2_est = np.array([[f2, 0, 0], [0, f2, 0], [0, 0, 1]])
        K1_est = np.array([[f1, 0, 0], [0, f1, 0], [0, 0, 1]])
        F_est = np.linalg.inv(K2_est.T) @ E_est @ K1_est

        # normalize F
        F = F / np.linalg.norm(F)
        F_est = F_est / np.linalg.norm(F_est)
        F_error = np.linalg.norm(F - F_est)
        print(f"F_error={F_error}")

        err_R = np.linalg.norm(R - R_est)
        err_t = np.linalg.norm(t / np.linalg.norm(t) - t_est / np.linalg.norm(t_est))
        print(f"solution={k}, residual={err}, rotation={err_R}, translation={err_t}")


if __name__ == "__main__":
    test_solver()