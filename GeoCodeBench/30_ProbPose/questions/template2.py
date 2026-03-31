
"""
LLM Template for compute_oks() function
Fill in the implementation below.
"""

import numpy as np


def compute_oks(gt, dt, use_area=True, per_kpt=False):
    sigmas = (
        np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])
        / 10.0
    )
    vars = (sigmas * 2) ** 2
    k = len(sigmas)
    visibility_condition = lambda x: x > 0
    g = np.array(gt["keypoints"]).reshape(k, 3)
    xg = g[:, 0]
    yg = g[:, 1]
    vg = g[:, 2]
    k1 = np.count_nonzero(visibility_condition(vg))
    bb = gt["bbox"]
    x0 = bb[0] - bb[2]
    x1 = bb[0] + bb[2] * 2
    y0 = bb[1] - bb[3]
    y1 = bb[1] + bb[3] * 2

    d = np.array(dt["keypoints"]).reshape((k, 3))
    xd = d[:, 0]
    yd = d[:, 1]

    if k1 > 0:
        ****EMPTY****

    else:
        ****EMPTY****

    if use_area:
        ****EMPTY****
    else:
        ****EMPTY****

    if per_kpt:
        ****EMPTY****
    else:
        ****EMPTY****

    return oks
