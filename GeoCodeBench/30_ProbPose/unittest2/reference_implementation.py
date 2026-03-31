"""
Reference Implementation for compute_oks() function
This serves as the ground truth for testing LLM-generated implementations.
"""

import numpy as np


def compute_oks(gt, dt, use_area=True, per_kpt=False):
    """
    Compute Object Keypoint Similarity (OKS) between ground truth and detection.
    
    Args:
        gt: Dictionary containing:
            - keypoints: List or array of shape (51,) representing 17 keypoints with (x, y, visibility)
            - bbox: List or array of shape (4,) representing [center_x, center_y, width, height]
            - area: Float, area of the bounding box (used if use_area=True)
        dt: Dictionary containing:
            - keypoints: List or array of shape (51,) representing 17 keypoints with (x, y, visibility)
        use_area: Boolean, whether to use gt["area"] for normalization (default: True)
        per_kpt: Boolean, whether to return per-keypoint OKS (default: False)
        
    Returns:
        If per_kpt=True: numpy array of shape (17,) with OKS for each keypoint
        If per_kpt=False: scalar float, average OKS over visible keypoints
    """
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
        # measure the per-keypoint distance if keypoints visible
        dx = xd - xg
        dy = yd - yg

    else:
        # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
        z = np.zeros((k))
        dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
        dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)

    if use_area:
        e = (dx**2 + dy**2) / vars / (gt["area"] + np.spacing(1)) / 2
    else:
        tmparea = gt["bbox"][3] * gt["bbox"][2] * 0.53
        e = (dx**2 + dy**2) / vars / (tmparea + np.spacing(1)) / 2

    if per_kpt:
        oks = np.exp(-e)
        if k1 > 0:
            oks[~visibility_condition(vg)] = 0

    else:
        if k1 > 0:
            e = e[visibility_condition(vg)]
        oks = np.sum(np.exp(-e)) / e.shape[0]

    return oks
