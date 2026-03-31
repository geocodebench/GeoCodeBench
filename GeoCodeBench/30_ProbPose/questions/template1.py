
"""
LLM Template for generate_probmaps() function
Fill in the ****FILL_HERE**** sections to complete the implementation.
"""

from itertools import product
from typing import Tuple

import numpy as np


def generate_probmaps(
    heatmap_size: Tuple[int, int], keypoints: np.ndarray, keypoints_visible: np.ndarray, sigma: float = 0.55
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Object Keypoint Similarity (OKS) maps for keypoints.

    This function generates OKS maps that represent the expected OKS score at each
    pixel location given the ground truth keypoint locations. The concept was
    introduced in `ProbPose`_ to enable probabilistic keypoint detection.

    Args:
        heatmap_size (Tuple[int, int]): Heatmap size in [W, H]
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)
        sigma (float): The sigma value to control the spread of the OKS map.
            If None, per-keypoint sigmas from COCO will be used. Default: 0.55

    Returns:
        tuple:
        - heatmaps (np.ndarray): The generated OKS maps in shape
            (K, H, W) where [W, H] is the `heatmap_size`
        - keypoint_weights (np.ndarray): The target weights in shape
            (N, K)

    .. _`ProbPose`: https://arxiv.org/abs/2412.02254
    """

    N, K, _ = keypoints.shape
    W, H = heatmap_size

    # The default sigmas are used for COCO dataset.
    sigmas = np.array([2.6, 2.5, 2.5, 3.5, 3.5, 7.9, 7.9, 7.2, 7.2, 6.2, 6.2, 10.7, 10.7, 8.7, 8.7, 8.9, 8.9]) / 100

    heatmaps = np.zeros((K, H, W), dtype=np.float32)
    keypoint_weights = keypoints_visible.copy()

    bbox_area = np.sqrt(H / 1.25 * W / 1.25)

    for n, k in product(range(N), range(K)):
        kpt_sigma = sigmas[k]
        # skip unlabled keypoints
        if keypoints_visible[n, k] < 0.5:
            continue

        ****FILL_HERE****

        vars = (kpt_sigma * 2) ** 2
        ****FILL_HERE****

    return heatmaps, keypoint_weights
