"""
Reference Implementation for BrightnessActivation
This serves as the ground truth for testing LLM-generated implementations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BrightnessActivation(nn.Module):
    """Piece-wise linear brightness mapping used by the light-control branch."""

    def __init__(self):
        super(BrightnessActivation, self).__init__()

    def forward(self, x):
        """Map raw [0,1] probability → perceptually-balanced brightness factor.

        Below a 0.75 threshold the mapping is identity (no amplification).  Above
        the threshold we linearly re-scale to the range [0.75,10] to give the
        optimisation headroom when compensating for very dark backgrounds.
        """
        output = torch.zeros_like(x)
        mask1 = (x <= 0.75)
        mask2 = (x > 0.75)

        # Linear part for x in [0, 0.75]
        output[mask1] = x[mask1]

        # Linear transformation for x in (0.75, 1]
        output[mask2] = 0.75 + (x[mask2] - 0.75) * ((10 - 0.75) / (1 - 0.75))

        return output

