
"""
LLM Template for SetCriterion.calc_similarity_map()
This file shows the template format for LLM code completion.
The template should only contain input/output examples, no hints.
"""

import torch
import torch.nn as nn


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, deep_supervision=True):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        self.deep_supervision = deep_supervision

    def calc_similarity_map(self, feats, point_coords):
        """
        Compute similarity map between feature map and point coordinates.

        Args:
            feats (Tensor): Feature map of shape N, res*res, C
            point_coords (Tensor): Point coordinates of shape (N, P, 2).

        Returns:
            Tensor: The similarity map of shape (N, P, H*W).
        """
        res = int(feats.shape[1] ** 0.5)

        ****EMPTY****
        
        return sim_map      # (N, P, res*res)
