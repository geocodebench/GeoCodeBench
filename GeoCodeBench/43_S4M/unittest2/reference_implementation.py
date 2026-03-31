"""
Reference Implementation for SetCriterion.calc_similarity_map()
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch
import torch.nn as nn


class SetCriterion(nn.Module):
    """This class computes the loss for DETR."""

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, deep_supervision=True):
        """Create the criterion."""
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

        point_indices = (point_coords[:, :, 0] * res + point_coords[:, :, 1]).long()  # (N, P)
        extracted_features = torch.gather(feats, dim=1, index=point_indices.unsqueeze(-1).expand(-1, -1, feats.shape[-1]))  # (N, P, C)
        
        extracted_features_norm = extracted_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (N, P, 1)
        extracted_features_normalized = extracted_features / extracted_features_norm  # (N, P, C)
        feats_norm = feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (N, res*res, 1)
        feats_normalized = feats / feats_norm  # (N, res*res, C)
        
        sim_map = torch.bmm(extracted_features_normalized, feats_normalized.permute(0, 2, 1))  # (N, P, res*res)
        
        return sim_map      # (N, P, res*res)
