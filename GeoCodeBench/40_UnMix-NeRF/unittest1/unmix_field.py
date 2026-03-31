"""
Minimal unmixField base class used by unittest scaffolding.

This file is intentionally lightweight so unittest code can import:
    from unmix_field import unmixField
"""

from typing import Literal, Optional

import torch
from torch import Tensor, nn

from nerfstudio.field_components.mlp import MLP
from nerfstudio.fields.nerfacto_field import NerfactoField


class unmixField(NerfactoField):
    """Base field used by unittest implementations."""

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        wavelengths: int = 128,
        method: Literal["rgb", "spectral", "rgb+spectral"] = "rgb",
        num_classes: int = 4,
        feature_dim: int = 256,
        temperature: float = 0.5,
        converter=None,
        pred_dino: bool = False,
        pred_specular: bool = False,
        load_vca: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(aabb=aabb, num_images=num_images, implementation=implementation, **kwargs)

        self.method = method
        self.num_classes = num_classes
        self.wavelengths = wavelengths
        self.feature_dim = feature_dim
        self.pred_specular = pred_specular
        self.average_init_density = 1
        self.num_images = num_images
        self.implementation = implementation

        if self.method in ("spectral", "rgb+spectral"):
            input_dim = self.position_encoding.get_out_dim() + self.geo_feat_dim
            out_dim_feature = self.num_classes + 1 if self.pred_specular else self.num_classes
            self.feature_mlp = MLP(
                in_dim=input_dim,
                num_layers=3,
                layer_width=hidden_dim_color,
                out_dim=out_dim_feature,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )

            if self.training:
                if load_vca:
                    import numpy as np

                    endmembers = np.load("vca.npy")
                    self.endmembers = nn.Parameter(
                        torch.tensor(endmembers, dtype=torch.float32, requires_grad=True)
                    )
                else:
                    self.endmembers = nn.Parameter(
                        torch.randn(self.num_classes, self.wavelengths), requires_grad=True
                    )
            else:
                self.endmembers = torch.randn(self.num_classes, self.wavelengths, device=aabb.device)

            self.mlp_head = MLP(
                in_dim=self.position_encoding.get_out_dim()
                + self.geo_feat_dim
                + self.appearance_embedding_dim,
                num_layers=num_layers_color,
                layer_width=hidden_dim_color,
                out_dim=num_classes,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )

            self.mlp_directional = MLP(
                in_dim=self.direction_encoding.get_out_dim() + self.position_encoding.get_out_dim(),
                num_layers=2,
                layer_width=16,
                out_dim=self.wavelengths,
                activation=nn.ReLU(),
                out_activation=nn.Sigmoid(),
                implementation=implementation,
            )

            self.converter = converter
            self.temperature = temperature
            self.pred_dino = pred_dino
            self.use_scalar = True

    def get_outputs(self, ray_samples, density_embedding: Optional[Tensor] = None):
        raise NotImplementedError("Use subclass implementations for get_outputs().")
