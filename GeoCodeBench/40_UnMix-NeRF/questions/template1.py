
"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import unmixField
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Literal, Optional, Any, Dict, Tuple

from unmix_field import unmixField as BaseUnmixField
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.fields.base_field import get_normalized_directions


class unmixField(BaseUnmixField):
    """Template implementation of unmixField with get_outputs method to be filled."""
    
    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[Any, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # Get appearance embedding
        embedded_appearance = None
        if self.embedding_appearance is not None:
            if self.training:
                embedded_appearance = self.embedding_appearance(camera_indices)
            else:
                if self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    ) * self.embedding_appearance.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    )

        if "spectral" in self.method:

            positions =  ray_samples.frustums.get_positions()
            positions_flat = self.position_encoding(positions.view(-1, 3))


            if len(positions_flat.shape) == 2:
                positions_flat = positions_flat.unsqueeze(0)
            else:
                positions_flat = positions_flat.view(-1, density_embedding.size(1),  self.position_encoding.get_out_dim() )


            if len(density_embedding.shape) == 2:
                density_embedding = density_embedding.unsqueeze(0)


            ****EMPTY****

            if self.pred_specular:

                input_spec = torch.cat(
                [
                    d,
                    positions_flat.view(-1, self.position_encoding.get_out_dim()),
                ],
                dim=-1,
                )

                specular = self.mlp_directional(input_spec).view(*outputs_shape, self.wavelengths) # (B, ray_sample, wavelengths)
                spec2 = spec +  (s1 * specular)

            if self.pred_specular:
                outputs["spectral"] = spec2.to(directions)
                outputs["spectral2"] = spec.to(directions)
                with torch.no_grad():
                    outputs["specular"] = (s1 * specular).to(directions)
            else:
                outputs["spectral"] = spec.to(directions)

            outputs["abundances"] = abundances.to(directions)


        return outputs
