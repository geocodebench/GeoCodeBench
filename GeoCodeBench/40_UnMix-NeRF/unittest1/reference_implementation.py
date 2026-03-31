"""
Reference Implementation for unmixField.get_outputs()
This serves as the ground truth for testing LLM-generated implementations.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import unmixField
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Literal, Optional, Any, Dict, Tuple

# Import from original implementation
from unmix_field import unmixField
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.fields.base_field import get_normalized_directions


class MockFrustums:
    """Mock Frustums class for testing."""
    def __init__(self, directions, positions):
        self.directions = directions
        self._positions = positions
        self.shape = directions.shape[:-1]
    
    def get_positions(self):
        return self._positions


class MockRaySamples:
    """Mock RaySamples class for testing."""
    def __init__(self, directions, positions, camera_indices):
        self.frustums = MockFrustums(directions, positions)
        self.camera_indices = camera_indices


# Import the reference implementation
# We'll use the original unmixField class as reference
# But we need to ensure device is CPU, not CUDA
class ReferenceUnmixField(unmixField):
    """Reference implementation with CPU device."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure endmembers are on CPU if not training
        if not self.training and hasattr(self, 'endmembers'):
            if isinstance(self.endmembers, nn.Parameter):
                self.endmembers = nn.Parameter(self.endmembers.data.cpu())
            else:
                self.endmembers = self.endmembers.cpu()
    
    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[Any, Tensor]:
        """Reference implementation of get_outputs - same as original."""
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


            h1 = torch.cat(
                [
                    #d,
                    positions_flat.view(-1, self.position_encoding.get_out_dim()),
                    density_embedding.view(-1, self.geo_feat_dim),
                ]
                + (
                    [embedded_appearance.view(-1, self.appearance_embedding_dim)] 
                    if embedded_appearance is not None else []
                ),
                dim=-1,
            ) # direction, density features, appeareance embeddings

            if self.use_scalar:
                scalar = self.mlp_head(h1).view(*outputs_shape, -1, self.num_classes)
                scalar = F.sigmoid(scalar)

            features_input = torch.cat([positions_flat, density_embedding], dim=-1) # positions, density

            size = features_input.size()
            features_input = features_input.view(-1, features_input.size(-1))
            
            features = self.feature_mlp(features_input)
            logits = features.view(*size[:-1], -1)

            if self.pred_specular:
                logits, s1 = torch.split(logits, [self.num_classes, 1], dim=-1)
                s1 = F.sigmoid(s1)

            abundances = F.softmax(logits / self.temperature, dim=-1)

            endmembers = self.endmembers.unsqueeze(0).unsqueeze(0)
    
            endmembers = endmembers.expand(abundances.shape[0], abundances.shape[1], -1, -1).transpose(2,3).squeeze(0)

            if self.use_scalar:
                adapted_endmembers = scalar * endmembers  # (B, ray_sample, wavelengths, num_classes)
            else:
                adapted_endmembers = endmembers

            spec = (adapted_endmembers  @ abundances.unsqueeze(-1)).squeeze() # linear mixing model spec = EA

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
