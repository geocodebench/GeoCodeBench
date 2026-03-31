"""
Reference Implementation for GaussianModel.re_simulation_get_visual_xyz_delta()
This serves as the ground truth for testing LLM-generated implementations.
"""

import numpy as np
import torch
from torch_cluster import radius


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = inv_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, *args, **kwargs):
        self.active_sh_degree = 0
        # the fluid particles, used for PBD constraints
        self._xyz = torch.empty(0)
        self._estimate_xyz = torch.empty(0)
        self._force = torch.empty(0)
        self._velocity = torch.empty(0)
        self._imass = torch.empty(0)
        self._particle_id = torch.empty(0)
        self._particle_id_max = 0
        self.hidden_particles_created = False

        # the visual particles, used for rendering
        self._visual_xyz = torch.empty(0)

        # currently, these GS attributes are constant
        self._visual_color = torch.empty(0)
        self._visual_scales = torch.empty(0)
        self._visual_rotation = torch.empty(0)
        self._visual_opacity = torch.empty(0)
        # self._visual_omega = torch.empty(0)
        self.visual_particles_created = False

        # Maybe we need them for densification
        # currently, densification related tensors are not used
        self.visual_max_radii2D = torch.empty(0)
        self.visual_xyz_gradient_accum = torch.empty(0)
        self.visual_denom = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.setup_functions()

    def setup_constants(self, H=0.00625, KNN_K=32, secs=0.01):
        """Setup constants needed for the function."""
        self._secs = secs
        self.H = H
        self.H2 = self.H**2
        self.H6 = self.H**6
        self.H9 = self.H**9
        self.EPSILON = 1e-8
        self.KNN_K = KNN_K
        self.poly6_term1 = 315.0 / (64.0 * np.pi * self.H9)

    def poly6(self, r2):
        term2 = self.H2 - r2
        mask = r2 < self.H2
        return mask * self.poly6_term1 * (term2**3)

    @torch.no_grad()
    def re_simulation_get_visual_xyz_delta(self, xyz, visual_xyz, velocity):
        V = visual_xyz.shape[0]

        # Use radius graph to find neighbors within radius H
        edge_index = radius(
            x=xyz,
            y=visual_xyz,
            r=self.H,
            max_num_neighbors=self.KNN_K,
        )
        row = edge_index[0]  # Indices in visual particles
        col = edge_index[1]  # Indices in estimated particles

        # Compute squared distances
        diff = visual_xyz[row] - xyz[col]
        dist2 = torch.sum(diff**2, dim=1)

        # Compute poly6 values
        p6 = self.poly6(dist2)

        # Retrieve velocity of the estimated particles at the indices
        velocity_knn = velocity[col]

        # Compute weighted sum of velocity
        weighted_velocity = velocity_knn * p6.unsqueeze(-1)
        visual_velocity = torch.zeros(V, 3, device=visual_xyz.device)
        visual_velocity.index_add_(0, row, weighted_velocity)

        # Sum of p6 values
        sum_p6 = torch.zeros(V, device=visual_xyz.device)
        sum_p6.index_add_(0, row, p6)
        sum_p6 = sum_p6.clamp_min(self.EPSILON)

        # Compute delta for updating visual particle positions
        estimate_visual_xyz_delta = visual_velocity * self._secs / sum_p6.unsqueeze(-1)

        # Update visual positions
        new_visual_xyz = visual_xyz + estimate_visual_xyz_delta
        return new_visual_xyz


# Dummy functions for setup_functions (not used in this test)
def build_scaling_rotation(scaling, rotation):
    return torch.eye(3).unsqueeze(0).repeat(scaling.shape[0], 1, 1)


def strip_symmetric(sym):
    return sym


def inv_sigmoid(x):
    return torch.log(x / (1 - x))
