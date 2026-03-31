
"""
LLM Template for GaussianModel.project_gas_constraints()
Fill in the ****EMPTY**** sections.
"""

import numpy as np
import time
import torch
from torch_cluster import radius_graph


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            # Dummy implementation for testing
            return torch.zeros(1)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = lambda x: torch.log(x / (1 - x))
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

    def setup_constants(self, H=0.00625, p0=1.0, k=1.0, KNN_K=32, record_time=False):
        """Setup constants for testing."""
        device = "cpu"  # Use CPU for testing
        
        self.H = H
        self.H2 = self.H**2
        self.H6 = self.H**6
        self.H9 = self.H**9

        self.EPSILON = 1e-8

        self.RELAXATION = 0.01
        self.K_P = 0.2
        self.E_P = 4
        self.DQ_P = 0.25

        self.p0 = p0
        self.k = k

        self.KNN_K = KNN_K

        self.record_time = record_time

        self.poly6_term1 = 315.0 / (64.0 * np.pi * self.H9)
        self.spiky_grad_term1 = 45.0 / (np.pi * self.H6)
        self.lamb_corr_denom = self.poly6(self.DQ_P * self.DQ_P * self.H * self.H)

    def poly6(self, r2):
        term2 = self.H2 - r2
        mask = r2 < self.H2
        return mask * self.poly6_term1 * (term2**3)

    def spiky_grad(self, r, rlen):
        mask = (rlen < self.H) & (rlen > 0)
        r_norm = r / (rlen.unsqueeze(-1) + self.EPSILON)
        term2 = (self.H - rlen).unsqueeze(-1) ** 2
        grad = -r_norm * self.spiky_grad_term1 * term2
        grad[~mask] = 0.0
        return grad

    @torch.no_grad()
    def project_gas_constraints(self):
        if self.record_time:
            # For CPU, we use time.time() instead of CUDA events
            start_time = time.time()

        N = self._estimate_xyz.shape[0]
        exyz = self._estimate_xyz

        ****EMPTY****

        # Compute pressure ratios and force corrections
        p_ratio = pi / self.p0  # Shape: (N, 1)
        force_delta = self._velocity * (1.0 - p_ratio) * -self.k  # Shape: (N, 3)
        self._force += force_delta

        # Compute lambdas
        lambdas = -(p_ratio - 1.0) / (denom + self.RELAXATION)  # Shape: (N, 1)

        # Compute lambda corrections
        poly6_values_ns = poly6_values[non_self_loop_mask]  # Shape: (E_ns,)
        lamb_corr = -self.K_P * (poly6_values_ns / self.lamb_corr_denom) ** self.E_P  # Shape: (E_ns,)

        ****EMPTY****

        # Compute estimate_xyz_delta_candidate
        estimate_xyz_delta_candidate = deltas_sum / (neighbors_len + self._counts)  # Shape: (N, 3)

        # Apply corrections
        self._estimate_xyz += estimate_xyz_delta_candidate

        if self.record_time:
            elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        else:
            elapsed_time = 0.0

        # Prepare return values for debugging or logging
        return_values = {
            "velocity": self._velocity.detach().clone().mean().item(),
            "xyz": self._xyz.detach().clone().mean().item(),
            "estimate_xyz": self._estimate_xyz.detach().clone().mean().item(),
            "diff": diff.detach().clone().mean().item(),
            "dist2": dist2.detach().clone().mean().item(),
            "poly6_values": poly6_values.detach().clone().mean().item(),
            "pi": pi.detach().clone().mean().item(),
            "rlen": rlen_ns.detach().clone().mean().item(),
            "spiky_grads": spiky_grads.detach().clone().mean().item(),
            "gr": gr.detach().clone().mean().item(),
            "gr_dot": gr_dot.detach().clone().mean().item(),
            "grad_dot": grad_dot.detach().clone().mean().item(),
            "denom": denom.detach().clone().mean().item(),
            "p_ratio": p_ratio.detach().clone().mean().item(),
            "force_delta": force_delta.detach().clone().mean().item(),
            "lambdas": lambdas.detach().clone().mean().item(),
            "lamb_corr": lamb_corr.detach().clone().mean().item(),
            "deltas": deltas.detach().clone().mean().item(),
            "estimate_xyz_delta": estimate_xyz_delta_candidate.detach().clone().mean().item(),
            "elapsed_time": elapsed_time,
        }

        return return_values
