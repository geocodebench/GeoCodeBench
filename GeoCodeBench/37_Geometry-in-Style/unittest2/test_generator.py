"""
Test Data Generator for calc_ARAP_global_solve().
Generates various test cases with different configurations.
"""

from __future__ import annotations

import sys
import torch
from pathlib import Path

# Add parent directory to path for deformations_dARAP
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from deformations_dARAP import (
    Meshes,
    SparseLaplaciansSolvers,
)

device = torch.device('cpu')
torch.set_default_device(device)


class TestDataGenerator:
    """Generate test data for calc_ARAP_global_solve() function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_simple_mesh(self, n_verts=10, n_faces=12):
        """Generate a simple mesh for testing."""
        torch.manual_seed(self.seed + n_verts)
        verts = torch.rand(n_verts, 3, device=device) * 2.0 - 1.0

        faces = []
        for i in range(min(n_verts - 2, n_faces)):
            v0 = 0
            v1 = (i + 1) % n_verts
            v2 = (i + 2) % n_verts
            if v0 != v1 and v1 != v2 and v0 != v2:
                faces.append([v0, v1, v2])

        if len(faces) < n_faces:
            for i in range(len(faces), n_faces):
                v0 = i % n_verts
                v1 = (i + 1) % n_verts
                v2 = (i + 2) % n_verts
                if v0 != v1 and v1 != v2 and v0 != v2:
                    faces.append([v0, v1, v2])

        faces = torch.tensor(faces[:n_faces], dtype=torch.long, device=device)
        return verts, faces

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with small mesh, spokes_and_rims_mine
        verts, faces = self.generate_simple_mesh(n_verts=8, n_faces=10)
        meshes = Meshes(verts=[verts], faces=[faces])
        laplacians_solvers = SparseLaplaciansSolvers.from_meshes(
            meshes,
            pin_first_vertex=False,
            compute_poisson_rhs_lefts=False,
            compute_igl_arap_rhs_lefts=None,
        )
        n_verts = verts.shape[0]
        per_vertex_rot_matrices = torch.eye(3, device=device).unsqueeze(0).repeat(n_verts, 1, 1)
        test_cases.append({
            'meshes': meshes,
            'laplacians_solvers': laplacians_solvers,
            'per_vertex_rot_matrices_packed': per_vertex_rot_matrices,
            'arap_energy_type': 'spokes_and_rims_mine',
            'postprocess': None,
            'description': f'Basic: small mesh (n_verts={n_verts}), spokes_and_rims_mine, no postprocess',
        })

        if num_tests > 1:
            verts, faces = self.generate_simple_mesh(n_verts=6, n_faces=8)
            meshes = Meshes(verts=[verts], faces=[faces])
            laplacians_solvers = SparseLaplaciansSolvers.from_meshes(
                meshes,
                pin_first_vertex=False,
                compute_poisson_rhs_lefts=False,
                compute_igl_arap_rhs_lefts=None,
            )
            n_verts = verts.shape[0]
            per_vertex_rot_matrices = torch.eye(3, device=device).unsqueeze(0).repeat(n_verts, 1, 1)
            test_cases.append({
                'meshes': meshes,
                'laplacians_solvers': laplacians_solvers,
                'per_vertex_rot_matrices_packed': per_vertex_rot_matrices,
                'arap_energy_type': 'spokes_mine',
                'postprocess': None,
                'description': f'Test 2: small mesh (n_verts={n_verts}), spokes_mine, no postprocess',
            })

        if num_tests > 2:
            verts, faces = self.generate_simple_mesh(n_verts=10, n_faces=14)
            meshes = Meshes(verts=[verts], faces=[faces])
            laplacians_solvers = SparseLaplaciansSolvers.from_meshes(
                meshes,
                pin_first_vertex=False,
                compute_poisson_rhs_lefts=False,
                compute_igl_arap_rhs_lefts=None,
            )
            n_verts = verts.shape[0]
            per_vertex_rot_matrices = torch.eye(3, device=device).unsqueeze(0).repeat(n_verts, 1, 1)
            test_cases.append({
                'meshes': meshes,
                'laplacians_solvers': laplacians_solvers,
                'per_vertex_rot_matrices_packed': per_vertex_rot_matrices,
                'arap_energy_type': 'spokes_and_rims_mine',
                'postprocess': 'recenter_only',
                'description': f'Test 3: medium mesh (n_verts={n_verts}), spokes_and_rims_mine, recenter_only',
            })

        if num_tests > 3:
            verts, faces = self.generate_simple_mesh(n_verts=12, n_faces=18)
            meshes = Meshes(verts=[verts], faces=[faces])
            laplacians_solvers = SparseLaplaciansSolvers.from_meshes(
                meshes,
                pin_first_vertex=False,
                compute_poisson_rhs_lefts=False,
                compute_igl_arap_rhs_lefts=None,
            )
            n_verts = verts.shape[0]
            per_vertex_rot_matrices = torch.eye(3, device=device).unsqueeze(0).repeat(n_verts, 1, 1)
            test_cases.append({
                'meshes': meshes,
                'laplacians_solvers': laplacians_solvers,
                'per_vertex_rot_matrices_packed': per_vertex_rot_matrices,
                'arap_energy_type': 'spokes_and_rims_mine',
                'postprocess': 'recenter_rescale',
                'description': f'Test 4: larger mesh (n_verts={n_verts}), spokes_and_rims_mine, recenter_rescale',
            })

        if num_tests > 4:
            verts, faces = self.generate_simple_mesh(n_verts=8, n_faces=10)
            meshes = Meshes(verts=[verts], faces=[faces])
            laplacians_solvers = SparseLaplaciansSolvers.from_meshes(
                meshes,
                pin_first_vertex=False,
                compute_poisson_rhs_lefts=False,
                compute_igl_arap_rhs_lefts=None,
            )
            n_verts = verts.shape[0]
            per_vertex_rot_matrices = torch.randn(n_verts, 3, 3, device=device)
            per_vertex_rot_matrices = torch.nn.functional.normalize(
                per_vertex_rot_matrices.view(-1, 3), p=2, dim=1
            ).view(n_verts, 3, 3)
            test_cases.append({
                'meshes': meshes,
                'laplacians_solvers': laplacians_solvers,
                'per_vertex_rot_matrices_packed': per_vertex_rot_matrices,
                'arap_energy_type': 'spokes_and_rims_mine',
                'postprocess': None,
                'description': f'Test 5: small mesh (n_verts={n_verts}), spokes_and_rims_mine, random rotations',
            })

        for i in range(num_tests - len(test_cases)):
            n_verts = 6 + (i % 8)
            n_faces = n_verts + 2
            verts, faces = self.generate_simple_mesh(n_verts=n_verts, n_faces=n_faces)
            meshes = Meshes(verts=[verts], faces=[faces])
            laplacians_solvers = SparseLaplaciansSolvers.from_meshes(
                meshes,
                pin_first_vertex=False,
                compute_poisson_rhs_lefts=False,
                compute_igl_arap_rhs_lefts=None,
            )
            n_verts_actual = verts.shape[0]
            per_vertex_rot_matrices = torch.eye(3, device=device).unsqueeze(0).repeat(n_verts_actual, 1, 1)
            arap_energy_type = 'spokes_and_rims_mine' if (i % 2 == 0) else 'spokes_mine'
            postprocess = None if (i % 3 == 0) else ('recenter_only' if (i % 3 == 1) else 'recenter_rescale')
            test_cases.append({
                'meshes': meshes,
                'laplacians_solvers': laplacians_solvers,
                'per_vertex_rot_matrices_packed': per_vertex_rot_matrices,
                'arap_energy_type': arap_energy_type,
                'postprocess': postprocess,
                'description': f'Additional test {i+1}: mesh (n_verts={n_verts_actual}), {arap_energy_type}, postprocess={postprocess}',
            })

        return test_cases[:num_tests]
