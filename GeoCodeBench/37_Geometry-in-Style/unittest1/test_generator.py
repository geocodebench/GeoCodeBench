"""
Test Data Generator for calc_rot_matrices_with_procrustes().
Generates various test cases with different configurations.
"""

from __future__ import annotations

import torch
from reference_implementation import (
    ProcrustesPrecompute,
    MeshesPackedIndexer,
)

device = torch.device("cpu")


class TestDataGenerator:
    """Generate test data for calc_rot_matrices_with_procrustes() function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def create_procrustes_precompute(
        self,
        n_verts_packed: int,
        max_cell_neighborhood_n_edges: int,
        device: torch.device = device,
    ) -> ProcrustesPrecompute:
        """Create a ProcrustesPrecompute instance for testing."""
        padded_cell_edges = torch.zeros(
            (n_verts_packed, max_cell_neighborhood_n_edges, 2),
            dtype=torch.long,
            device=device,
        )
        for v_idx in range(n_verts_packed):
            n_edges = max_cell_neighborhood_n_edges
            for e_idx in range(n_edges):
                v0 = (v_idx + e_idx) % n_verts_packed
                v1 = (v_idx + e_idx + 1) % n_verts_packed
                padded_cell_edges[v_idx, e_idx, 0] = v0
                padded_cell_edges[v_idx, e_idx, 1] = v1

        covar_lefts_packed = torch.randn(
            (n_verts_packed, 3, max_cell_neighborhood_n_edges + 1),
            device=device,
            dtype=torch.float32,
        )
        num_per_mesh = torch.tensor([n_verts_packed], dtype=torch.long, device=device)
        verts_packed_idxr = MeshesPackedIndexer.from_num_per_mesh(num_per_mesh)
        mesh_to_verts_packed_first_idx = torch.tensor([0], dtype=torch.long, device=device)

        return ProcrustesPrecompute(
            padded_cell_edges_per_vertex_packed=padded_cell_edges,
            covar_lefts_packed=covar_lefts_packed,
            _verts_packed_idxr=verts_packed_idxr,
            _num_verts_per_mesh=num_per_mesh,
            _mesh_to_verts_packed_first_idx=mesh_to_verts_packed_first_idx,
        )

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        n_verts = 10
        max_edges = 6
        procrustes_precompute = self.create_procrustes_precompute(n_verts, max_edges, device)
        curr_deformed_verts = torch.randn(n_verts, 3, device=device, dtype=torch.float32)
        target_normals = torch.randn(n_verts, 3, device=device, dtype=torch.float32)
        target_normals = target_normals / (target_normals.norm(dim=-1, keepdim=True) + 1e-8)
        test_cases.append({
            'procrustes_precompute': procrustes_precompute,
            'curr_deformed_verts_packed': curr_deformed_verts,
            'target_verts_normals_packed': target_normals,
            'description': f'Basic: n_verts={n_verts}, max_edges={max_edges}',
        })

        if num_tests > 1:
            n_verts = 1
            max_edges = 3
            procrustes_precompute = self.create_procrustes_precompute(n_verts, max_edges, device)
            curr_deformed_verts = torch.randn(n_verts, 3, device=device, dtype=torch.float32)
            target_normals = torch.randn(n_verts, 3, device=device, dtype=torch.float32)
            target_normals = target_normals / (target_normals.norm(dim=-1, keepdim=True) + 1e-8)
            test_cases.append({
                'procrustes_precompute': procrustes_precompute,
                'curr_deformed_verts_packed': curr_deformed_verts,
                'target_verts_normals_packed': target_normals,
                'description': f'Single vertex: n_verts={n_verts}, max_edges={max_edges}',
            })

        if num_tests > 2:
            n_verts = 20
            max_edges = 8
            procrustes_precompute = self.create_procrustes_precompute(n_verts, max_edges, device)
            curr_deformed_verts = torch.randn(n_verts, 3, device=device, dtype=torch.float32)
            target_normals = torch.randn(n_verts, 3, device=device, dtype=torch.float32)
            target_normals = target_normals / (target_normals.norm(dim=-1, keepdim=True) + 1e-8)
            test_cases.append({
                'procrustes_precompute': procrustes_precompute,
                'curr_deformed_verts_packed': curr_deformed_verts,
                'target_verts_normals_packed': target_normals,
                'description': f'Larger: n_verts={n_verts}, max_edges={max_edges}',
            })

        if num_tests > 3:
            n_verts = 15
            max_edges = 10
            procrustes_precompute = self.create_procrustes_precompute(n_verts, max_edges, device)
            curr_deformed_verts = torch.randn(n_verts, 3, device=device, dtype=torch.float32)
            target_normals = torch.randn(n_verts, 3, device=device, dtype=torch.float32)
            target_normals = target_normals / (target_normals.norm(dim=-1, keepdim=True) + 1e-8)
            test_cases.append({
                'procrustes_precompute': procrustes_precompute,
                'curr_deformed_verts_packed': curr_deformed_verts,
                'target_verts_normals_packed': target_normals,
                'description': f'Medium: n_verts={n_verts}, max_edges={max_edges}',
            })

        if num_tests > 4:
            n_verts = 8
            max_edges = 2
            procrustes_precompute = self.create_procrustes_precompute(n_verts, max_edges, device)
            curr_deformed_verts = torch.randn(n_verts, 3, device=device, dtype=torch.float32)
            target_normals = torch.randn(n_verts, 3, device=device, dtype=torch.float32)
            target_normals = target_normals / (target_normals.norm(dim=-1, keepdim=True) + 1e-8)
            test_cases.append({
                'procrustes_precompute': procrustes_precompute,
                'curr_deformed_verts_packed': curr_deformed_verts,
                'target_verts_normals_packed': target_normals,
                'description': f'Small edges: n_verts={n_verts}, max_edges={max_edges}',
            })

        if num_tests > 5:
            n_verts = 50
            max_edges = 12
            procrustes_precompute = self.create_procrustes_precompute(n_verts, max_edges, device)
            curr_deformed_verts = torch.randn(n_verts, 3, device=device, dtype=torch.float32)
            target_normals = torch.randn(n_verts, 3, device=device, dtype=torch.float32)
            target_normals = target_normals / (target_normals.norm(dim=-1, keepdim=True) + 1e-8)
            test_cases.append({
                'procrustes_precompute': procrustes_precompute,
                'curr_deformed_verts_packed': curr_deformed_verts,
                'target_verts_normals_packed': target_normals,
                'description': f'Large: n_verts={n_verts}, max_edges={max_edges}',
            })

        if num_tests > 6:
            n_verts = 5
            max_edges = 4
            procrustes_precompute = self.create_procrustes_precompute(n_verts, max_edges, device)
            curr_deformed_verts = torch.zeros(n_verts, 3, device=device, dtype=torch.float32)
            target_normals = torch.ones(n_verts, 3, device=device, dtype=torch.float32)
            target_normals = target_normals / (target_normals.norm(dim=-1, keepdim=True) + 1e-8)
            test_cases.append({
                'procrustes_precompute': procrustes_precompute,
                'curr_deformed_verts_packed': curr_deformed_verts,
                'target_verts_normals_packed': target_normals,
                'description': f'Edge case: n_verts={n_verts}, zero vertices',
            })

        if num_tests > 7:
            torch.manual_seed(123)
            n_verts = 12
            max_edges = 5
            procrustes_precompute = self.create_procrustes_precompute(n_verts, max_edges, device)
            curr_deformed_verts = torch.randn(n_verts, 3, device=device, dtype=torch.float32)
            target_normals = torch.randn(n_verts, 3, device=device, dtype=torch.float32)
            target_normals = target_normals / (target_normals.norm(dim=-1, keepdim=True) + 1e-8)
            torch.manual_seed(self.seed)
            test_cases.append({
                'procrustes_precompute': procrustes_precompute,
                'curr_deformed_verts_packed': curr_deformed_verts,
                'target_verts_normals_packed': target_normals,
                'description': f'Different seed: n_verts={n_verts}, max_edges={max_edges}',
            })

        if num_tests > 8:
            n_verts = 10
            max_edges = 15
            procrustes_precompute = self.create_procrustes_precompute(n_verts, max_edges, device)
            curr_deformed_verts = torch.randn(n_verts, 3, device=device, dtype=torch.float32)
            target_normals = torch.randn(n_verts, 3, device=device, dtype=torch.float32)
            target_normals = target_normals / (target_normals.norm(dim=-1, keepdim=True) + 1e-8)
            test_cases.append({
                'procrustes_precompute': procrustes_precompute,
                'curr_deformed_verts_packed': curr_deformed_verts,
                'target_verts_normals_packed': target_normals,
                'description': f'Many edges: n_verts={n_verts}, max_edges={max_edges}',
            })

        if num_tests > 9:
            n_verts = 7
            max_edges = 6
            procrustes_precompute = self.create_procrustes_precompute(n_verts, max_edges, device)
            curr_deformed_verts = torch.randn(n_verts, 3, device=device, dtype=torch.float32)
            curr_deformed_verts = (curr_deformed_verts - curr_deformed_verts.mean()) / (curr_deformed_verts.std() + 1e-8)
            target_normals = torch.randn(n_verts, 3, device=device, dtype=torch.float32)
            target_normals = target_normals / (target_normals.norm(dim=-1, keepdim=True) + 1e-8)
            test_cases.append({
                'procrustes_precompute': procrustes_precompute,
                'curr_deformed_verts_packed': curr_deformed_verts,
                'target_verts_normals_packed': target_normals,
                'description': f'Normalized: n_verts={n_verts}, max_edges={max_edges}',
            })

        for i in range(num_tests - len(test_cases)):
            n_verts = 5 + (i % 15)
            max_edges = 3 + (i % 10)
            procrustes_precompute = self.create_procrustes_precompute(n_verts, max_edges, device)
            curr_deformed_verts = torch.randn(n_verts, 3, device=device, dtype=torch.float32)
            target_normals = torch.randn(n_verts, 3, device=device, dtype=torch.float32)
            target_normals = target_normals / (target_normals.norm(dim=-1, keepdim=True) + 1e-8)
            test_cases.append({
                'procrustes_precompute': procrustes_precompute,
                'curr_deformed_verts_packed': curr_deformed_verts,
                'target_verts_normals_packed': target_normals,
                'description': f'Additional test {i+1}: n_verts={n_verts}, max_edges={max_edges}',
            })

        return test_cases[:num_tests]
