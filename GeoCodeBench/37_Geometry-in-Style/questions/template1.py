
"""
LLM Template for calc_rot_matrices_with_procrustes()
This is the template that LLMs should complete by filling in the EMPTY section.
DO NOT include any hints or comments in the EMPTY section - only code.
"""

from typing import Union, List, Tuple
from dataclasses import dataclass
import torch


@dataclass(slots=True)
class MeshesPackedIndexer:
    """Simplified version for testing purposes."""
    padded_aranges: torch.Tensor
    num_per_mesh: torch.Tensor
    mesh_to_packed_first_idx: torch.Tensor

    @classmethod
    def from_num_per_mesh(cls, num_per_mesh: torch.Tensor):
        """Create indexer from number of elements per mesh."""
        assert not num_per_mesh.is_floating_point()
        assert num_per_mesh.ndim == 1
        packed_sz = int(num_per_mesh.sum().item())
        mesh_to_packed_first_idx = torch.cumsum(num_per_mesh, dim=0) - num_per_mesh
        return cls(
            padded_aranges=torch.stack(
                tuple(
                    torch.nn.functional.pad(
                        torch.arange(_n := int(n.item())),
                        (0, packed_sz - _n),
                        value=-999999999999,
                    )
                    for n in num_per_mesh
                ),
                dim=0,
            ).to(num_per_mesh.device),
            num_per_mesh=num_per_mesh,
            mesh_to_packed_first_idx=mesh_to_packed_first_idx.unsqueeze(-1),
        )

    def n_meshes_in_batch(self) -> int:
        return self.num_per_mesh.size(0)


@dataclass(slots=True)
class ProcrustesPrecompute:
    """Precomputed data for Procrustes analysis."""
    padded_cell_edges_per_vertex_packed: torch.Tensor
    """
    (n_verts_packed, max_cell_neighborhood_n_edges, 2) int tensor; last dim contains edge vertex indices.
    negative ints are padding
    """
    covar_lefts_packed: torch.Tensor
    """
    (n_verts_packed, 3, max_cell_neighborhood_n_edges + 1)
    """
    # bookkeeping for indexing
    _verts_packed_idxr: MeshesPackedIndexer
    _num_verts_per_mesh: torch.Tensor
    _mesh_to_verts_packed_first_idx: torch.Tensor


def calc_rot_matrices_with_procrustes(
    procrustes_precompute: ProcrustesPrecompute,
    curr_deformed_verts_packed: torch.Tensor,
    target_verts_normals_packed: torch.Tensor,
) -> torch.Tensor:
    """
    curr_deformed_verts_packed (n_verts_packed, 3)
    target_normals_packed (n_verts_packed, 3), the targeted normals
    """
    pcepv_v1i = procrustes_precompute.padded_cell_edges_per_vertex_packed[:, :, 1]
    pcepv_v0i = procrustes_precompute.padded_cell_edges_per_vertex_packed[:, :, 0]
    pcepv_v1 = curr_deformed_verts_packed[pcepv_v1i]
    pcepv_v0 = curr_deformed_verts_packed[pcepv_v0i]
    current_cell_edge_vecs_packed = pcepv_v1 - pcepv_v0
    # ^ (n_verts_packed, max_cell_neighborhood_n_edges, 3)
    current_cell_edge_vecs_packed[pcepv_v1i < 0] = 0
    # ^ (n_verts_packed, max_cell_neighborhood_n_edges, 3)
    target_verts_normals_packed = target_verts_normals_packed.unsqueeze(1)
    # ^ (n_verts_packed, 1, 3)
    
    ****EMPTY****

    # vvt[:, [0, 1]] *= -1  # it does not work if i don't do this
    # not inplace:
    vvt = torch.stack((-vvt[:, 0], -vvt[:, 1], vvt[:, 2]), dim=1)
    rots_packed = vvt.transpose(-1, -2).bmm(uu.transpose(-1, -2))

    # svd det correction code from https://github.com/OllieBoyne/pytorch-arap/blob/master/pytorch_arap/arap.py
    # for any det(Ri) <= 0
    entries_to_flip = torch.nonzero(rots_packed.det() <= 0, as_tuple=False).flatten()
    # ^idxs where det(R) <= 0
    if len(entries_to_flip) > 0:
        uumod = uu.clone()
        # minimum singular value is the last one
        uumod[entries_to_flip, :, -1] *= -1  # flip cols
        rots_packed[entries_to_flip] = (
            vvt[entries_to_flip]
            .transpose(-1, -2)
            .bmm(uumod[entries_to_flip].transpose(-1, -2))
        )

    return rots_packed.transpose(-1, -2)
