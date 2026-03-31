from typing import Tuple, Literal, Optional, List, Callable, Sequence, Union, cast, Set
import os
import random
import warnings
from enum import Enum
from dataclasses import dataclass

# pytorch3d cot_laplacian still uses SparseTensor rather than sparse_coo_tensor
# Until upstream updates, there will be a deprecation warning, which we'll silence
warnings.filterwarnings("ignore", message="torch.sparse.SparseTensor")

import torch

TORCH_PLS_BE_DETERMINISTIC = os.environ.get("TORCH_PLS_BE_DETERMINISTIC")
if TORCH_PLS_BE_DETERMINISTIC:
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import igl
import cholespy
import numpy as np
import torch.nn as nn

from pytorch3d.structures import Meshes
import pytorch3d.transforms as pt3d_transforms
import pytorch3d.ops as pt3d_ops

import thad
from thlog import (
    Thlogger,
    LOG_INFO,
    LOG_DEBUG,
    LOG_TRACE,
    LOG_NONE,
    VIZ_INFO,
    VIZ_DEBUG,
    VIZ_TRACE,
    VIZ_NONE,
    _PolyscopeRegisteredStructProxy,
)
from thronf import Thronfig, InvalidConfigError

thlog = Thlogger(LOG_INFO, VIZ_INFO, "deformations")




@dataclass(slots=True)
class ProcrustesPrecompute:
    padded_cell_edges_per_vertex_packed: torch.Tensor
    """
    (n_verts_packed, max_cell_neighborhood_n_edges, 2) int tensor; last dim contains edge vertex indices.
    negative ints are padding
    """
    covar_lefts_packed: torch.Tensor
    """
    (n_verts_packed, 3, max_cell_neighborhood_n_edges + 1)
    which is found by a batch matmul between
    (n_verts_packed, 3, max_cell_neighborhood_n_edges + 1) bmm (n_verts_packed, max_cell_neighborhood_n_edges + 1,max_cell_neighborhood_n_edges + 1)

    left-multiplies with a (max_cell_neighborhood_n_edges + 1, 3) matrix
    which is formed by grabbing the edge vectors corresponding to pcepv_packed, which
    would be (n_verts_packed, max_cell_neighborhood_n_edges,3) concatenated with
    the target normals (with dim1 unsqueezed so with shape (n_verts_packed, 1,
    3)) in dim1.

    then we batch_svd solve this (n_verts_packed, 3, 3) matrix to get the rotation
    """
    # bookkeeping for indexing
    _verts_packed_idxr: MeshesPackedIndexer
    _num_verts_per_mesh: torch.Tensor
    _mesh_to_verts_packed_first_idx: torch.Tensor

    @classmethod
    def from_meshes(
        cls,
        local_step_procrustes_lambda: float,
        arap_energy_type: Optional[ARAPEnergyTypeName],
        laplacians_solvers: "SparseLaplaciansSolvers",
        patient_meshes: Meshes,
    ):
        """
        (need the laplacians solvers just for the laplacian weights)
        """
        thlog.info("Calculating procrustes solve precomputation")
        verts_packed = patient_meshes.verts_packed()

        n_verts_packed = len(verts_packed)
        pcepv_packed: Tuple[Set[Tuple[int, int]], ...] = tuple(
            set() for _ in range(n_verts_packed)
        )
        need_spokes_and_rims = (
            arap_energy_type == "spokes_and_rims_mine"
            or arap_energy_type == "spokes_and_rims_igl"
        )
        for v0i_, v1i_, v2i_ in patient_meshes.faces_packed():
            v0i = int(v0i_.item())
            v1i = int(v1i_.item())
            v2i = int(v2i_.item())

            # correct procrustes neighborhood with directed edges, and each face
            # only contributing the edges that go in its CCW orientation
            e01i = (v0i, v1i)
            e12i = (v1i, v2i)
            e20i = (v2i, v0i)

            pcev0_set = pcepv_packed[v0i]
            pcev1_set = pcepv_packed[v1i]
            pcev2_set = pcepv_packed[v2i]

            # add spokes (radiating from vertex)
            pcev0_set.add(e01i)
            pcev1_set.add(e12i)
            pcev2_set.add(e20i)
            if need_spokes_and_rims:
                # other face-edge pointing to vertex
                pcev0_set.add(e20i)
                pcev1_set.add(e01i)
                pcev2_set.add(e12i)
                # rims
                pcev0_set.add(e12i)
                pcev1_set.add(e20i)
                pcev2_set.add(e01i)

        cell_neighborhood_n_edges = tuple(map(len, pcepv_packed))
        max_cell_neighborhood_n_edges = max(cell_neighborhood_n_edges)
        f: Callable[[Tuple[Set[Tuple[int, int]], int]], Tuple[Tuple[int, int], ...]] = (
            lambda _tup: (
                _set := _tup[0],
                _setlen := _tup[1],
                (
                    tuple(_set)
                    + tuple(
                        (-1, -1) for _ in range(max_cell_neighborhood_n_edges - _setlen)
                    )
                )
                if _setlen < max_cell_neighborhood_n_edges
                else tuple(_set),
            )[-1]
        )
        z = zip(pcepv_packed, cell_neighborhood_n_edges)
        pcepv_packed_tuples = tuple(map(f, z))
        padded_cell_edges_per_vertex_packed = torch.tensor(
            pcepv_packed_tuples, device=patient_meshes.device
        )
        thlog.debug("[procrustes precompute] done padded_cell_edges_per_vertex")
        ######################################## done computing padded_cell_edges_per_vertex

        cell_laplacian_weights_list = []
        for L, verts_packed_first_idx, n_verts in zip(
            laplacians_solvers.Ls,
            patient_meshes.mesh_to_verts_packed_first_idx(),
            patient_meshes.num_verts_per_mesh(),
        ):
            pcepv_this_mesh = (
                padded_cell_edges_per_vertex_packed[
                    verts_packed_first_idx : verts_packed_first_idx + n_verts
                ]
                - verts_packed_first_idx
            )
            pcepv_v0i_this_mesh = pcepv_this_mesh[:, :, 0]
            pcepv_v1i_this_mesh = pcepv_this_mesh[:, :, 1]
            pcepv_shape = pcepv_v1i_this_mesh.shape
            # cell_laplacian_weights_this_mesh = index_sparse_coo_matrix_rowcol(
            #     L, pcepv_v0i_this_mesh.flatten(), pcepv_v1i_this_mesh.flatten()
            # ).view(pcepv_shape)
            # ^ this runs out of mem on my laptop!
            # let's chunk this operation
            pcepv_v0i_numel = pcepv_v0i_this_mesh.numel()
            cell_laplacian_weights_this_mesh = torch.zeros(
                (pcepv_v0i_numel,), device=L.device, dtype=L.dtype
            )
            # each chunk fills the laplacian weights array for SPLITSZ edges
            SPLITSZ = 8192
            for chunk_idxs, v0idxs_this_chunk, v1idxs_this_chunk in zip(
                torch.arange(pcepv_v0i_numel).split(SPLITSZ),
                pcepv_v0i_this_mesh.flatten().split(SPLITSZ),
                pcepv_v1i_this_mesh.flatten().split(SPLITSZ),
            ):
                # dense indexing is way faster so we fetch the rows sparsely and index their columns densely
                L_v0idxs_this_chunk = L.index_select(0, v0idxs_this_chunk).to_dense()
                cell_laplacian_weights_this_mesh[chunk_idxs] = L_v0idxs_this_chunk[
                    torch.arange(v1idxs_this_chunk.size(-1)), v1idxs_this_chunk
                ]
                ## the older way:
                # cell_laplacian_weights_this_mesh[chunk_idxs] = (
                #     index_sparse_coo_matrix_rowcol(L, v0idxs_this_chunk, v1idxs_this_chunk)
                # )

            cell_laplacian_weights_this_mesh = cell_laplacian_weights_this_mesh.view(
                pcepv_shape
            ).to(patient_meshes.device)

            # wherever pcepv is negative, that's padding
            cell_laplacian_weights_this_mesh[pcepv_v1i_this_mesh < 0] = 0
            # ^ (n_verts, max_cell_neighborhood_n_edges,) float
            cell_laplacian_weights_list.append(cell_laplacian_weights_this_mesh)
        cell_laplacian_weights_packed = torch.cat(cell_laplacian_weights_list, dim=0)
        # (n_verts_packed, max_cell_neighborhood_n_edges,)
        thlog.debug("[procrustes precompute] done cotangent weights")
        ######################################## done putting cotan weights into pcepv format

        voronoi_verts_massmatrix__scipy = igl.massmatrix(
            verts_packed.cpu().detach().numpy(),
            patient_meshes.faces_packed().cpu().detach().numpy(),
        )
        # get diag of this thing
        voronoi_verts_mass_packed = (
            torch.from_numpy(voronoi_verts_massmatrix__scipy.diagonal())
            .float()
            .to(patient_meshes.device)
        ) * local_step_procrustes_lambda
        # ^ (n_verts_packed)
        assert voronoi_verts_mass_packed.shape == (n_verts_packed,)

        # form the middle neighborhood-size-by-neighborhood-size matrix
        diags_packed = torch.cat(
            (cell_laplacian_weights_packed, voronoi_verts_mass_packed.unsqueeze(-1)), dim=-1
        )
        # ^ (n_verts_packed, max_cell_neighborhood_n_edges + 1)
        diagmats_packed = torch.diag_embed(diags_packed)
        # ^ (n_verts_packed, max_cell_neighborhood_n_edges+1, max_cell_neighborhood_n_edges+1, )
        thlog.debug("[procrustes precompute] done diagonal matrix")
        if thlog.logguard(LOG_TRACE):
            torch.set_printoptions(precision=1)
            np.set_printoptions(precision=3)
            thlog.trace(f"""
            vertsmass
{voronoi_verts_mass_packed}
            diags:
{diags_packed.cpu().detach().numpy()}
            diagmats
            {diagmats_packed.cpu().detach().numpy()}
            L
            {laplacians_solvers.Ls[0].to_dense().cpu().detach().numpy()}
            pcepv
            {padded_cell_edges_per_vertex_packed}
            """)
            # assert False
        ############################################### done making middle diag matrix

        # NOTE this bit of code is also how you compute the covar_rights matrix
        # for the in-progress edge vecs and target normals (target vert normals taking the place of original_vert_normals)

        pcepv_v1i = padded_cell_edges_per_vertex_packed[:, :, 1]
        pcepv_v0i = padded_cell_edges_per_vertex_packed[:, :, 0]
        original_cell_edge_vecs_packed = verts_packed[pcepv_v1i] - verts_packed[pcepv_v0i]
        # ^ (n_verts_packed, max_cell_neighborhood_n_edges, 3)
        # zero out wherever there is padding
        original_cell_edge_vecs_packed[pcepv_v1i < 0] = 0
        original_vert_normals = patient_meshes.verts_normals_packed().unsqueeze(1)
        # ^ (n_verts_packed, 1, 3)
        covar_lefts_lefts_packed = torch.cat(
            (original_cell_edge_vecs_packed, original_vert_normals), dim=1
        )
        # ^ (n_verts_packed, max_cell_neighborhood_n_edges + 1, 3)
        covar_lefts_packed = covar_lefts_lefts_packed.transpose(-1, -2).bmm(diagmats_packed)
        # ^ (n_verts_packed, 3, max_cell_neighborhood_n_edges + 1)
        thlog.debug("[procrustes precompute] done covariance matrix")

        # make misc indexing bookkeeping
        _num_verts_per_mesh = patient_meshes.num_verts_per_mesh()
        _verts_packed_idxr = MeshesPackedIndexer.from_num_per_mesh(_num_verts_per_mesh)
        return cls(
            padded_cell_edges_per_vertex_packed=padded_cell_edges_per_vertex_packed.to(
                patient_meshes.device
            ),
            covar_lefts_packed=covar_lefts_packed,
            _verts_packed_idxr=_verts_packed_idxr,
            _num_verts_per_mesh=_num_verts_per_mesh,
            _mesh_to_verts_packed_first_idx=patient_meshes.mesh_to_verts_packed_first_idx(),
        )

    def __getitem__(self, mesh_indices: Union[int, List[int], torch.Tensor]):
        new_packed_idxr = self._verts_packed_idxr[mesh_indices]
        pcepv_packed_to_mesh_idx = torch.arange(
            new_packed_idxr.n_meshes_in_batch(),
            device=new_packed_idxr.num_per_mesh.device,
        ).repeat_interleave(new_packed_idxr.num_per_mesh)
        packed_idx = self._verts_packed_idxr(mesh_indices)
        new_pcepv_packed_noadjust = self.padded_cell_edges_per_vertex_packed[packed_idx]

        # apply offset adjustment to the indices inside new_faceadj_noadjust
        new_num_verts_per_mesh = self._num_verts_per_mesh[mesh_indices]
        new_mesh_to_verts_packed_first_idx = (
            torch.cumsum(new_num_verts_per_mesh, dim=0) - new_num_verts_per_mesh
        )
        old_mesh_to_verts_packed_first_idx = self._mesh_to_verts_packed_first_idx[
            mesh_indices
        ]
        new_pcepv_packed_adjusted = (
            new_pcepv_packed_noadjust
            - old_mesh_to_verts_packed_first_idx[pcepv_packed_to_mesh_idx, None, None]
            + new_mesh_to_verts_packed_first_idx[pcepv_packed_to_mesh_idx, None, None]
        )
        return __class__(
            padded_cell_edges_per_vertex_packed=new_pcepv_packed_adjusted,
            covar_lefts_packed=self.covar_lefts_packed[packed_idx],
            _verts_packed_idxr=new_packed_idxr,
            _num_verts_per_mesh=new_num_verts_per_mesh,
            _mesh_to_verts_packed_first_idx=new_mesh_to_verts_packed_first_idx,
        )


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
    if thlog.guard(VIZ_TRACE, needs_polyscope=True):
        pcepv_v1_for_vertex0 = pcepv_v1[0]
        pcepv_v1_for_vertex0 = pcepv_v1_for_vertex0[pcepv_v1i[0] >= 0]
        pcepv_v0_for_vertex0 = pcepv_v0[0]
        pcepv_v0_for_vertex0 = pcepv_v0_for_vertex0[pcepv_v0i[0] >= 0]
        cunet_pts = torch.cat((pcepv_v0_for_vertex0, pcepv_v1_for_vertex0), dim=0)
        cunet_edges = torch.stack(
            (
                torch.arange(len(pcepv_v0_for_vertex0)),
                len(pcepv_v0_for_vertex0) + torch.arange(len(pcepv_v0_for_vertex0)),
            ),
            dim=-1,
        )

        thlog.psr.register_curve_network(
            "v0 cell neigh",
            cunet_pts.cpu().detach().numpy(),
            cunet_edges.cpu().detach().numpy(),
        )
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
    if thlog.logguard(LOG_TRACE):
        thlog.trace(f"entries to flip {entries_to_flip}")
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

