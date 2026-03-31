
"""
LLM Template for calc_ARAP_global_solve()
This file shows the template format for LLM code completion.
The EMPTY section should be filled by the LLM.
"""

import sys
import os
from pathlib import Path
from typing import Optional

# Add parent directory to path to import from original file
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from deformations_dARAP import (
    Meshes,
    SparseLaplaciansSolvers,
    ARAPEnergyTypeName,
    PostprocessAfterSolveName,
    index_sparse_coo_matrix_rowcol,
    CholespySymmetricSolve_AutogradFn,
    recenter_to_centroid,
    recenter_to_centroid_and_rescale_new_verts_to_fit_old_bboxes,
)
import torch
import torch.nn as nn

# Note: calc_ARAP_global_solve__better_with_rhs_lefts is not imported here
# as it's only used in the IGL branch which is not part of the code completion task


def calc_ARAP_global_solve(
    meshes: Meshes,
    laplacians_solvers: SparseLaplaciansSolvers,
    per_vertex_rot_matrices_packed: torch.Tensor,
    arap_energy_type: ARAPEnergyTypeName,
    postprocess: Optional[PostprocessAfterSolveName],
) -> torch.Tensor:
    """
    per_vertex_rot_matrices_packed: shape (n_verts_packed, 3, 3)
    returns the ARAP global solve result, i.e. solution p' such that
    Lp' = b (equation 9 in the ARAP paper)
    where L is the cotangent laplacian, and b is the right hand side described in the paper
    """
    if laplacians_solvers.igl_arap_rhs_lefts is not None:
        assert (
            arap_energy_type == "spokes_and_rims_igl" or arap_energy_type == "spokes_igl"
        ), (
            "solver has IGL ARAP RHS constructors so must use spokes_and_rims_igl or spokes_igl for arap_energy_type"
        )
        # the rhs constructor from IGL is available, use this other function!
        # this goes with arap_energy_type == "spokes_igl" and "spokes_and_rims_igl"
        # Note: This branch is not part of the code completion task
        # For testing purposes, we'll raise an error if this branch is reached
        raise NotImplementedError("IGL ARAP RHS branch not implemented in template")
    # the rest of this function here computes rhs directly from the 2004 paper formula
    # if arap_energy_type == spokes_mine, or the rhs from the spokes-and-rims energy
    # from Chao et al 2011 (also used in normal analogies; formula described in CGAL docs)
    # the rhs to find has shape (n_verts, 3)
    solutions = []
    # can we write a batched version of this without a loop? (though we'll still end up
    # looping through the cholespy solvers anyhow...)
    for i, (L, verts_padded, faces, n_verts_this_mesh, verts_packed_first_idx) in enumerate(
        zip(
            laplacians_solvers.Ls,
            meshes.verts_padded(),
            meshes.faces_list(),
            meshes.num_verts_per_mesh(),
            meshes.mesh_to_verts_packed_first_idx(),
        )
    ):
        # it might be possible that verts_padded for this mesh has shorter dim0 length than
        # L because the meshes might have been indexed from a larger meshes batch, with
        # padding shrunken to fit just the largest mesh in the extracted batch. in this
        # case, we expand verts with padding to match the dim0 and dim1 size of the square L
        if (vp_sz0 := verts_padded.size(0)) < (L_sz0 := L.size(0)):
            verts_padded = nn.functional.pad(verts_padded, (0, 0, 0, L_sz0 - vp_sz0))

        # for each edge between a vertex i and vertex j, compute (w_ij / 2) * ((R_i
        # + R_j) @ (p_i - p_j)) (this is a 3d point)
        L = L.coalesce()

        if arap_energy_type == "spokes_mine":
            L_sp_indices = L.indices()
            # there are (2*n_edges) directed edges. we can get an array of (directed)edge
            # weights directly from the COO tensor's values array rather than going through
            # confusing index_selects (sparse tensors don't support tensor indexing). For some
            # reason the ordering has to be this way, for the i and j vert order in each index
            # in the ARAP rhs formula. If I were to do the obvious (i is indices[0]) then each
            # solve would give the right result except the y axis is flipped (?!)
            dir_edges_vi = L_sp_indices[1]
            dir_edges_vj = L_sp_indices[0]
            dir_edges_weight = L.values()

            # Ri + Rj
            rot_vi_plus_rot_vj = (
                per_vertex_rot_matrices_packed[dir_edges_vi + verts_packed_first_idx]
                + per_vertex_rot_matrices_packed[dir_edges_vj + verts_packed_first_idx]
            )

            # pi - pj
            pi_minus_pj = verts_padded[dir_edges_vi] - verts_padded[dir_edges_vj]

            # (w_ij / 2) * ((R_i + R_j) @ (p_i - p_j))
            rhs_per_dir_edge = (dir_edges_weight / 2).unsqueeze(1) * rot_vi_plus_rot_vj.bmm(
                pi_minus_pj.unsqueeze(-1)
            ).squeeze(-1)

            # the rhs vector is the same shape as verts_padded; then each slot corresponding to
            # vertex index j in the rhs vector is the sum of the values of the directed edges
            # out of vertex j. Here we use j because it corresponded to L_sp_indices[0]; if we
            # use i, then we still get the right system soln but the y axis is flipped (probably
            # a sign bug that managed to cancel out if I do this ij swap..)
            # rhs = torch.index_add(torch.zeros_like(verts), 0, dir_edges_vj, rhs_per_dir_edge)
            rhs = torch.index_put(
                torch.zeros_like(verts_padded),
                (dir_edges_vj,),
                rhs_per_dir_edge,
                accumulate=True,
            )
            # for this, index_put gives essentially the same result as index_add
            # there, but is not undefined behavior on duplicate indices, unlike index_add
        elif arap_energy_type == "spokes_and_rims_mine":
            faces_v0idx = faces[:, 0]
            faces_v1idx = faces[:, 1]
            faces_v2idx = faces[:, 2]

            ****EMPTY****
            
        else:
            raise AssertionError(
                f"shouldn't use calc_ARAP_global_solve with this arap_energy_type setting: {arap_energy_type} (if it's an igl arap energy type, maybe I forgot to init the solvers with igl_arap_rhs_lefts)"
            )

        # now do the solve
        solver = laplacians_solvers.cholespy_solvers[i]
        if _haspin := (laplacians_solvers.removed_first_L_column is not None):
            removed_first_L_column = laplacians_solvers.removed_first_L_column[i]
            # removed first L column has shape (n_verts, 1)
            # verts[None, 0] has shape (1, 3)
            # multiplied has shape (n_verts, 3)
            rhs = (rhs[:vp_sz0] - removed_first_L_column * verts_padded[None, 0])[1:vp_sz0]
        else:
            rhs = rhs[:vp_sz0]

        soln = CholespySymmetricSolve_AutogradFn.apply(solver, rhs)
        assert isinstance(soln, torch.Tensor)
        # soln won't have any padding to trim because solver's system matrix is not padded
        if _haspin:
            soln = torch.cat((verts_padded[None, 0], soln), dim=0)
        solutions.append(soln)

    soln_verts_packed = torch.cat(solutions, dim=0)  # (n_verts_packed,3)
    if postprocess == "recenter_rescale":
        soln_verts_packed = recenter_to_centroid_and_rescale_new_verts_to_fit_old_bboxes(
            meshes, soln_verts_packed
        )
    elif postprocess == "recenter_only":
        soln_verts_packed = recenter_to_centroid(meshes, soln_verts_packed)

    return soln_verts_packed
