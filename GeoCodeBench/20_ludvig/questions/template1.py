
"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
"""

import torch
import numpy as np


def normalize_torch(A, n):
    """
    Normalize sparse adjacency matrix using symmetric normalization.
    Returns D^{-1/2} @ A @ D^{-1/2}
    """
    # TODO: Implement this helper function if needed
    raise NotImplementedError("Please implement this function")


def get_stationary(
    self,
    neighbors,
    similarities,
    normalize=False,
    normalize_f=True,
    f=None,
    binarize=None,
    unary_term=None,
    symmetrize=True
):
    """
    Constructs adjacency matrix based on feature similarities and neighbor indices and runs graph diffusion.

    Args:
        neighbors (numpy.ndarray): Indices of node-to-node connections in the graph.
        similarities (torch.Tensor): Edge weights corresponding to the `neighbors` indices.
        normalize (bool, optional): Whether to normalize the graph adjacency matrix.
        normalize_f (bool, optional): Whether to normalize features at each iteration (power method).
        f (torch.Tensor, optional): Initial feature values for the nodes.
        binarize (float, optional): A threshold to binarize similarities.
            If set, similarities greater than this value are converted to 1, and others to 0.
        unary_term (torch.Tensor, optional): Regularization per node and per dimension.
        symmetrize (bool, optional): Whether to symmetrize the adjacency matrix.

    Returns:
        torch.Tensor: The features after graph diffusion.
    """
    n = self.knn_neighbor_indices.shape[0]
    if f is None:
        f = self.initial_features

    if binarize:
        # TODO: Fill in the code to binarize similarities
        pass

    row_indices = torch.tensor(neighbors[0])
    col_indices = torch.tensor(neighbors[1])

    if symmetrize:
        # TODO: Fill in the code to symmetrize the adjacency matrix
        # Should create A_coo sparse COO tensor and convert to CSR
        pass
        A = A_coo.to_sparse_csr()
    else:
        crow_indices = torch._convert_indices_from_coo_to_csr(row_indices, n)
        A = torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            similarities,
            (n, n),
            dtype=f.dtype,
            device=f.device
        )
    if normalize:
        if unary_term is not None:
            # TODO: Fill in the code to normalize with unary_term
            pass
        else:
            # TODO: Fill in the code to normalize A
            pass
    # TODO: Fill in the diffusion iteration loop
    return f
