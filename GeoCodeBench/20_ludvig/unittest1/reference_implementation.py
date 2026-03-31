"""
Reference Implementation for get_stationary
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch
import numpy as np


def normalize_torch(A, n):
    """
    Normalize sparse adjacency matrix using symmetric normalization.
    Returns D^{-1/2} @ A @ D^{-1/2}
    """
    # Compute degrees (row sums) using matrix multiplication
    # CSR format doesn't support torch.sparse.sum(), so use A @ ones
    ones = torch.ones((n, 1), dtype=A.dtype, device=A.device)
    degrees = (A @ ones).squeeze()
    degrees_inv_sqrt = torch.pow(degrees + 1e-8, -0.5)
    
    # Convert to COO for element-wise operations
    A_coo = A.to_sparse_coo()
    row_indices = A_coo.indices()[0]
    col_indices = A_coo.indices()[1]
    values = A_coo.values()
    
    # Apply D^{-1/2} on both row and column sides
    new_values = values * degrees_inv_sqrt[row_indices] * degrees_inv_sqrt[col_indices]
    
    A_normalized = torch.sparse_coo_tensor(
        A_coo.indices(),
        new_values,
        A_coo.shape,
        dtype=A.dtype,
        device=A.device
    ).to_sparse_csr()
    
    return A_normalized


class GraphDiffusion:
    """Mock GraphDiffusion class for testing get_stationary method."""
    
    def __init__(self, knn_neighbor_indices, initial_features, num_iterations, eps=1e-8):
        """
        Args:
            knn_neighbor_indices: KNN neighbor indices (n, k)
            initial_features: Initial feature values (n, feature_dim)
            num_iterations: Number of diffusion iterations
            eps: Small value for numerical stability
        """
        self.knn_neighbor_indices = knn_neighbor_indices
        self.initial_features = initial_features
        self.num_iterations = num_iterations
        self.eps = eps
        self.trace_name = None  # For compatibility with LLM implementations
    
    def trace_f(self, f, t):
        """Dummy trace_f method - does nothing during testing."""
        pass
    
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
            similarities = (similarities>binarize).type(torch.float32)

        row_indices = torch.tensor(neighbors[0], device=f.device)
        col_indices = torch.tensor(neighbors[1], device=f.device)

        if symmetrize:
            row_sym = torch.cat([row_indices, col_indices])
            col_sym = torch.cat([col_indices, row_indices])
            values_sym = torch.cat([similarities, similarities])
            A_coo = torch.sparse_coo_tensor(
                torch.stack([row_sym, col_sym]),
                values_sym,
                (n, n),
                device=f.device,
                dtype=f.dtype
            ).coalesce()
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
                unary_term /= torch.sqrt(unary_term) * (A @ torch.sqrt(unary_term)) + 1e-8
            else:
                A = normalize_torch(A, n)
        for i in range(self.num_iterations):
            if normalize_f:
                f /= self.eps + f.norm(dim=0, keepdim=True)
            if unary_term is not None:
                f = torch.sqrt(unary_term) * (A @ (torch.sqrt(unary_term) * f))
            else:
                f = A @ f
        return f

