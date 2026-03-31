from __future__ import annotations

import torch
from typing import Tuple
from sklearn.cluster import KMeans

def simplify_gmm_vectorized(mu: torch.Tensor, cov: torch.Tensor, w: torch.Tensor, num_clusters: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reference implementation of simplify_gmm_vectorized (CPU-only).
    Matches the original algorithm: KMeans clustering on mu, weight-aware merge.

    Args:
        mu: Tensor of shape (N, 3)
        cov: Tensor of shape (N, 3, 3)
        w: Tensor of shape (N,)
        num_clusters: int, number of clusters (K)

    Returns:
        new_mu: (K_valid, 3)
        new_cov: (K_valid, 3, 3)
        new_w: (K_valid,)
    """
    original_device = mu.device
    mu_np = mu.cpu().numpy()
    w = w.to(original_device)

    # KMeans clustering on CPU
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(mu_np)
    cluster_ids_tensor = torch.tensor(cluster_ids, device=original_device, dtype=torch.long)

    # Total weight per cluster
    total_weight_per_cluster = torch.bincount(cluster_ids_tensor, weights=w, minlength=num_clusters)

    # Weighted mean per cluster
    sum_mu = torch.zeros(num_clusters, mu.size(1), device=original_device)
    sum_mu.index_add_(0, cluster_ids_tensor, w[:, None] * mu)
    merged_mu_all = sum_mu / (total_weight_per_cluster[:, None] + 1e-8)

    # Delta per component to its cluster mean
    merged_mu_per_component = merged_mu_all[cluster_ids_tensor]
    delta = mu - merged_mu_per_component

    # Expanded covariance and weighted sum per cluster
    expanded_cov = cov + torch.einsum('ni,nj->nij', delta, delta)
    sum_cov = torch.zeros(num_clusters, mu.size(1), mu.size(1), device=original_device)
    sum_cov.index_add_(0, cluster_ids_tensor, w[:, None, None] * expanded_cov)
    merged_cov_all = sum_cov / (total_weight_per_cluster[:, None, None] + 1e-8)

    # Filter empty clusters and normalize weights
    valid_clusters = total_weight_per_cluster > 1e-8
    new_mu = merged_mu_all[valid_clusters]
    new_cov = merged_cov_all[valid_clusters]
    new_w = total_weight_per_cluster[valid_clusters]
    new_w = new_w / (new_w.sum() + 1e-8)

    return new_mu, new_cov, new_w
