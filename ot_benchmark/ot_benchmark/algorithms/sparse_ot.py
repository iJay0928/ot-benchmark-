"""Sparse-aware Sinkhorn — leverages CSR cost matrices for O(nnz) complexity.

For high-dimensional sparse data, the cost matrix is often sparse because
most pairs are far apart and contribute negligibly to the Sinkhorn kernel
K = exp(-M / reg). Using CSR matrix-vector products reduces each iteration
from O(n^2) to O(nnz).

Core insight: When M_ij is large, K_ij = exp(-M_ij / reg) ≈ 0. We only
need to consider entries where M_ij is below a cutoff.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def run_sparse_sinkhorn(
    source: np.ndarray,
    target: np.ndarray,
    cost_matrix: np.ndarray,
    cost_matrix_csr: csr_matrix | None = None,
    reg: float = 0.1,
    max_iter: int = 2000,
    stop_threshold: float = 1e-9,
) -> float:
    """Sinkhorn algorithm optimized for sparse cost matrices.

    Uses CSR matrix-vector products when cost_matrix_csr is provided,
    falling back to dense operations otherwise.

    Args:
        source: Source distribution.
        target: Target distribution.
        cost_matrix: Dense cost matrix (fallback).
        cost_matrix_csr: CSR sparse cost matrix (preferred).
        reg: Entropic regularization.
        max_iter: Maximum iterations.
        stop_threshold: Convergence threshold.

    Returns:
        Wasserstein distance estimate.
    """
    a = source.astype(np.float64).copy()
    b = target.astype(np.float64).copy()
    a = a / a.sum()
    b = b / b.sum()
    source_active = np.flatnonzero(a > 1e-15)
    target_active = np.flatnonzero(b > 1e-15)
    a = a[source_active]
    b = b[target_active]
    a = a / a.sum()
    b = b / b.sum()
    cost_matrix = cost_matrix[np.ix_(source_active, target_active)]
    if cost_matrix_csr is not None:
        cost_matrix_csr = cost_matrix_csr[source_active][:, target_active]
    n_source, n_target = len(a), len(b)

    use_sparse = (
        cost_matrix_csr is not None
        and cost_matrix_csr.shape == (n_source, n_target)
    )

    def dense_log_fallback() -> float:
        from .sinkhorn import run_sinkhorn
        return run_sinkhorn(
            a, b, cost_matrix, reg,
            max_iter=max_iter, stop_threshold=stop_threshold,
        )

    if use_sparse:
        # Build the sparse kernel directly so zero-cost diagonal entries remain.
        cutoff = min(float(np.max(cost_matrix)), -reg * np.log(1e-8))
        rows, cols = np.where(cost_matrix <= cutoff)
        kernel_values = np.exp(-cost_matrix[rows, cols] / reg)
        K_sp = csr_matrix(
            (kernel_values, (rows, cols)), shape=(n_source, n_target)
        )

        u = np.ones(n_source)
        for it in range(max_iter):
            kt_u = K_sp.T.dot(u)
            if np.any((b > 0) & (kt_u <= 1e-300)):
                return dense_log_fallback()
            v = b / np.maximum(kt_u, 1e-300)
            k_v = K_sp.dot(v)
            if np.any((a > 0) & (k_v <= 1e-300)):
                return dense_log_fallback()
            u_next = a / np.maximum(k_v, 1e-300)

            row_error = np.sum(np.abs(u_next * K_sp.dot(v) - a))
            col_error = np.sum(np.abs(v * K_sp.T.dot(u_next) - b))
            err = max(row_error, col_error)
            u = u_next
            if err < stop_threshold:
                break

        if not np.isfinite(err) or err > 1e-7:
            return dense_log_fallback()

        transport_values = u[rows] * kernel_values * v[cols]
        distance = float(np.dot(transport_values, cost_matrix[rows, cols]))
    else:
        distance = dense_log_fallback()

    return float(distance)
