"""GPU-accelerated Sinkhorn using PyTorch CUDA tensors.

Provides a native PyTorch implementation of the Sinkhorn algorithm
that runs entirely on GPU, avoiding CPU-GPU transfers during iteration.
"""

from __future__ import annotations

import numpy as np


def run_sinkhorn_gpu(
    source: np.ndarray,
    target: np.ndarray,
    cost_matrix: np.ndarray,
    reg: float = 0.1,
    max_iter: int = 2000,
    stop_threshold: float = 1e-9,
    backend=None,  # optional, for compatibility with run_single_algorithm
    cost_matrix_csr=None,  # optional, for compatibility
) -> float:
    """Compute Sinkhorn distance using PyTorch GPU acceleration.

    Performs the entire Sinkhorn iteration on GPU. For small problems
    the overhead of GPU transfer may dominate; for n > 500 the GPU
    typically provides significant speedup.

    Args:
        source: Source distribution (will be normalized).
        target: Target distribution (will be normalized).
        cost_matrix: Pairwise cost matrix (n, n).
        reg: Entropic regularization strength.
        max_iter: Maximum Sinkhorn iterations.
        stop_threshold: Convergence threshold for marginal error.

    Returns:
        Sinkhorn distance.
    """
    import torch

    if not torch.cuda.is_available():
        # Fall back to CPU Sinkhorn via POT
        from .sinkhorn import run_sinkhorn
        return run_sinkhorn(source, target, cost_matrix, reg, max_iter, stop_threshold)

    device = torch.device("cuda")
    dtype = torch.float64

    a = source.astype(np.float64).copy()
    b = target.astype(np.float64).copy()
    M = cost_matrix.astype(np.float64).copy()

    a = a / a.sum()
    b = b / b.sum()
    source_active = np.flatnonzero(a > 1e-15)
    target_active = np.flatnonzero(b > 1e-15)
    a = a[source_active]
    b = b[target_active]
    a = a / a.sum()
    b = b / b.sum()
    M = M[np.ix_(source_active, target_active)]

    a_t = torch.from_numpy(a).to(device, dtype=dtype)
    b_t = torch.from_numpy(b).to(device, dtype=dtype)
    M_t = torch.from_numpy(M).to(device, dtype=dtype)

    # Sinkhorn kernel
    K = torch.exp(-M_t / reg)  # (n, n) on GPU

    u = torch.ones_like(a_t)

    for _ in range(max_iter):
        # Column scaling
        v = b_t / (K.T @ u + 1e-16)
        # Row scaling
        u = a_t / (K @ v + 1e-16)
        # Check column marginal error: how well does v * K.T @ u match b?
        err = torch.max(torch.abs(v * (K.T @ u) - b_t))
        if err < stop_threshold:
            break

    # Final distance: sum over transport plan P = diag(u) * K * diag(v)
    P = u.unsqueeze(1) * K * v.unsqueeze(0)
    distance = torch.sum(P * M_t)

    return float(distance.cpu())
