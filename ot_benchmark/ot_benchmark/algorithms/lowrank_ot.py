"""Low-Rank Optimal Transport via SVD approximation."""

import numpy as np
import ot


def run_lowrank_ot(
    source: np.ndarray,
    target: np.ndarray,
    cost_matrix: np.ndarray,
    rank: int = 10,
    reg: float = 0.1,
) -> float:
    """Compute OT distance using low-rank SVD approximation of the cost matrix.

    Reduces complexity from O(n^2) to O(n × r) per Sinkhorn iteration.

    Args:
        source: Source distribution.
        target: Target distribution.
        cost_matrix: Pairwise cost matrix (will be low-rank approximated).
        rank: Target rank for SVD truncation.
        reg: Entropic regularization parameter.
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
    active_cost = cost_matrix[np.ix_(source_active, target_active)]

    try:
        U, S, Vt = np.linalg.svd(active_cost, full_matrices=False)
        r = min(rank, len(S))
        lowrank_M = (U[:, :r] * S[:r]) @ Vt[:r, :]
        lowrank_M = np.abs(lowrank_M)
        return float(ot.sinkhorn2(a, b, lowrank_M, reg, method="sinkhorn_log"))
    except (np.linalg.LinAlgError, ValueError, MemoryError):
        return float(ot.sinkhorn2(a, b, active_cost, reg, method="sinkhorn_log"))
