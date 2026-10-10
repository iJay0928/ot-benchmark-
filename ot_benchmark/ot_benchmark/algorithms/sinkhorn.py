"""Sinkhorn (entropy-regularized) Optimal Transport."""

import numpy as np
import ot
try:
    from ..data_generator import validate_transport_plan
except ImportError:
    from data_generator import validate_transport_plan


def run_sinkhorn(
    source: np.ndarray,
    target: np.ndarray,
    cost_matrix: np.ndarray,
    reg: float = 0.1,
    max_iter: int = 10000,
    stop_threshold: float = 1e-9,
) -> float:
    """Compute entropic Wasserstein distance using the Sinkhorn algorithm.

    Args:
        source: Source distribution (will be normalized).
        target: Target distribution (will be normalized).
        cost_matrix: Pairwise cost matrix.
        reg: Entropic regularization strength.
        max_iter: Maximum Sinkhorn iterations.
        stop_threshold: Convergence threshold for marginal error.

    Returns:
        Sinkhorn distance (transport cost under regularized plan).
    """
    if reg <= 0 or max_iter < 1 or stop_threshold <= 0:
        raise ValueError("positive reg, max_iter and stop_threshold required")
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

    plan = ot.sinkhorn(
        a, b, M, reg,
        numItermax=max_iter,
        stopThr=stop_threshold,
        verbose=False,
        method="sinkhorn_log",
    )
    validate_transport_plan(plan, a, b)
    return float(np.sum(plan * M))
