"""Greenkhorn (greedy coordinate-descent Sinkhorn)."""

import numpy as np
import ot
try:
    from ..data_generator import validate_transport_plan
except ImportError:
    from data_generator import validate_transport_plan


def run_greenkhorn(
    source: np.ndarray,
    target: np.ndarray,
    cost_matrix: np.ndarray,
    reg: float = 0.1,
    max_iter: int = 1000,
) -> float:
    """Compute OT distance with the Greenkhorn greedy Sinkhorn variant.

    Calls POT's Greenkhorn plan solver. Unavailable or non-converged
    implementations are reported as failures, never relabeled Sinkhorn.
    """
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

    plan = ot.bregman.greenkhorn(
        a, b, M, reg, numItermax=max_iter, stopThr=1e-9
    )
    validate_transport_plan(plan, a, b)
    return float(np.sum(plan * M))
