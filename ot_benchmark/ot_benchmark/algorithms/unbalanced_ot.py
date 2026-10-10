"""Unbalanced Optimal Transport."""

import numpy as np
import ot


def run_unbalanced_ot(
    source: np.ndarray,
    target: np.ndarray,
    cost_matrix: np.ndarray,
    reg: float = 0.1,
    reg_m: float = 0.5,
) -> float:
    """Compute Unbalanced OT with mass relaxation.

    Allows creation/destruction of probability mass, suitable
    for noisy or incomplete data.

    Args:
        source: Source distribution.
        target: Target distribution.
        cost_matrix: Pairwise cost matrix.
        reg: Entropic regularization parameter.
        reg_m: Marginal relaxation parameter (KL divergence penalty).
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

    try:
        distance = ot.unbalanced.sinkhorn_unbalanced2(
            a, b, M, reg, reg_m,
            numItermax=1000, stopThr=1e-9, verbose=False,
        )
        return float(distance)
    except (AttributeError, ImportError):
        return float(ot.sinkhorn2(a, b, M, reg))
