"""Exact Earth Mover's Distance (network simplex)."""

import numpy as np
import ot


def run_emd_exact(
    source: np.ndarray,
    target: np.ndarray,
    cost_matrix: np.ndarray,
) -> float:
    """Compute exact Wasserstein distance via network simplex.

    Only suitable for small problems (n <= 300).
    """
    a = source.astype(np.float64).copy()
    b = target.astype(np.float64).copy()
    a = a / a.sum()
    b = b / b.sum()
    source_active = np.flatnonzero(a > 1e-15)
    target_active = np.flatnonzero(b > 1e-15)
    active_size = len(np.union1d(source_active, target_active))
    if active_size > 1000:
        raise ValueError(f"active support {active_size} exceeds exact EMD limit 1000")
    a = a[source_active]
    b = b[target_active]
    a = a / a.sum()
    b = b / b.sum()
    M = cost_matrix[np.ix_(source_active, target_active)].astype(np.float64, copy=False)
    return float(ot.emd2(a, b, M))
