"""Sliced Wasserstein distance via random projections."""

import numpy as np
import ot


def run_sliced_ot(
    source: np.ndarray,
    target: np.ndarray,
    cost_matrix: np.ndarray,  # kept for interface consistency, not used
    n_projections: int = 200,
    seed: int = 42,
    support_points: np.ndarray | None = None,
) -> float:
    """Compute Sliced Wasserstein distance via 1-D projections.

    Projects high-dimensional distributions onto random directions
    and computes 1-D Wasserstein distances via sorting.

    Args:
        source: Source distribution.
        target: Target distribution.
        cost_matrix: Ignored (sliced OT uses projection-based cost).
        n_projections: Number of random projection directions.
        seed: Random seed for reproducibility.
    """
    n = len(source)
    if support_points is None:
        points = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    else:
        points = np.asarray(support_points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(-1, 1)
        if points.shape[0] != n:
            raise ValueError("support_points must align with distribution weights")

    a = source.astype(np.float64).copy()
    b = target.astype(np.float64).copy()
    a = a / a.sum()
    b = b / b.sum()
    source_active = np.flatnonzero(a > 1e-15)
    target_active = np.flatnonzero(b > 1e-15)
    source_points = points[source_active]
    target_points = points[target_active]
    a = a[source_active]
    b = b[target_active]
    a = a / a.sum()
    b = b / b.sum()

    try:
        distance = ot.sliced_wasserstein_distance(
            source_points, target_points, a, b, n_projections, seed=seed
        )
        return float(distance ** 2)
    except (TypeError, ValueError, MemoryError):
        rng = np.random.RandomState(seed)
        distances = []
        proj = rng.randn(points.shape[1], n_projections)
        for p in range(n_projections):
            projected_source = source_points @ proj[:, p]
            projected_target = target_points @ proj[:, p]
            d = ot.wasserstein_1d(
                projected_source, projected_target, a, b, p=2,
            )
            distances.append(float(d))
        return float(np.mean(distances))
