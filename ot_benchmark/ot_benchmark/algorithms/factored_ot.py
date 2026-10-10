"""Factored Optimal Transport — group-structured decomposition.

For data with known group structure (e.g., block-sparse patterns),
decomposes the OT problem: first transport between groups, then within.
"""

import numpy as np
import ot


def _partition_into_groups(
    x: np.ndarray, n_groups: int = 5
) -> list[np.ndarray]:
    """Partition a distribution into equally-sized contiguous groups."""
    n = len(x)
    group_size = n // n_groups
    groups = []
    for i in range(n_groups):
        start = i * group_size
        end = start + group_size if i < n_groups - 1 else n
        groups.append(x[start:end])
    return groups


def run_factored_ot(
    source: np.ndarray,
    target: np.ndarray,
    cost_matrix: np.ndarray,
    n_groups: int = 5,
    reg: float = 0.1,
    reg_between: float = 0.05,
) -> float:
    """Compute Factored OT distance by decomposing into group-level + within-group OT.

    1. Aggregate mass per group from source and target.
    2. Solve a small OT problem between groups.
    3. Distribute within each group using standard Sinkhorn.

    Args:
        source: Source distribution.
        target: Target distribution.
        cost_matrix: Full pairwise cost matrix.
        n_groups: Number of groups to partition into.
        reg: Regularization for within-group Sinkhorn.
        reg_between: Regularization for between-group Sinkhorn.
    """
    n = len(source)
    source_groups = _partition_into_groups(source, n_groups)
    target_groups = _partition_into_groups(target, n_groups)

    group_mass_source = np.array([g.sum() for g in source_groups])
    group_mass_target = np.array([g.sum() for g in target_groups])
    group_mass_source = group_mass_source / group_mass_source.sum()
    group_mass_target = group_mass_target / group_mass_target.sum()

    group_centers = np.array([
        np.average(np.arange(len(g)), weights=g + 1e-12) + i * (n // n_groups)
        for i, g in enumerate(source_groups)
    ])
    group_target_centers = np.array([
        np.average(np.arange(len(g)), weights=g + 1e-12) + i * (n // n_groups)
        for i, g in enumerate(target_groups)
    ])
    M_between = np.abs(group_centers[:, None] - group_target_centers[None, :])
    M_between = M_between / (M_between.max() + 1e-10)

    try:
        source_active = np.flatnonzero(group_mass_source > 1e-15)
        target_active = np.flatnonzero(group_mass_target > 1e-15)
        between_distance = float(ot.sinkhorn2(
            group_mass_source[source_active], group_mass_target[target_active],
            M_between[np.ix_(source_active, target_active)], reg_between,
            method="sinkhorn_log",
        ))
    except Exception:
        between_distance = 0.0

    within_distance = 0.0
    for i, (sg, tg) in enumerate(zip(source_groups, target_groups)):
        if sg.sum() <= 1e-15 or tg.sum() <= 1e-15:
            continue
        sg_norm = sg / (sg.sum() + 1e-12)
        tg_norm = tg / (tg.sum() + 1e-12)
        local_size = len(sg)
        M_local = np.abs(
            np.arange(local_size)[:, None].astype(np.float64)
            - np.arange(local_size)[None, :].astype(np.float64)
        )
        M_local = M_local / (M_local.max() + 1e-10)
        try:
            source_local = np.flatnonzero(sg_norm > 1e-15)
            target_local = np.flatnonzero(tg_norm > 1e-15)
            d = float(ot.sinkhorn2(
                sg_norm[source_local], tg_norm[target_local],
                M_local[np.ix_(source_local, target_local)], reg,
                method="sinkhorn_log",
            ))
            within_distance += d * group_mass_source[i]
        except Exception:
            within_distance += float(np.sum(np.abs(sg_norm - tg_norm)))

    return between_distance + within_distance
