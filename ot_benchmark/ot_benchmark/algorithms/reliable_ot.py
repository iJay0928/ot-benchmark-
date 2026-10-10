"""Bounded sparse attempt followed by a validated log-domain fallback.

This is an engineering policy, not a new OT objective or an optimality claim.
It requires an already constructed dense cost matrix.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from .sinkhorn import run_sinkhorn


@dataclass
class ReliableOTResult:
    cost: float
    path: str
    sparse_iterations: int
    sparse_residual: float | None
    fallback_reason: str
    kernel_density: float


def solve_reliable_ot(source, target, cost_matrix, reg=.01,
                      sparse_max_iter=200, dense_max_iter=10000,
                      kernel_tolerance=1e-8, feasibility_tolerance=1e-7):
    """Accept only a feasible sparse plan; otherwise solve the full kernel.

    Iteration budgets are deterministic, not hard wall-clock timeouts.
    Exceptions from the fallback propagate and must be counted as failures.
    """
    a, b, cost = (np.asarray(x, dtype=float) for x in (source, target, cost_matrix))
    if (a.ndim != 1 or b.ndim != 1 or cost.shape != (len(a), len(b))
            or not all(np.isfinite(x).all() for x in (a, b, cost))
            or any(np.any(x < 0) for x in (a, b, cost))
            or a.sum() <= 0 or b.sum() <= 0):
        raise ValueError('invalid OT inputs')
    if (not np.isfinite(reg) or reg <= 0 or sparse_max_iter < 0 or dense_max_iter < 1
            or not 0 < kernel_tolerance < 1 or not 0 < feasibility_tolerance <= 1e-7):
        raise ValueError('invalid solver budget or tolerance')
    ia, ib = np.flatnonzero(a > 0), np.flatnonzero(b > 0)
    a, b = a[ia]/a.sum(), b[ib]/b.sum()
    cost = cost[np.ix_(ia, ib)]
    rows, cols = np.where(cost <= -reg*np.log(kernel_tolerance))
    values = np.exp(-cost[rows, cols]/reg)
    kernel = csr_matrix((values, (rows, cols)), shape=cost.shape)
    density = float(kernel.nnz / cost.size)
    reason = 'sparse iteration budget exhausted'
    residual = None
    iteration = 0
    if np.any(np.diff(kernel.indptr) == 0) or np.any(np.asarray(kernel.sum(axis=0)).ravel() == 0):
        reason = 'truncated kernel has uncovered positive-mass row or column'
    else:
        transpose = kernel.T.tocsr()
        u = np.ones(len(a))
        for iteration in range(1, sparse_max_iter + 1):
            with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                v = b / (transpose @ u)
                u = a / (kernel @ v)
                residual = float(max(np.abs(u*(kernel@v)-a).sum(),
                                     np.abs(v*(transpose@u)-b).sum()))
            if not np.isfinite(residual):
                reason = 'non-finite sparse scaling'
                residual = None
                break
            if residual <= feasibility_tolerance:
                plan_values = u[rows]*values*v[cols]
                value = float(np.dot(plan_values, cost[rows, cols]))
                if np.isfinite(plan_values).all() and np.isfinite(value):
                    return ReliableOTResult(value, 'sparse', iteration, residual, '', density)
                reason = 'non-finite sparse plan'
                break
    value = run_sinkhorn(a, b, cost, reg=reg, max_iter=dense_max_iter, stop_threshold=1e-9)
    return ReliableOTResult(value, 'log_fallback', iteration, residual, reason, density)
