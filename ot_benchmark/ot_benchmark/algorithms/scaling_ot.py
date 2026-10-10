"""Full-kernel scaling with a bounded attempt and validated log fallback.

No sparsification, new objective, or novelty claim. Row/column cost shifts
preserve a balanced transport minimizer; returned cost uses original costs.
"""
from dataclasses import dataclass
import numpy as np
from .sinkhorn import run_sinkhorn


@dataclass
class ScalingResult:
    cost: float
    path: str
    iterations: int
    residual: float | None
    fallback_reason: str
    numerical_zero_fraction: float


def solve_scaling_ot(source, target, cost_matrix, reg=.01, max_iter=200,
                     shift=True, dense_max_iter=10000):
    a, b, cost = (np.asarray(x, dtype=float) for x in (source, target, cost_matrix))
    if (a.ndim != 1 or b.ndim != 1 or cost.shape != (len(a), len(b))
            or not all(np.isfinite(x).all() for x in (a,b,cost))
            or any(np.any(x < 0) for x in (a,b,cost))
            or a.sum() <= 0 or b.sum() <= 0
            or not np.isfinite(reg) or reg <= 0
            or not isinstance(max_iter, int) or max_iter < 0
            or not isinstance(dense_max_iter, int) or dense_max_iter < 1):
        raise ValueError('invalid OT inputs or budget')
    ia, ib = np.flatnonzero(a > 0), np.flatnonzero(b > 0)
    a, b = a[ia]/a.sum(), b[ib]/b.sum()
    cost = cost[np.ix_(ia,ib)]
    shifted = cost.copy()
    if shift:
        shifted -= shifted.min(axis=1,keepdims=True)
        shifted -= shifted.min(axis=0,keepdims=True)
    with np.errstate(under='ignore', over='ignore'):
        kernel = np.exp(-shifted/reg)
    zeros = float(np.mean(kernel == 0))
    iteration, residual = 0, None
    reason = 'iteration budget exhausted'
    u = np.ones(len(a))
    for iteration in range(1,max_iter+1):
        with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
            v = b/(kernel.T@u)
            u = a/(kernel@v)
            residual = float(max(np.abs(u*(kernel@v)-a).sum(),
                                 np.abs(v*(kernel.T@u)-b).sum()))
        if not np.isfinite(residual):
            reason, residual = 'non-finite scaling', None
            break
        if residual <= 1e-9:
            plan = u[:,None]*kernel*v[None,:]
            value = float(np.sum(plan*cost))
            if np.isfinite(plan).all() and np.isfinite(value):
                return ScalingResult(value,'full_scaling',iteration,residual,'',zeros)
            reason = 'non-finite plan'
            break
    value = run_sinkhorn(a,b,cost,reg=reg,max_iter=dense_max_iter)
    return ScalingResult(value,'log_fallback',iteration,residual,reason,zeros)
