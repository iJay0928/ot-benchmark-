"""Offline objective-unit representation diagnostics; not a stopping certificate.

Entropy continuity uses Audenaert, arXiv:quant-ph/0610146, Theorem 1,
in natural logarithms. Inputs are normalized here to make the probability
assumption explicit; the measured normalization correction is also reported.
"""
import numpy as np
from .transport_diagnostics import round_plan


def representation_budget(a, b, cost, reg, direct, reconstructed):
    cost, direct, reconstructed = [np.asarray(x, dtype=float) for x in (cost, direct, reconstructed)]
    if (not np.isfinite(reg) or reg <= 0 or cost.shape != direct.shape
        or cost.shape != reconstructed.shape or cost.size == 0
        or any(not np.isfinite(x).all() for x in (cost, direct, reconstructed))
        or np.any(direct < 0) or np.any(reconstructed < 0)):
        raise ValueError('invalid representation inputs')
    q, r = round_plan(direct, a, b), round_plan(reconstructed, a, b)
    def objective(p):
        positive = p > 0
        return float(np.sum(cost*p)+reg*np.sum(p[positive]*(np.log(p[positive])-1)))
    unnormalized_difference = objective(r)-objective(q)
    q, r = q/q.sum(), r/r.sum()
    difference = objective(r)-objective(q)
    correction = abs(unnormalized_difference-difference)
    tv = float(np.clip(np.abs(r-q).sum()/2, 0, 1))
    binary_entropy = -sum(x*np.log(x) for x in (tv, 1-tv) if x > 0)
    entropy_bound = tv*np.log(cost.size-1)+binary_entropy if cost.size > 1 else 0.
    primal_bound = float(np.ptp(cost)*tv+reg*entropy_bound+correction)
    # D_R uses the true exponential reconstruction. D_P is only a surrogate.
    dual_difference = float(reg*(reconstructed.sum()-direct.sum()))
    dual_bound = float(reg*np.abs(reconstructed-direct).sum())
    return dict(total_variation=tv, normalization_correction=correction,
                primal_difference=unnormalized_difference, primal_bound=primal_bound,
                dual_mass_difference=dual_difference, dual_bound=dual_bound,
                observed_gap_distortion=abs(unnormalized_difference+dual_difference),
                gap_distortion_bound=primal_bound+dual_bound)
