import numpy as np
import pytest
from ot_benchmark.algorithms.continuation_ot import solve_continuation_ot


def test_modes_agree_and_warm_preserves_progress():
    a=np.array([.4,.6]);b=np.array([.7,.3]);c=np.array([[0.,.2],[.2,0.]])
    results={mode:solve_continuation_ot(a,b,c,mode=mode,switch_iter=10) for mode in ['full','log','cold','warm']}
    for r in results.values():
        assert r.residual<=1e-9
        assert r.cost==pytest.approx(results['log'].cost,abs=1e-9)
    assert results['warm'].warm_started
    assert results['warm'].iterations<=results['cold'].iterations


def test_underflow_switch_uses_original_cost():
    a=np.ones(2);c=np.full((2,2),100.)
    with pytest.raises(ValueError,match='non-finite'):solve_continuation_ot(a,a,c,mode='full')
    for mode in ['cold','warm']:
        r=solve_continuation_ot(a,a,c,mode=mode)
        assert r.switched and r.cost==pytest.approx(100.)


def test_shared_budget_is_not_reset():
    with pytest.raises(ValueError,match='total iteration budget'):
        solve_continuation_ot(np.array([.4,.6]),np.array([.7,.3]),np.eye(2),mode='warm',switch_iter=2,max_iter=3)


def test_zero_mass_support():
    r=solve_continuation_ot(np.array([1.,0.]),np.array([0.,1.]),np.array([[0.,1.],[1.,0.]]))
    assert r.cost==pytest.approx(1.)
