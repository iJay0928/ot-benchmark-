import numpy as np
import pytest
from ot_benchmark.algorithms.scaling_ot import solve_scaling_ot
from ot_benchmark.algorithms.sinkhorn import run_sinkhorn


def test_shift_preserves_original_cost_even_with_large_offsets():
    a=np.array([.4,.6]); b=np.array([.7,.3])
    cost=np.array([[100.,100.02],[101.03,101.]])
    result=solve_scaling_ot(a,b,cost,reg=.1)
    assert result.path=='full_scaling'
    assert result.residual<=1e-9
    assert result.cost==pytest.approx(run_sinkhorn(a,b,cost,reg=.1),abs=1e-7)
    assert result.cost>100


def test_unshifted_underflow_falls_back():
    result=solve_scaling_ot(np.ones(2),np.ones(2),np.full((2,2),100.),shift=False)
    assert result.path=='log_fallback'
    assert result.numerical_zero_fraction==1
    assert result.cost==pytest.approx(100.)


def test_budget_fallback_and_zero_mass():
    a=np.array([1.,0.]); b=np.array([0.,1.])
    result=solve_scaling_ot(a,b,np.array([[0.,1.],[1.,0.]]),max_iter=0)
    assert result.path=='log_fallback' and result.iterations==0
    assert result.cost==pytest.approx(1.)


def test_fallback_error_not_suppressed(monkeypatch):
    import ot_benchmark.algorithms.scaling_ot as module
    def fail(*args,**kwargs):
        raise ValueError('failed validation')
    monkeypatch.setattr(module,'run_sinkhorn',fail)
    with pytest.raises(ValueError,match='failed validation'):
        solve_scaling_ot(np.ones(2),np.ones(2),np.eye(2),max_iter=0)
