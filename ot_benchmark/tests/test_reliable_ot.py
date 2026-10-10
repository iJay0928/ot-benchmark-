import numpy as np
import pytest
from ot_benchmark.algorithms.reliable_ot import solve_reliable_ot
from ot_benchmark.algorithms.sinkhorn import run_sinkhorn


def test_uncovered_kernel_falls_back_without_changing_regularization():
    a = np.array([1., 0.])
    b = np.array([0., 1.])
    cost = np.array([[0., 1.], [1., 0.]])
    result = solve_reliable_ot(a, b, cost)
    assert result.path == 'log_fallback' and result.sparse_iterations == 0
    assert result.cost == pytest.approx(1.)


def test_sparse_budget_and_dense_agreement():
    a, b = np.array([.4, .6]), np.array([.7, .3])
    cost = np.array([[0., .2], [.2, 0.]])
    result = solve_reliable_ot(a, b, cost, sparse_max_iter=2)
    assert result.path == 'log_fallback'
    assert result.sparse_iterations <= 2
    assert result.cost == pytest.approx(run_sinkhorn(a,b,cost,reg=.01), abs=1e-8)


def test_feasible_sparse_path():
    a = np.ones(4)/4
    cost = np.ones((4,4))-np.eye(4)
    result = solve_reliable_ot(a,a,cost)
    assert result.path == 'sparse'
    assert result.sparse_residual <= 1e-7
    assert result.cost == 0.


def test_covered_kernel_can_still_be_infeasible():
    # Every row/column has an edge, but diagonal components have unequal mass.
    a, b = np.array([.4, .6]), np.array([.7, .3])
    cost = np.array([[0., 1.], [1., 0.]])
    result = solve_reliable_ot(a, b, cost, sparse_max_iter=3)
    assert result.path == 'log_fallback'
    assert result.sparse_iterations == 3
    assert result.sparse_residual > 1e-7
    assert result.cost == pytest.approx(.3, abs=1e-7)


def test_zero_mass_entries_do_not_change_result():
    a, b = np.array([.4, 0., .6]), np.array([.7, 0., .3])
    cost = np.array([[0., .5, .2], [.5, 0., .5], [.2, .5, 0.]])
    result = solve_reliable_ot(a, b, cost, sparse_max_iter=0)
    reduced = solve_reliable_ot(a[[0,2]], b[[0,2]], cost[np.ix_([0,2],[0,2])], sparse_max_iter=0)
    assert result.cost == pytest.approx(reduced.cost, abs=1e-12)


def test_fallback_failure_propagates(monkeypatch):
    import ot_benchmark.algorithms.reliable_ot as module
    def fail(*args, **kwargs):
        raise ValueError('did not converge')
    monkeypatch.setattr(module, 'run_sinkhorn', fail)
    with pytest.raises(ValueError, match='did not converge'):
        solve_reliable_ot(np.ones(2),np.ones(2),np.eye(2),sparse_max_iter=0)


@pytest.mark.parametrize('source,reg', [(np.zeros(2), .01),(np.array([-1.,2.]),.01),(np.ones(2),0)])
def test_invalid_inputs_rejected(source,reg):
    with pytest.raises(ValueError):
        solve_reliable_ot(source,np.ones(2),np.eye(2),reg=reg)
