import numpy as np
import pytest
from ot_benchmark.algorithms.segmented_rescue import ordinary_state,rescue


@pytest.mark.parametrize('action',['ordinary_continue','warm_stabilized','cold_stabilized','cold_epsilon'])
def test_actions_preserve_snapshot_and_return_feasible_quality(action):
    a=np.array([.2,.8]);b=np.array([.7,.3]);c=np.array([[0.,.3],[.3,0.]])
    state=ordinary_state(a,b,c,.01,iterations=1);u=state['u'].copy();v=state['v'].copy()
    r=rescue(a,b,c,.01,state,action,block_size=100,max_blocks=10)
    assert np.array_equal(u,state['u']) and np.array_equal(v,state['v'])
    assert r['accepted'] and r['diagnostics']['numerical_gap']<=1e-8
    assert np.max(np.abs(r['plan'].sum(0)-b))<1e-12
    assert np.max(np.abs(r['plan'].sum(1)-a))<1e-12


def test_insufficient_budget_remains_failed():
    a=np.array([.2,.8]);b=np.array([.7,.3]);c=np.array([[0.,.3],[.3,0.]])
    state=ordinary_state(a,b,c,.01,iterations=1)
    r=rescue(a,b,c,.01,state,'ordinary_continue',block_size=1,max_blocks=1)
    assert not r['accepted'] and r['additional_work']==1
