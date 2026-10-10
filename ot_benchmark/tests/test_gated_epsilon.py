import numpy as np
import pytest
from ot_benchmark.algorithms.gated_epsilon import solve_gated_epsilon

def test_gate_and_final_quality():
    r=solve_gated_epsilon([.4,.6],[.7,.3],[[0.,.3],[.3,0.]],.01,max_blocks=100,inner=100)
    assert r['accepted'] and r['reached_reg']==.01 and r['diagnostics']['numerical_gap']<=1e-8
    assert r['inner_work']<=10000
    for t in r['trace']:
        if t['action']=='cool':assert t['gap']<=1e-8 and t['raw_residual']<=1e-8
    assert np.abs(r['plan'].sum(1)-[.4,.6]).sum()<=1e-12

def test_budget_cannot_accept_intermediate_stage():
    r=solve_gated_epsilon([.5,.5],[.5,.5],[[0.,.1],[.1,0.]],.001,max_blocks=1,inner=10)
    assert not r['accepted'] and r['reached_reg']==1 and r['inner_work']<=10

@pytest.mark.parametrize('budget',[0,-1,1.5,True])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):solve_gated_epsilon([1.],[1.],[[0.]],max_blocks=budget)

def test_fixed_schedule_does_not_wait_for_gate():
    r=solve_gated_epsilon([.4,.6],[.7,.3],[[0.,1.],[1.,0.]],max_blocks=5,inner=1,fixed_hold=2)
    assert [t['reg'] for t in r['trace']]==[1.,1.,.5,.5,.25]
    assert not r['accepted'] and r['inner_work']==5

@pytest.mark.parametrize('hold',[0,-1,1.5,True])
def test_invalid_fixed_hold(hold):
    with pytest.raises(ValueError):solve_gated_epsilon([1.],[1.],[[0.]],fixed_hold=hold)

def test_best_return_preserves_trajectory_and_rejection():
    args=([.4,.6],[.7,.3],[[0.,1.],[1.,0.]])
    old=solve_gated_epsilon(*args,max_blocks=3,inner=3)
    new=solve_gated_epsilon(*args,max_blocks=3,inner=3,keep_best=True)
    assert old['trace']==new['trace'] and not new['accepted']
    q=new['returned_plan_quality'];p=new['plan'];mask=p>0
    objective=np.sum(np.asarray(args[2])*p)+.001*np.sum(p[mask]*(np.log(p[mask])-1))
    assert abs(objective-q['rounded_primal'])<1e-12
    assert q['rounded_primal']<=q['original_primal'] and q['rounded_residual']<=1e-12
    assert np.abs(p.sum(0)-args[1]).sum()<=1e-12

def test_best_return_preserves_successful_plan():
    args=([.4,.6],[.7,.3],[[0.,.3],[.3,0.]])
    old=solve_gated_epsilon(*args,.01,max_blocks=100,inner=100)
    new=solve_gated_epsilon(*args,.01,max_blocks=100,inner=100,keep_best=True)
    assert new['accepted'] and np.array_equal(old['plan'],new['plan']) and old['trace']==new['trace']
