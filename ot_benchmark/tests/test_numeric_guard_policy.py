import numpy as np
import pytest
from ot_benchmark.algorithms.numeric_guard_policy import solve_guard_policy


@pytest.mark.parametrize('policy',['guarded','guarded_250','guarded_cold_250','warm_budget','cold_budget'])
def test_complete_policy_rescue_returns_correct_full_matrix(policy):
    a=np.array([.2,0,.8]);b=np.array([.7,0,.3]);c=np.array([[0.,.2,.3],[.2,0.,.2],[.3,.2,0.]])
    r=solve_guard_policy(a,b,c,.01,policy=policy,ordinary_budget=1);p=r['plan']
    assert r['accepted'] and r['used_rescue']
    assert np.max(np.abs(p.sum(0)-b))<1e-12 and np.max(np.abs(p.sum(1)-a))<1e-12
    pos=p>0;f=np.sum(c*p)+.01*np.sum(p[pos]*(np.log(p[pos])-1))
    assert abs(f-r['diagnostics']['rounded_primal'])<1e-12


def test_nonfinite_update_restores_finite_warm_start():
    r=solve_guard_policy([.4,.6],[.7,.3],[[1.,2.],[2.,1.]],.001)
    assert r['reason']=='nonfinite ordinary update' and r['accepted']


def test_zero_kernel_alone_does_not_force_switch():
    r=solve_guard_policy([.5,.5],[.5,.5],[[0.,1.],[1.,0.]],.001)
    assert r['accepted'] and not r['used_rescue']


def test_short_guard_keeps_trigger_and_checks_quality_in_short_blocks():
    a=[.4,.6];b=[.7,.3];c=[[0.,.9],[.9,0.]]
    old=solve_guard_policy(a,b,c,.001,discrepancy_threshold=1e-30)
    new=solve_guard_policy(a,b,c,.001,policy='guarded_250',discrepancy_threshold=1e-30)
    assert old['inspection_trace']==new['inspection_trace']
    assert new['accepted'] and new['stages'][0]['action']=='warm_stabilized'
    assert new['stages'][0]['work']<=200000
    assert all(h['executed']<=250 for h in new['stages'][0]['history'])
    assert new['diagnostics']['numerical_gap']<=1e-8


def test_cold_ablation_keeps_trigger_and_matches_zero_start_stage():
    args=([.4,.6],[.7,.3],[[0.,.9],[.9,0.]],.001)
    warm=solve_guard_policy(*args,policy='guarded_250',discrepancy_threshold=1e-30)
    cold=solve_guard_policy(*args,policy='guarded_cold_250',discrepancy_threshold=1e-30)
    direct=solve_guard_policy(*args,policy='direct_stabilized_250')
    assert warm['inspection_trace']==cold['inspection_trace']
    assert cold['accepted'] and cold['stages'][0]['action']=='cold_stabilized'
    assert cold['stages'][0]['history']==direct['stages'][0]['history']
    assert np.array_equal(cold['plan'],direct['plan'])


def test_fixed_switch_respects_time_and_common_quality():
    r=solve_guard_policy([.4,.6],[.7,.3],[[0.,.9],[.9,0.]],.001,policy='fixed_warm_250',switch_iteration=7)
    assert r['ordinary_iterations']==7 and r['reason']=='fixed switch'
    assert r['accepted'] and r['stages'][0]['action']=='warm_stabilized'
    assert all(h['executed']<=250 for h in r['stages'][0]['history'])


def test_quality_acceptance_precedes_fixed_switch():
    r=solve_guard_policy([.5,.5],[.5,.5],[[0.,1.],[1.,0.]],.001,policy='fixed_warm_250',switch_iteration=1)
    assert r['accepted'] and not r['used_rescue']


@pytest.mark.parametrize('point',[0,-1,1.5,True])
def test_invalid_fixed_point(point):
    with pytest.raises(ValueError):
        solve_guard_policy([1.],[1.],[[0.]],switch_iteration=point)


def test_quality_fallback_after_stage_error_preserves_full_shape(monkeypatch):
    import ot_benchmark.algorithms.numeric_guard_policy as module
    def fail(*args,**kwargs):raise FloatingPointError('injected first-stage failure')
    monkeypatch.setattr(module,'rescue',fail)
    a=np.array([.4,0,.6]);b=np.array([.7,0,.3]);c=np.array([[0.,.2,.9],[.2,0.,.2],[.9,.2,0.]])
    r=module.solve_guard_policy(a,b,c,.001,policy='guarded_250',ordinary_budget=1,epsilon_stop='quality')
    assert r['accepted'] and r['used_epsilon']
    assert r['stages'][0]['error']=='injected first-stage failure'
    assert r['stages'][1]['quality_trace'][-1]['accepted']
    assert np.abs(r['plan'].sum(1)-a).sum()<=1e-12
    assert np.abs(r['plan'].sum(0)-b).sum()<=1e-12


def test_unused_quality_fallback_does_not_change_output():
    args=([.5,.5],[.5,.5],[[0.,1.],[1.,0.]],.001)
    a=solve_guard_policy(*args,policy='guarded_250')
    b=solve_guard_policy(*args,policy='guarded_250',epsilon_stop='quality')
    assert not b['used_epsilon'] and a['inspection_trace']==b['inspection_trace']
    np.testing.assert_array_equal(a['plan'],b['plan'])


@pytest.mark.parametrize('policy',['direct_stabilized','direct_stabilized_250'])
def test_direct_baseline_skips_ordinary_prefix(policy):
    a=np.array([.2,.8]);b=np.array([.7,.3]);c=np.array([[0.,.3],[.3,0.]])
    r=solve_guard_policy(a,b,c,.01,policy=policy)
    assert r['accepted'] and r['ordinary_iterations']==0
    assert r['stages'][0]['action']=='cold_stabilized'
    assert np.max(np.abs(r['plan'].sum(0)-b))<1e-12
    assert np.max(np.abs(r['plan'].sum(1)-a))<1e-12


@pytest.mark.parametrize('threshold',[0.,-1.,float('nan'),float('inf')])
def test_invalid_discrepancy_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        solve_guard_policy([1.],[1.],[[0.]],discrepancy_threshold=threshold)


@pytest.mark.parametrize('threshold',[1e-10,1e-8,1e-6])
def test_threshold_does_not_relax_quality_gate(threshold):
    r=solve_guard_policy([.2,.8],[.7,.3],[[0.,.3],[.3,0.]],.01,discrepancy_threshold=threshold)
    assert r['accepted'] and r['diagnostics']['numerical_gap']<=1e-8
