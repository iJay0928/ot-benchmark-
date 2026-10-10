import numpy as np
import pytest
from ot_benchmark.algorithms.quality_policy import inspection_interval,solve_quality_policy


def test_interval_bounded_for_stagnation_and_expensive_check():
    assert inspection_interval((250,1e-3),(500,1e-3),1e-5,1e-5,1e-8)==2000
    assert inspection_interval(None,(250,1e-3),1.,1e-6,1e-8)==2000
    assert 50<=inspection_interval((250,1e-3),(500,1.001e-8),0.,1e-5,1e-8)<=2000


def test_higher_check_cost_does_not_shorten_interval():
    args=((250,1e-3),(500,1e-5))
    cheap=inspection_interval(*args,1e-5,1e-5,1e-8)
    expensive=inspection_interval(*args,.01,1e-5,1e-8)
    assert expensive>=cheap


def test_adaptive_returns_quality_checked_plan():
    a=np.array([.2,.8]);b=np.array([.7,.3]);c=np.array([[0.,.3],[.3,0.]])
    r=solve_quality_policy(a,b,c,reg=.01,policy='adaptive')
    assert r['accepted'] and r['diagnostics']['numerical_gap']<=1e-8
    assert np.max(np.abs(r['plan'].sum(0)-b))<1e-12
    assert np.max(np.abs(r['plan'].sum(1)-a))<1e-12
    assert r['inspection_trace'][-1]['accepted']
    assert all(50<=x['next_interval']<=2000 for x in r['inspection_trace'])


def test_ablations_ignore_removed_information():
    args=((250,1e-3),(500,1e-5))
    assert inspection_interval(*args,1.,1e-5,1e-8,mode='gap')==inspection_interval(*args,0.,1.,1e-8,mode='gap')
    assert inspection_interval(*args,.001,1e-5,1e-8,mode='cost')==inspection_interval(None,(500,1.),.001,1e-5,1e-8,mode='cost')


@pytest.mark.parametrize('policy',['adaptive_gap','adaptive_cost'])
def test_ablation_actual_output_quality(policy):
    r=solve_quality_policy([.2,.8],[.7,.3],[[0.,.3],[.3,0.]],reg=.01,policy=policy)
    assert r['accepted'] and r['diagnostics']['numerical_gap']<=1e-8
    assert np.max(np.abs(r['plan'].sum(0)-[.7,.3]))<1e-12
    assert np.max(np.abs(r['plan'].sum(1)-[.2,.8]))<1e-12
