import numpy as np
import pytest
import ot
from ot_benchmark.algorithms.quality_epsilon import solve_quality_epsilon


@pytest.mark.parametrize('reg',[.01,.001])
def test_legacy_schedule_exactly_matches_installed_pot(reg):
    a=np.array([.3,.7]);b=np.array([.6,.4]);c=np.array([[.2,.8],[.7,.1]])
    raw,log=ot.bregman.sinkhorn_epsilon_scaling(a,b,c,reg,numItermax=45,numInnerItermax=1000,epsilon0=1.,stopThr=1e-22,log=True,warn=False)
    r=solve_quality_epsilon(a,b,c,reg,max_outer=45,quality_stop=False)
    assert r['outer']==log['niter']+1
    np.testing.assert_array_equal(r['raw_plan'],raw)
    np.testing.assert_array_equal(r['alpha'],log['alpha'])
    np.testing.assert_array_equal(r['beta'],log['beta'])


def test_quality_stop_uses_target_and_feasible_output():
    r=solve_quality_epsilon([.3,.7],[.6,.4],[[.2,.8],[.7,.1]],.001,max_outer=100)
    assert r['accepted'] and r['reason']=='target quality'
    assert r['diagnostics']['numerical_gap']<=1e-8
    assert all(t['target_reg']==.001 for t in r['trace'])
    assert np.abs(r['plan'].sum(1)-[.3,.7]).sum()<=1e-12


@pytest.mark.parametrize('period',[0,-1,1.2,True])
def test_bad_check_period(period):
    with pytest.raises(ValueError):solve_quality_epsilon([1.],[1.],[[0.]],check_every=period)


def test_compatible_quality_check_matches_legacy_easy_endpoint():
    args=([.3,.7],[.6,.4],[[.2,.8],[.7,.1]],.001)
    old=solve_quality_epsilon(*args,quality_stop=False)
    new=solve_quality_epsilon(*args,legacy_quality_check=True)
    assert new['accepted'] and new['outer']==old['outer']
    np.testing.assert_array_equal(new['raw_plan'],old['raw_plan'])


def test_legacy_residual_alone_cannot_accept(monkeypatch):
    import ot_benchmark.algorithms.quality_epsilon as module
    original=module.diagnose;calls=[]
    def temporarily_bad(*args):
        q,raw=original(*args);calls.append(1)
        if len(calls)==1:q['numerical_gap']=1.
        return q,raw
    monkeypatch.setattr(module,'diagnose',temporarily_bad)
    r=module.solve_quality_epsilon([.3,.7],[.6,.4],[[.2,.8],[.7,.1]],legacy_quality_check=True)
    assert not r['trace'][0]['accepted'] and r['trace'][0]['legacy_due']
    assert r['accepted'] and r['outer']>r['trace'][0]['outer']
