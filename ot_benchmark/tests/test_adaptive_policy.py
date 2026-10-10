import numpy as np
from ot_benchmark.algorithms.adaptive_policy import solve_policy


def test_easy_case_avoids_fallback():
    for policy in ['budget','rolling']:
        result=solve_policy(np.ones(2),np.ones(2),np.eye(2),policy=policy)
        assert result['accepted'] and not result['used_epsilon']


def test_failed_fallback_not_marked_success(monkeypatch):
    import ot_benchmark.algorithms.adaptive_policy as module
    calls=[]
    def fake(a,b,cost,reg,**kwargs):
        calls.append(kwargs)
        return np.zeros_like(cost),{'niter':0}
    monkeypatch.setattr(module.ot.bregman,'sinkhorn_epsilon_scaling',fake)
    r=solve_policy(np.array([.4,.6]),np.array([.7,.3]),np.array([[0.,1.],[1.,0.]]),ordinary_budget=1)
    assert not r['accepted'] and r['used_epsilon'] and r['ordinary_iterations']==1 and len(calls)==1


def test_always_policy_skips_ordinary(monkeypatch):
    import ot_benchmark.algorithms.adaptive_policy as module
    def fake(a,b,cost,reg,**kwargs):return np.outer(a,b),{'niter':0}
    monkeypatch.setattr(module.ot.bregman,'sinkhorn_epsilon_scaling',fake)
    r=solve_policy(np.ones(2),np.ones(2),np.zeros((2,2)),policy='always')
    assert r['accepted'] and r['ordinary_iterations']==0
