import numpy as np
from tools.measure_branch_costs import ordinary


def test_continuation_does_not_modify_shared_snapshot():
    a=np.array([.4,.6]);b=np.array([.7,.3]);kernel=np.exp(-np.array([[0.,.2],[.2,0.]])/.01)
    state=(np.ones(2),np.ones(2));plan,info=ordinary(a,b,kernel,state,5000)
    assert info['accepted']
    assert np.all(state[0]==1) and np.all(state[1]==1)
    assert np.max(np.abs(plan.sum(axis=1)-a))<1e-9


def test_exhaustion_keeps_failed_status():
    a=np.array([.4,.6]);b=np.array([.7,.3]);kernel=np.eye(2)
    _,info=ordinary(a,b,kernel,(np.ones(2),np.ones(2)),10)
    assert not info['accepted'] and info['iterations']==10
