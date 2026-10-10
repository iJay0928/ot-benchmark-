import numpy as np
from tools.probe_switch_prediction import forecast,rolling_decision


def test_known_exponential_finish():
    history=[dict(iteration=i,residual=float(np.exp(-i/100))) for i in range(500,1001,50)]
    f=forecast(history)
    assert abs(f['predicted_finish']-100*np.log(1e9))<1e-6
    assert not f['switch']


def test_plateau_switch_does_not_use_later_recovery():
    history=[dict(iteration=i,residual=.01) for i in range(50,1001,50)]
    assert rolling_decision(history)==1000
    history += [dict(iteration=i,residual=1e-12) for i in range(1050,2001,50)]
    assert rolling_decision(history)==1000


def test_already_converged_does_not_switch():
    assert not forecast([dict(iteration=100,residual=1e-12)])['switch']
    assert rolling_decision([dict(iteration=100,residual=1e-12)]) is None
