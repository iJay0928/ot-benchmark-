import time
import numpy as np
import pytest
from core import METHODS, array_sha, diagnostics, execute, make_input, objective
import run_baselines

P=dict(max_iterations=1000000,max_outer=1000,max_inner=1000,stopThr=1e-12,
       gap_min=-1e-10,gap_max=1e-8,rounded_residual_limit=1e-12,
       timeout_seconds=30.,startup_timeout_seconds=60.)

@pytest.mark.parametrize('n',[16,64])
def test_generation(n):
    base=make_input(210001,n,0.)
    for delta in (0.,1e-5,1e-4,1e-3):
        d=make_input(210001,n,delta)
        assert np.array_equal(d['points'],base['points'])
        assert np.array_equal(d['cost'],base['cost'])
        assert np.isclose((d['b'][:n//2]-d['a'][:n//2]).sum(),delta,atol=1e-15)
        assert np.all(d['b']>0)
        assert array_sha(d)==array_sha(make_input(210001,n,delta))

@pytest.mark.parametrize('method',METHODS)
@pytest.mark.parametrize('eps',[.1,.01,.001])
def test_fields_and_independent_diagnostics(method,eps):
    d=make_input(210001,16,0.)
    result, arrays, log=execute(d,eps,method,P)
    assert result['status']=='SUCCESS',result
    raw=arrays['plan'];a,b,c=(d[k] for k in ('a','b','cost'))
    lu,lv=arrays['log_u'],arrays['log_v']
    rebuilt=np.exp(lu[:,None]+lv[None,:]-c/eps)
    np.testing.assert_allclose(raw,rebuilt,rtol=1e-9,atol=1e-12)
    rounded=diagnostics.round_plan(raw,a,b)
    entropy=sum(float(v)*(np.log(float(v))-1) for v in rounded.flat if v>0)
    independent_primal=sum(float(x*y) for x,y in zip(c.flat,rounded.flat))+eps*entropy
    independent_dual=eps*(sum(float(x*y) for x,y in zip(a,lu))+
                          sum(float(x*y) for x,y in zip(b,lv))-sum(map(float,rebuilt.flat)))
    assert independent_dual<=independent_primal+1e-10
    assert abs(independent_primal-result['native_rounded_objective'])<1e-12
    assert abs(independent_dual-result['dual'])<1e-12

@pytest.mark.parametrize('method',METHODS)
def test_nonzero_mass_development(method):
    result,_,_=execute(make_input(210001,16,.001),.1,method,P)
    assert result['status']=='SUCCESS',result

def slow_worker(connection,data,task,protocol):
    connection.send('READY');connection.recv();time.sleep(10)

def test_timeout_terminates_process(monkeypatch):
    monkeypatch.setattr(run_baselines,'worker',slow_worker)
    before=time.perf_counter()
    result,_,_=run_baselines.one({},dict(method='sinkhorn_knopp'),{**P,'timeout_seconds':.1})
    assert result['status']=='TIMEOUT'
    assert result['elapsed_seconds'] is None
    assert time.perf_counter()-before<8

def test_low_iteration_is_not_silently_success():
    result,_,_=execute(make_input(210001,16,.001),.01,'sinkhorn_stabilized',
                       {**P,'max_iterations':1})
    assert result['status']=='QUALITY_FAILED'
    assert result['numerical_gap']>P['gap_max']
