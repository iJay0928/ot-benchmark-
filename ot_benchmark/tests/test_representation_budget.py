import numpy as np
import pytest
from ot_benchmark.algorithms.representation_budget import representation_budget


def test_identical_representation():
    p=np.array([[.2,.3],[.1,.4]])
    d=representation_budget(p.sum(1),p.sum(0),np.eye(2),.01,p,p)
    assert d['gap_distortion_bound']==0
    assert d['observed_gap_distortion']==0


def test_random_distortion_is_bounded():
    rng=np.random.default_rng(23001)
    for _ in range(250):
        n,m=rng.integers(2,20,size=2)
        a=rng.dirichlet(np.ones(n));b=rng.dirichlet(np.ones(m))
        p=rng.lognormal(0,4,(n,m));r=p*rng.lognormal(0,2,(n,m))
        d=representation_budget(a,b,rng.uniform(-2,3,(n,m)),.001,p,r)
        assert abs(d['primal_difference'])<=d['primal_bound']+1e-12
        assert d['observed_gap_distortion']<=d['gap_distortion_bound']+1e-10


def test_single_entry_and_mass_distortion():
    d=representation_budget([1.],[1.],[[2.]],.1,[[1.]],[[3.]])
    assert d['primal_bound']==0
    assert d['observed_gap_distortion']==pytest.approx(.2)
    assert d['gap_distortion_bound']==pytest.approx(.2)


@pytest.mark.parametrize('reg',[0,-1,np.inf,np.nan])
def test_invalid_regularization(reg):
    with pytest.raises(ValueError):
        representation_budget([1.],[1.],[[0.]],reg,[[1.]],[[1.]])
