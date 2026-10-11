import pytest
import torch
from transport_reference import grid_squared_distance, log_sinkhorn, residual


def data():
    a = torch.tensor([0.4, 0.6], dtype=torch.float64)
    b = torch.tensor([0.7, 0.3], dtype=torch.float64)
    cost = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64)
    return a, b, cost


def test_feasible_finite():
    a, b, cost = data()
    plan, info = log_sinkhorn(a, b, cost)
    assert info['accepted'] and torch.isfinite(plan).all()
    assert float(residual(plan, a, b)) < 1e-10


def test_row_only_analytic():
    a, b, _ = data()
    c = torch.tensor([2., 40.], dtype=torch.float64)
    plan, _ = log_sinkhorn(a, b, c[:, None].repeat(1, 2))
    assert torch.allclose(plan, a[:, None] * b[None, :], atol=1e-12, rtol=0)
    assert abs(float((plan * c[:, None]).sum() - (a * c).sum())) < 1e-10


def test_row_and_column_potential_invariance():
    a, b, cost = data()
    first, _ = log_sinkhorn(a, b, cost)
    second, _ = log_sinkhorn(a, b, cost + torch.tensor([4., 8.])[:, None]
                              + torch.tensor([2., 3.])[None, :])
    assert torch.allclose(first, second, atol=1e-10, rtol=0)


def test_target_matters_with_pair_cost():
    a, b, cost = data()
    first, _ = log_sinkhorn(a, b, cost)
    second, _ = log_sinkhorn(a, b.flip(0), cost)
    assert abs(float((first * cost).sum() - (second * cost).sum())) > 0.1


def test_row_gradient_not_zero():
    logits = torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True)
    a = logits.softmax(0)
    b = torch.tensor([0.7, 0.3], dtype=torch.float64)
    c = torch.tensor([2., 40.], dtype=torch.float64)
    plan, _ = log_sinkhorn(a, b, c[:, None].repeat(1, 2))
    grad = torch.autograd.grad((plan * c[:, None]).sum(), logits)[0]
    assert torch.isfinite(grad).all() and float(grad.norm()) > 0.1


def test_unconverged_is_not_accepted():
    a, b, cost = data()
    _, info = log_sinkhorn(a, b, cost, max_iter=1)
    assert not info['accepted']


@pytest.mark.parametrize('fault', ['negative', 'zero', 'mass', 'shape', 'nan', 'reg'])
def test_invalid_contract_rejected(fault):
    a, b, cost = data()
    reg = 0.1
    if fault == 'negative': a[0] = -1
    if fault == 'zero': a[0] = 0
    if fault == 'mass': a = a * 2
    if fault == 'shape': cost = cost[:1]
    if fault == 'nan': cost[0, 0] = float('nan')
    if fault == 'reg': reg = 0
    with pytest.raises(ValueError): log_sinkhorn(a, b, cost, reg=reg)


def test_grid_uses_two_axes():
    cost = grid_squared_distance(2, 3, device='cpu', dtype=torch.float64)
    assert cost.shape == (6, 6)
    assert cost[0, 3].item() == 1 and cost[0, 2].item() == 4
    assert torch.equal(cost, cost.T)
