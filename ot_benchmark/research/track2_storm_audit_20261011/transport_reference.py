"""Differentiable positive-mass log-domain EOT reference for source auditing."""
import torch


def validate(a, b, cost, reg):
    if a.ndim != 1 or b.ndim != 1 or cost.shape != (a.numel(), b.numel()):
        raise ValueError('Shape mismatch')
    if not reg > 0 or not all(torch.isfinite(x).all().item() for x in (a, b, cost)):
        raise ValueError('Invalid regularization or non-finite input')
    if not (a > 0).all().item() or not (b > 0).all().item():
        raise ValueError('This minimal reference requires strictly positive masses')
    if not all(abs(float(x.sum().detach()) - 1) < 1e-8 for x in (a, b)):
        raise ValueError('Each marginal must have unit mass')
    if not (a.dtype == b.dtype == cost.dtype and a.device == b.device == cost.device):
        raise ValueError('Dtype/device mismatch')


def residual(plan, a, b):
    return (plan.sum(1) - a).abs().sum() + (plan.sum(0) - b).abs().sum()


def log_sinkhorn(a, b, cost, reg=0.1, max_iter=2000, tol=1e-10):
    validate(a, b, cost, reg)
    if max_iter < 1 or tol <= 0:
        raise ValueError('Invalid stopping parameters')
    kernel = -cost / reg
    logu, logv = torch.zeros_like(a), torch.zeros_like(b)
    loga, logb = a.log(), b.log()
    for iteration in range(1, max_iter + 1):
        logu = loga - torch.logsumexp(kernel + logv[None, :], 1)
        logv = logb - torch.logsumexp(kernel + logu[:, None], 0)
        if iteration == 1 or iteration % 10 == 0 or iteration == max_iter:
            plan = (logu[:, None] + kernel + logv[None, :]).exp()
            error = residual(plan, a, b)
            if float(error.detach()) <= tol:
                break
    plan = (logu[:, None] + kernel + logv[None, :]).exp()
    error = float(residual(plan, a, b).detach())
    return plan, {'iterations': iteration, 'marginal_l1': error,
                  'accepted': error <= tol, 'criterion': 'marginal feasibility only, not dual gap'}


def grid_squared_distance(height, width, *, device, dtype):
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                          torch.arange(width, device=device, dtype=dtype), indexing='ij')
    coords = torch.stack((y.flatten(), x.flatten()), 1)
    return (coords[:, None, :] - coords[None, :, :]).square().sum(-1)
