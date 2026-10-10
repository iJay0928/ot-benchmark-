"""Module A only: unmodified POT methods and shared external diagnostics."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
for backend in ('TENSORFLOW', 'PYTORCH', 'JAX', 'CUPY'):
    os.environ['POT_BACKEND_DISABLE_' + backend] = '1'
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import warnings
import numpy as np
import ot

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('shared_diagnostics', ROOT.parents[1] /
    'ot_benchmark/algorithms/transport_diagnostics.py')
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)
METHODS = ('sinkhorn_knopp', 'sinkhorn_stabilized', 'sinkhorn_epsilon_scaling')

def clean(value):
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    return value

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                              allow_nan=False), encoding='utf-8')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def make_input(seed, n, delta):
    rng = np.random.default_rng(seed)
    x = np.r_[rng.uniform(0, .02, n//2), rng.uniform(.98, 1, n//2)]
    a = np.ones(n)/n
    b = a.copy(); b[:n//2] += 2*delta/n; b[n//2:] -= 2*delta/n
    c = (x[:, None]-x[None, :])**2; c /= c.max()
    return dict(points=x, a=a, b=b, cost=c)

def array_sha(data):
    h = hashlib.sha256()
    for key in ('points', 'a', 'b', 'cost'):
        arr = np.asarray(data[key], dtype='<f8', order='C')
        h.update(key.encode()); h.update(str(arr.shape).encode()); h.update(arr.tobytes())
    return h.hexdigest()

def config(method, protocol):
    common = dict(numItermax=protocol['max_iterations'], stopThr=protocol['stopThr'],
                  log=True, warn=True)
    if method == 'sinkhorn_stabilized': common.update(tau=1000., print_period=20)
    if method == 'sinkhorn_epsilon_scaling':
        common.update(numItermax=protocol['max_outer'],
                      numInnerItermax=protocol['max_inner'], epsilon0=1., tau=1000.,
                      print_period=10)
    return common

def objective(p, c, eps):
    pos = p > 0
    return float(np.sum(c*p)+eps*np.sum(p[pos]*(np.log(p[pos])-1)))

def execute(data, eps, method, protocol):
    a, b, c = (data[k] for k in ('a', 'b', 'cost'))
    result = dict(status='NUMERICAL_ERROR', solve_seconds=None, audit_seconds=None,
                  elapsed_seconds=None, timed_out=False, actual_iterations=None,
                  inner_iterations=None, error=None)
    log = {}; p = lu = lv = None
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        try:
            p, log = getattr(ot.bregman, method)(a, b, c, eps, **config(method, protocol))
            solved = time.perf_counter(); result['solve_seconds'] = solved-started
            if not np.isfinite(p).all() or np.any(p < 0):
                raise ValueError('invalid native plan')
            if method == 'sinkhorn_knopp':
                if np.any(log['u'] <= 0) or np.any(log['v'] <= 0):
                    raise ValueError('non-positive native scaling; no clipping allowed')
                lu, lv = np.log(log['u']), np.log(log['v'])
            elif method == 'sinkhorn_stabilized': lu, lv = log['logu'], log['logv']
            else: lu, lv = log['alpha']/eps, log['beta']/eps
            if not np.isfinite(lu).all() or not np.isfinite(lv).all():
                raise ValueError('non-finite native potentials')
            diag, rebuilt = diagnostics.diagnose(a, b, c, eps, lu, lv)
            result.update(diag)
            result['reconstruction_max_abs'] = float(np.max(np.abs(p-rebuilt)))
            result['reconstruction_l1'] = float(np.abs(p-rebuilt).sum())
            if not np.allclose(p, rebuilt, rtol=1e-9, atol=1e-12):
                result['status'] = 'DIAGNOSTIC_INVALID'
                raise ValueError('native plan and final-epsilon reconstructed plan differ')
            result['raw_objective'] = objective(p, c, eps)
            result['native_raw_residual'] = float(max(np.abs(p.sum(1)-a).sum(),
                                                      np.abs(p.sum(0)-b).sum()))
            # Recheck the native returned plan, not only the reconstructed matrix.
            rounded_native = diagnostics.round_plan(p, a, b)
            result['native_rounded_objective'] = objective(rounded_native, c, eps)
            result['native_numerical_gap'] = result['native_rounded_objective']-diag['dual']
            ok = (diag['rounded_residual'] <= protocol['rounded_residual_limit'] and
                  protocol['gap_min'] <= diag['numerical_gap'] <= protocol['gap_max'] and
                  protocol['gap_min'] <= result['native_numerical_gap'] <= protocol['gap_max'])
            result['status'] = 'SUCCESS' if ok else 'QUALITY_FAILED'
        except Exception as exc:
            result['error'] = type(exc).__name__ + ': ' + str(exc)
        finally:
            ended = time.perf_counter()
            result['elapsed_seconds'] = ended-started
            if result['solve_seconds'] is not None:
                result['audit_seconds'] = ended-started-result['solve_seconds']
            result['warnings'] = [dict(category=type(w.message).__name__, message=str(w.message))
                                  for w in captured]
    index = log.get('niter', log.get('n_iter'))
    result['actual_iterations'] = int(index)+1 if index is not None else None
    if method == 'sinkhorn_knopp': result['inner_iterations'] = result['actual_iterations']
    if method == 'sinkhorn_stabilized': result['inner_iterations'] = result['actual_iterations']
    result['native_last_error'] = float(log['err'][-1]) if log.get('err') else None
    result['native_stop_met'] = (result['native_last_error'] <= protocol['stopThr']
                                if result['native_last_error'] is not None else None)
    result['config'] = config(method, protocol)
    return clean(result), dict(plan=p, log_u=lu, log_v=lv), clean(log)
