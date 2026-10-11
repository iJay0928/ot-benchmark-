"""Run a hash-pinned minimal source/numerical audit, not an AIGC benchmark."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

from transport_reference import grid_squared_distance, log_sinkhorn, residual
from upstream_loader import load_numerics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)


def make_inputs(seed, shape, device):
    g = torch.Generator('cpu').manual_seed(seed)
    logits = torch.randn(*shape, generator=g, dtype=torch.float64)
    a = torch.softmax(logits.flatten(), 0).to(device)
    b = torch.softmax(torch.roll(logits, shifts=4, dims=1).flatten(), 0).to(device)
    reference = torch.softmax(torch.randn(*shape, generator=g, dtype=torch.float64).flatten(), 0)
    return a, b, reference.reshape(shape).to(device)


def direction_probe(storm, seed):
    if not torch.cuda.is_available():
        return {'status': 'not_run', 'reason': 'Original circular-distribution method requires CUDA'}
    g = torch.Generator('cpu').manual_seed(seed + 1)
    # A structural input only; these are not attention maps from a model.
    maps = torch.rand(16, 16, 6, generator=g).to(device='cuda', dtype=torch.float32) * 0.02
    maps.requires_grad_(True)
    storm.device = torch.device('cuda')
    results = {}
    gradients = {}
    for name in ('left', 'right', 'above', 'below'):
        storm.prompt = 'a cat ' + name + ' a dog'
        loss, coordinate = storm._storm_loss(8, maps, [[1, 2], [None, None]],
                                            smooth_attentions=False, normalize_eot=False)
        gradient = torch.autograd.grad(loss, maps, retain_graph=False)[0]
        results[name] = {'loss': float(loss.detach()), 'coordinate': coordinate,
                         'finite_gradient': bool(torch.isfinite(gradient).all()),
                         'gradient_norm': float(gradient.norm())}
        gradients[name] = gradient.detach()
    pairs = {}
    for first, second in [('left', 'right'), ('above', 'below')]:
        pairs[first + '_' + second] = {
            'loss_abs_difference': abs(results[first]['loss'] - results[second]['loss']),
            'gradient_max_abs_difference': float((gradients[first] - gradients[second]).abs().max()),
        }
    return {'status': 'executed', 'results': results, 'opposite_pairs': pairs}


def run(args):
    protocol_path = Path(__file__).with_name('protocol_minimal.json')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory; evidence is immutable')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA unavailable; no silent CPU substitution')
    storm, source = load_numerics(Path(args.upstream) / 'pipeline.py')
    a, b, reference = make_inputs(protocol['seed'], protocol['attention_shape'], device)
    # Freeze actual arrays before any solver calls.
    dump(output / 'inputs.json', {'seed': protocol['seed'], 'shape': list(reference.shape),
                                 'a': a.cpu().tolist(), 'b': b.cpu().tolist(),
                                 'reference_attention': reference.cpu().tolist()})
    dump(output / 'protocol.json', protocol)
    row_cost = storm._compute_cost_function(reference, 'left', w=protocol['weight'])
    distance = grid_squared_distance(*reference.shape, device=device, dtype=a.dtype)
    lam = protocol['paper_mixture_lambda']
    costs = {'row_only': row_cost, 'distance_only': lam * distance,
             'mixed_p2_audit': lam * distance + (1 - lam) * row_cost}
    plans, records = {}, []
    for label, cost in costs.items():
        plan, info = log_sinkhorn(a, b, cost, reg=protocol['regularization'],
                                  max_iter=protocol['reference_iterations'],
                                  tol=protocol['reference_tolerance'])
        if not info['accepted']:
            raise RuntimeError('Reference marginal check failed: ' + label)
        plans[label] = plan.detach()
        records.append(dict(case=label, method='log_reference', dtype=str(a.dtype),
                            mass=float(plan.sum()), marginal_l1=info['marginal_l1'],
                            linear_cost=float((plan * cost).sum()), accepted=info['accepted'],
                            iterations=info['iterations']))
    for dtype in (torch.float64, torch.float32):
        aa, bb, cc = a.to(dtype), b.to(dtype), row_cost.to(dtype)
        original = storm.sinkhorn(aa, bb, cc, reg=protocol['regularization'],
                                  num_iters=protocol['original_iterations'])
        records.append(dict(case='row_only', method='original', dtype=str(dtype),
                            mass=float(original.sum()), marginal_l1=float(residual(original, aa, bb)),
                            linear_cost=float((original * cc).sum()), accepted=None,
                            iterations=protocol['original_iterations']))
    outer = a[:, None] * b[None, :]
    analytic = (a * row_cost[:, 0]).sum()
    uniform = torch.full_like(reference, 1 / reference.numel())
    vertical = storm._compute_cost_function(uniform, 'above', w=100)[:, 0].reshape(reference.shape)
    # Differentiate through both source and reference map, not only frozen C.
    logits = a.log().clone().detach().requires_grad_(True)
    ref_logits = reference.log().clone().detach().requires_grad_(True)
    aa = logits.softmax(0)
    rr = ref_logits.flatten().softmax(0).reshape(reference.shape)
    cc = storm._compute_cost_function(rr, 'left', w=100)
    pp, _ = log_sinkhorn(aa, b, cc, tol=protocol['reference_tolerance'])
    solved_value = (pp * cc).sum()
    analytic_value = (aa * cc[:, 0]).sum()
    g_solved = torch.autograd.grad(solved_value, (logits, ref_logits), retain_graph=True)
    g_analytic = torch.autograd.grad(analytic_value, (logits, ref_logits))
    with (output / 'transport_summary.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    evidence = {
        'row_cost_columns_max_abs_difference': float((row_cost - row_cost[:, :1]).abs().max()),
        'row_reference_outer_product_max_error': float((plans['row_only'] - outer).abs().max()),
        'row_reference_analytic_value_error': abs(float((plans['row_only'] * row_cost).sum() - analytic)),
        'mixed_vs_distance_plan_max_error': float((plans['mixed_p2_audit'] - plans['distance_only']).abs().max()),
        'row_vs_mixed_plan_l1_difference': float((plans['row_only'] - plans['mixed_p2_audit']).abs().sum()),
        'source_gradient_vs_analytic_max_error': float((g_solved[0] - g_analytic[0]).abs().max()),
        'reference_gradient_vs_analytic_max_error': float((g_solved[1] - g_analytic[1]).abs().max()),
        'analytic_reference_gradient_norm': float(g_analytic[1].norm()),
        'above_cost_row_variation': float((vertical[1:] - vertical[:-1]).abs().max()),
        'above_cost_column_variation': float((vertical[:, 1:] - vertical[:, :-1]).abs().max()),
        'direction_probe': direction_probe(storm, protocol['seed']),
        'transport': records,
    }
    environment = {'python': sys.version, 'torch': torch.__version__, 'platform': platform.platform(),
                   'device': str(device), 'cuda_available': torch.cuda.is_available(),
                   'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
                   'elapsed_seconds_including_input_and_audit': time.perf_counter() - started}
    paths = ['audit_storm.py', 'transport_reference.py', 'upstream_loader.py', 'protocol_minimal.json', 'model_contract.md']
    source.update({'commit': protocol['public_source_commit'],
                   'url': 'https://github.com/MICV-yonsei/STORM/blob/' + protocol['public_source_commit'] + '/pipeline.py',
                   'local_source_path': str(Path(args.upstream).resolve() / 'pipeline.py'),
                   'own_source_hashes': {name: sha(Path(__file__).with_name(name)) for name in paths},
                   'input_sha256': sha(output / 'inputs.json')})
    dump(output / 'source_manifest.json', source)
    dump(output / 'environment.json', environment)
    dump(output / 'summary.json', evidence)
    print(json.dumps({'output': str(output), 'device': str(device),
                      'original_mass': records[-1]['mass'], 'original_residual': records[-1]['marginal_l1'],
                      'opposite_direction_probe': evidence['direction_probe'].get('opposite_pairs'),
                      'reference_matches_analytic': evidence['row_reference_analytic_value_error'],
                      'elapsed_seconds': environment['elapsed_seconds_including_input_and_audit']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    run(parser.parse_args())
