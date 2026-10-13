"""One bounded correctness pass before either formal continuation."""
from __future__ import annotations

import gc
import time

import numpy as np
import torch
from torch.nn import functional as F

from .common import *
from .loss import mixed_loss
from .teacher import raw_training, build, blocks


def cpu_checks():
    path = HERE / 'cpu_contracts.json'
    if path.exists():
        previous = read(path)
        if previous['execution_code'] != execution_hashes():
            raise RuntimeError('CPU contracts refer to different execution code')
        return previous
    torch.set_num_threads(2)
    z = torch.tensor([-4., -.3, .8, 5.], requires_grad=True)
    y = torch.tensor([0., 1., 0., 1.])
    q = torch.tensor([.2, .7, .5, .9], requires_grad=True)
    a = torch.tensor([.1, .4, .8, 1.])
    j = torch.tensor([3, 2, 0, 1])
    m = torch.tensor([0., 1., 1., 0.])
    value, hard, soft = mixed_loss(z, y, j, a, q, m, 1.)
    value.backward()
    expected = ((z.detach().sigmoid()-a*y-(1-a)*y[j])
                + a*m*(z.detach().sigmoid()-q.detach())
                + (1-a)*m[j]*(z.detach().sigmoid()-q.detach()[j]))/len(z)
    torch.testing.assert_close(z.grad, expected, rtol=1e-5, atol=1e-7)
    if q.grad is not None:
        raise RuntimeError('Teacher target received gradients')
    plain, _, _ = mixed_loss(z.detach(), y, j, a, q, m, 0.)
    reference = (a*F.binary_cross_entropy_with_logits(z.detach(), y, reduction='none')
                 + (1-a)*F.binary_cross_entropy_with_logits(z.detach(), y[j], reduction='none')).mean()
    torch.testing.assert_close(plain, reference, rtol=0, atol=0)
    no_teacher, _, zero_soft = mixed_loss(z.detach(), y, j, a, q, torch.zeros_like(m), 1.)
    torch.testing.assert_close(no_teacher, reference, rtol=0, atol=0)
    if float(zero_soft) != 0:
        raise RuntimeError('Unavailable teachers contributed loss')
    extreme = torch.tensor([-1000., 1000., -100., 100.], requires_grad=True)
    v, _, _ = mixed_loss(extreme, y, j, a, torch.tensor([0., 1., 1., 0.]), m, 1.)
    v.backward()
    if not torch.isfinite(v) or not torch.isfinite(extreme.grad).all():
        raise RuntimeError('Extreme logits produce nonfinite loss or gradient')
    parts, schema = load_inputs()
    tr, va = parts['train'], parts['validation']
    with np.load(PARENT / 'predictions_validation.npz') as saved:
        np.testing.assert_array_equal(saved['ids'], va['ids'])
        np.testing.assert_array_equal(saved['y'], va['y'])
    chronological_blocks = blocks(tr['dt'])
    if abs(sum(schema['excel']['mi_weights'])-1.) > 1e-6:
        raise RuntimeError('MI weights are not a probability partition')
    raw = raw_training()
    prefix = raw.iloc[:2048].copy()
    future = raw.iloc[2048:3072].copy()
    first, _, _ = build(prefix.copy(), future.copy(), 'train_only')
    perturbed = future.copy()
    perturbed['TransactionAmt'] = perturbed['TransactionAmt'] + np.float32(100000.)
    second, _, _ = build(prefix.copy(), perturbed, 'train_only')
    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())
    refused = False
    try:
        build(prefix.assign(isFraud=0), future.assign(isFraud=0), 'train_only')
    except ValueError:
        refused = True
    if not refused:
        raise RuntimeError('Teacher feature builder accepted a label column')
    del raw, prefix, future, first, second, parts
    gc.collect()
    result = {'status': 'PASS', 'analytic_logit_gradient': 'PASS', 'teacher_gradients_detached': True,
              'lambda_zero_matches_parent_hard_loss': True, 'unavailable_teachers_contribute_zero': True,
              'extreme_logits_finite': True, 'future_covariate_perturbation_invariance': True,
              'label_columns_rejected': True, 'train_validation_alignment': True,
              'teacher_blocks': chronological_blocks, 'execution_code': execution_hashes(), 'test_accessed': False, 'finished_utc': now()}
    write(path, result)
    print('CPU_CONTRACTS_PASS', flush=True)
    return result


def cuda_smoke(budget, resource):
    from ..identity_uid_transformer_transfer_20261011 import models
    from .training import update
    path = HERE / 'smoke.json'
    if path.exists():
        receipt = read(path)
        if receipt['execution_code'] != execution_hashes():
            raise RuntimeError('Smoke executable version mismatch')
        return receipt
    parts, schema = load_inputs()
    va, tr = parts['validation'], parts['train']
    cfg = freeze()['student_config']
    models.seed_all(42)
    torch.cuda.reset_peak_memory_stats()
    model = models.build('excel', schema, cfg).cuda()
    cp = torch.load(PARENT / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(cp['model'])
    vn = torch.as_tensor(np.array(va['x_excel'], copy=True), device='cuda')
    p, inference_seconds = models.predict(model, vn, None, 128)
    with np.load(PARENT / 'predictions_validation.npz') as saved:
        error = float(np.max(np.abs(p-saved['p'])))
        np.testing.assert_allclose(p, saved['p'], rtol=1e-5, atol=1e-6)
    actual_metrics = metrics(va['y'], p, va['ids'])
    if abs(actual_metrics['average_precision']-freeze()['parent_metrics']['average_precision']) > 1e-8:
        raise RuntimeError('Parent AP changed during identical eval-mode replay')
    first, _ = models.predict(model, vn[:256], None, 128)
    second, _ = models.predict(model, vn[:256], None, 128)
    np.testing.assert_array_equal(first, second)
    params = sum(p.numel() for p in model.parameters())
    if params != 470738:
        raise RuntimeError('Parent parameter count changed')
    del model, vn
    cuda_cleanup()
    x = torch.as_tensor(np.array(tr['x_excel'][:256], copy=True), device='cuda')
    y = torch.as_tensor(np.array(tr['y'][:256], copy=True), device='cuda', dtype=torch.float32)
    q = torch.linspace(0., 1., 256, device='cuda')
    available = torch.arange(256, device='cuda') >= 64
    mi = torch.tensor(schema['excel']['mi_weights'], device='cuda', dtype=torch.float32)
    traces = []
    for coefficient in (0., 1.):
        models.seed_all(42)
        model = models.build('excel', schema, cfg).cuda()
        model.load_state_dict(cp['model'])
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.)
        model.train()
        trace = hashlib.sha256()
        value, hard, soft = update(model, opt, x, y, q, available, torch.arange(256, device='cuda'), mi, coefficient, trace)
        if not np.isfinite([value, hard, soft]).all():
            raise RuntimeError('AMP smoke returned nonfinite metrics')
        traces.append(trace.hexdigest())
        del model, opt
        cuda_cleanup()
    if traces[0] != traces[1]:
        raise RuntimeError('Paired smoke has different mixing draws')
    peak = torch.cuda.max_memory_reserved()/2**30
    if peak > 12:
        raise RuntimeError('Smoke exceeded existing 12GiB allocation cap')
    np.savez_compressed(HERE / 'parent_replay_validation.npz', ids=va['ids'], y=va['y'], p=p)
    receipt = {'status': 'PASS', 'parent_metrics': actual_metrics, 'parent_reload_max_abs_error': error,
               'inference_seconds': inference_seconds, 'inference_determinism': 'PASS', 'parameter_count': params,
               'bf16_backward_update': 'PASS', 'paired_mix_trace_sha256': traces[0],
               'peak_reserved_gib': peak, 'gpu': torch.cuda.get_device_name(0), 'resource': resource,
               'execution_code': execution_hashes(), 'prediction_sha256': sha(HERE / 'parent_replay_validation.npz'), 'finished_utc': now()}
    write(path, receipt)
    del x, y, q, available, mi, cp
    cuda_cleanup()
    budget.heartbeat()
    print('CUDA_SMOKE_PASS', actual_metrics, flush=True)
    return receipt
