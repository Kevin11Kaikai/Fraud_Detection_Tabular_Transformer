"""Matched fixed-epoch continuation; teacher is used only in the loss."""
from __future__ import annotations

import hashlib
import math
import shutil
import time

import pandas as pd
import torch

from .common import *
from .loss import mixed_loss
from ..identity_uid_transformer_transfer_20261011 import models


def update(model, optimizer, x, y, q, available, indices, mi, coefficient, trace):
    optimizer.zero_grad(set_to_none=True)
    totals = np.zeros(3, dtype=float)
    for chunk in indices.split(32):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, mask, permutation = model(x[chunk], None, mixup=True, beta=.5, mtype='feat_mix')
            a = (mask.float()*mi).sum(1)
            value, hard, soft = mixed_loss(logits, y[chunk], permutation, a, q[chunk], available[chunk], coefficient)
        if not torch.isfinite(value):
            raise RuntimeError('Nonfinite continuation loss')
        (value*(len(chunk)/len(indices))).backward()
        totals += np.array([float(value.detach()), float(hard), float(soft)])*len(chunk)
        trace.update(np.asarray(permutation, dtype=np.int64).tobytes())
        trace.update(np.packbits(mask.detach().cpu().numpy()).tobytes())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise RuntimeError('Nonfinite gradient norm')
    optimizer.step()
    return tuple(totals)


def checkpoint(path, data):
    models.checkpoint(path, data)


def run(trial, budget, resource):
    root = HERE / 'runs' / trial
    root.mkdir(parents=True, exist_ok=True)
    protocol = freeze()
    config = protocol['student_config'] | {'trial': trial, 'lambda': protocol['trials'][trial], 'temperature': 1.}
    if (root / 'result.json').exists():
        result = read(root / 'result.json')
        if result['config'] != config or result['execution_code'] != execution_hashes():
            raise RuntimeError('Completed continuation fingerprint mismatch')
        for name, expected in result['files'].items():
            if sha(root / name) != expected:
                raise RuntimeError('Completed continuation artifact hash mismatch')
        print('REUSE_COMPLETE', trial, flush=True)
        return result
    write(root / 'config.json', config)
    parts, schema = load_inputs()
    tr, va = parts['train'], parts['validation']
    teacher_path = HERE / 'teacher/soft_targets_train.npz'
    teacher_receipt = read(HERE / 'teacher/receipt.json')
    if sha(teacher_path) != teacher_receipt['files']['soft_targets_train.npz']:
        raise RuntimeError('Teacher target cache changed')
    with np.load(teacher_path) as teacher:
        np.testing.assert_array_equal(teacher['ids'], tr['ids'])
        np.testing.assert_array_equal(teacher['dt'], tr['dt'])
        q = torch.as_tensor(teacher['q'].copy(), device='cuda')
        available = torch.as_tensor(teacher['available'].copy(), device='cuda')
    models.seed_all(42)
    torch.cuda.reset_peak_memory_stats()
    x = torch.as_tensor(np.array(tr['x_excel'], copy=True), device='cuda')
    vn = torch.as_tensor(np.array(va['x_excel'], copy=True), device='cuda')
    y = torch.as_tensor(np.array(tr['y'], copy=True), device='cuda', dtype=torch.float32)
    mi = torch.tensor(schema['excel']['mi_weights'], device='cuda', dtype=torch.float32)
    model = models.build('excel', schema, config).cuda()
    parent = torch.load(PARENT / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(parent['model'])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.)
    history = []
    best_ap = protocol['parent_metrics']['average_precision']
    best_epoch = 0
    next_epoch = 1
    if (root / 'last.pt').exists():
        cp = torch.load(root / 'last.pt', map_location='cpu', weights_only=False)
        if cp['config'] != config or cp['execution_code'] != execution_hashes() or cp['parent_sha256'] != PARENT_SHA:
            raise RuntimeError('Resume checkpoint fingerprint mismatch')
        model.load_state_dict(cp['model'])
        opt.load_state_dict(cp['optimizer'])
        models.restore_rng(cp['rng'])
        history = cp['history']
        best_ap, best_epoch, next_epoch = cp['best_ap'], cp['best_epoch'], cp['epoch']+1
    else:
        checkpoint(root / 'epoch_00.pt', {'model': model.state_dict(), 'config': config, 'epoch': 0, 'parent_sha256': PARENT_SHA})
        shutil.copyfile(root / 'epoch_00.pt', root / 'best.pt')
        shutil.copyfile(HERE / 'parent_replay_validation.npz', root / 'predictions_epoch_00.npz')
        shutil.copyfile(root / 'predictions_epoch_00.npz', root / 'predictions_validation.npz')
    stop_reason = 'MAX_EPOCHS'
    try:
        for epoch in range(next_epoch, 6):
            estimate = float(np.median([r['epoch_seconds'] for r in history])) if history else 420.
            if budget.remaining() < estimate+45.:
                stop_reason = 'BUDGET_EXHAUSTED_BEFORE_COMPLETE_EPOCH'
                break
            assert_lease(resource)
            tick = time.perf_counter()
            model.train()
            order = models.order('excel', 'pre', len(y), np.asarray(tr['y']), 42, 16+epoch)
            order_hash = hashlib.sha256(order.numpy().tobytes()).hexdigest()
            trace = hashlib.sha256()
            totals = np.zeros(3, dtype=float)
            count = 0
            train_tick = time.perf_counter()
            heartbeat = train_tick
            for off in range(0, len(order), 256):
                indices = order[off:off+256].to('cuda')
                totals += update(model, opt, x, y, q, available, indices, mi, config['lambda'], trace)
                count += len(indices)
                if time.perf_counter()-heartbeat > 10.:
                    budget.heartbeat()
                    heartbeat = time.perf_counter()
                if (off//256) % 128 == 0:
                    write(root / 'progress.json', {'status': 'TRAINING', 'trial': trial, 'epoch': epoch, 'processed': count,
                        'total': len(order), 'best_ap': best_ap, 'budget_remaining_seconds': budget.remaining(), 'updated_utc': now()})
                if (off//256) % 256 == 0:
                    assert_lease(resource)
            torch.cuda.synchronize()
            train_seconds = time.perf_counter()-train_tick
            p, inference = models.predict(model, vn, None, 128)
            measured = metrics(va['y'], p, va['ids'])
            epoch_file = root / f'epoch_{epoch:02d}.pt'
            checkpoint(epoch_file, {'model': model.state_dict(), 'config': config, 'epoch': epoch,
                                  'parent_sha256': PARENT_SHA, 'execution_code': execution_hashes()})
            prediction_file = root / f'predictions_epoch_{epoch:02d}.npz'
            np.savez_compressed(prediction_file, ids=va['ids'], y=va['y'], p=p)
            if measured['average_precision'] > best_ap:
                best_ap, best_epoch = measured['average_precision'], epoch
                shutil.copyfile(epoch_file, root / 'best.pt')
                shutil.copyfile(prediction_file, root / 'predictions_validation.npz')
            row = {'epoch': epoch, 'parent_epoch_offset': 16+epoch, 'validation_ap': measured['average_precision'],
                   'training_loss': float(totals[0]/count), 'hard_loss': float(totals[1]/count), 'soft_loss': float(totals[2]/count),
                   'training_seconds': train_seconds, 'inference_seconds': inference, 'epoch_seconds': time.perf_counter()-tick,
                   'sampler_order_sha256': order_hash, 'mix_trace_sha256': trace.hexdigest(), 'samples': count,
                   'optimizer_steps': math.ceil(count/256), 'learning_rate': 3e-5,
                   'checkpoint_sha256': sha(epoch_file), 'prediction_sha256': sha(prediction_file), **measured}
            history.append(row)
            pd.DataFrame(history).to_csv(root / 'history.csv', index=False)
            checkpoint(root / 'last.pt', {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'rng': models.rng_state(),
                'epoch': epoch, 'history': history, 'best_ap': best_ap, 'best_epoch': best_epoch,
                'config': config, 'execution_code': execution_hashes(), 'parent_sha256': PARENT_SHA})
            budget.heartbeat()
            if torch.cuda.max_memory_reserved() > 12*2**30:
                raise RuntimeError('Existing 12GiB memory cap exceeded')
            print(trial, 'epoch', epoch, 'AP', round(measured['average_precision'], 6), 'best', round(best_ap, 6),
                  'R1', round(measured['recall_top_1pct'], 6), 'seconds', round(row['epoch_seconds'], 1), flush=True)
        with np.load(root / 'predictions_validation.npz') as saved:
            selected_metrics = metrics(saved['y'], saved['p'], saved['ids'])
        result = {'status': 'COMPLETE', 'trial': trial, 'config': config, 'metrics': selected_metrics, 'best_epoch': best_epoch,
                  'completed_epochs': len(history), 'stop_reason': stop_reason,
                  'training_seconds': sum(r['training_seconds'] for r in history),
                  'fit_seconds': sum(r['epoch_seconds'] for r in history), 'resource_wait_seconds': resource['waiting_seconds'],
                  'parameter_count': sum(p.numel() for p in model.parameters()), 'gpu': torch.cuda.get_device_name(0), 'precision': 'bfloat16',
                  'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30, 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                  'parent_checkpoint_sha256': PARENT_SHA, 'input_hash': protocol['input_hash'], 'execution_code': execution_hashes(),
                  'teacher_receipt_sha256': sha(HERE / 'teacher/receipt.json'), 'test_accessed': False, 'finished_utc': now(),
                  'files': {name: sha(root / name) for name in ('best.pt', 'predictions_validation.npz', 'history.csv', 'config.json') if (root/name).exists()}}
        write(root / 'result.json', result)
        write(root / 'progress.json', {'status': 'COMPLETE', 'metrics': selected_metrics, 'best_epoch': best_epoch, 'updated_utc': now()})
        return result
    finally:
        model = opt = x = vn = y = q = available = mi = None
        cuda_cleanup()
