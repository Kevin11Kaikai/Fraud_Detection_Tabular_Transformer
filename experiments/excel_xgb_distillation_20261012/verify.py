from __future__ import annotations

import gc

import numpy as np
import torch

from .common import *
from ..identity_uid_transformer_transfer_20261011 import models


METRIC_KEYS = ('average_precision', 'roc_auc', 'recall_top_1pct', 'precision_top_1pct', 'recall_top_5pct', 'brier_score')


def selected_epoch(trial, maximum):
    result = read(HERE / 'runs' / trial / 'result.json')
    import pandas as pd
    history_path = HERE / 'runs' / trial / 'history.csv'
    history = pd.read_csv(history_path).to_dict('records') if history_path.exists() else []
    candidates = [(0, freeze()['parent_metrics']['average_precision'])] + [(int(r['epoch']), r['validation_ap']) for r in history if r['epoch'] <= maximum]
    return max(candidates, key=lambda item: (item[1], -item[0]))[0]


def cpu_verify():
    import pandas as pd
    protocol = freeze()
    immutable = verify_original(protocol)
    parts, _ = load_inputs()
    va, tr = parts['validation'], parts['train']
    results, histories = {}, {}
    for trial in protocol['trials']:
        root = HERE / 'runs' / trial
        result = read(root / 'result.json')
        if result['execution_code'] != execution_hashes() or result['parent_checkpoint_sha256'] != PARENT_SHA:
            raise RuntimeError('Run provenance mismatch')
        for name, expected in result['files'].items():
            if sha(root / name) != expected:
                raise RuntimeError('Run artifact changed before verification')
        with np.load(root / 'predictions_validation.npz') as saved:
            np.testing.assert_array_equal(saved['ids'], va['ids'])
            np.testing.assert_array_equal(saved['y'], va['y'])
            rescored = metrics(saved['y'], saved['p'], saved['ids'])
        for key in METRIC_KEYS:
            if abs(rescored[key]-result['metrics'][key]) > 1e-12:
                raise RuntimeError(f'Metric rescore mismatch: {trial}/{key}')
        history = pd.read_csv(root / 'history.csv').to_dict('records') if (root/'history.csv').exists() else []
        for row in history:
            epoch = int(row['epoch'])
            if sha(root / f'epoch_{epoch:02d}.pt') != row['checkpoint_sha256'] or sha(root / f'predictions_epoch_{epoch:02d}.npz') != row['prediction_sha256']:
                raise RuntimeError('Complete-epoch artifact mismatch')
        results[trial], histories[trial] = result, history
    common_epochs = min(r['completed_epochs'] for r in results.values())
    for a, b in zip(histories['bce_control'][:common_epochs], histories['xgb_kd_lambda1'][:common_epochs]):
        for key in ('epoch', 'sampler_order_sha256', 'mix_trace_sha256', 'samples', 'optimizer_steps', 'learning_rate'):
            if a[key] != b[key]:
                raise RuntimeError(f'Paired training variable mismatch: {key}')
    config0 = results['bce_control']['config'].copy()
    config1 = results['xgb_kd_lambda1']['config'].copy()
    for config in (config0, config1):
        config.pop('trial')
        config.pop('lambda')
    if config0 != config1:
        raise RuntimeError('Unrelated student config changed')
    teacher = read(HERE / 'teacher/receipt.json')
    with np.load(HERE / 'teacher/soft_targets_train.npz') as targets:
        np.testing.assert_array_equal(targets['ids'], tr['ids'])
        np.testing.assert_array_equal(targets['dt'], tr['dt'])
        for fold in teacher['folds']:
            start, end = fold['start'], fold['end']
            if tr['dt'][start-1] >= tr['dt'][start] or not targets['available'][start:end].all() or not (targets['fold'][start:end] == fold['fold']).all():
                raise RuntimeError('Chronological target coverage failed')
        if targets['available'][:teacher['folds'][0]['start']].any():
            raise RuntimeError('Warmup rows unexpectedly have a teacher')
    result = {'status': 'PASS', 'immutable': immutable, 'metric_rescore': 'PASS', 'row_alignment': 'PASS',
              'paired_sampler_and_hid_mix': 'PASS', 'teacher_targets': 'PASS', 'common_completed_epochs': common_epochs,
              'paired_selected_epochs': {trial: selected_epoch(trial, common_epochs) for trial in protocol['trials']},
              'test_accessed': False, 'finished_utc': now()}
    write(HERE / 'verification_cpu.json', result)
    return result


def cuda_verify(budget, resource):
    protocol = freeze()
    cpu = cpu_verify()
    parts, schema = load_inputs()
    va = parts['validation']
    vn = torch.as_tensor(np.array(va['x_excel'], copy=True), device='cuda')
    records = []
    try:
        for trial in protocol['trials']:
            root = HERE / 'runs' / trial
            result = read(root / 'result.json')
            epochs = sorted(set([result['best_epoch'], cpu['paired_selected_epochs'][trial]]))
            for epoch in epochs:
                budget.check(reserve=40.)
                assert_lease(resource)
                models.seed_all(42)
                model = models.build('excel', schema, protocol['student_config']).cuda()
                cp = torch.load(root / f'epoch_{epoch:02d}.pt', map_location='cpu', weights_only=False)
                model.load_state_dict(cp['model'])
                p, seconds = models.predict(model, vn, None, 128)
                with np.load(root / f'predictions_epoch_{epoch:02d}.npz') as saved:
                    np.testing.assert_array_equal(saved['ids'], va['ids'])
                    np.testing.assert_array_equal(saved['y'], va['y'])
                    np.testing.assert_allclose(p, saved['p'], rtol=1e-5, atol=1e-6)
                    error = float(np.max(np.abs(p-saved['p'])))
                np.savez_compressed(root / f'replay_epoch_{epoch:02d}.npz', ids=va['ids'], y=va['y'], p=p)
                records.append({'trial': trial, 'epoch': epoch, 'metrics': metrics(va['y'], p, va['ids']),
                                'fresh_reload_max_abs_error': error, 'inference_seconds': seconds,
                                'checkpoint_sha256': sha(root / f'epoch_{epoch:02d}.pt')})
                del model, cp
                cuda_cleanup()
                budget.heartbeat()
    finally:
        vn = None
        cuda_cleanup()
    result = cpu | {'fresh_model_checkpoint_replay': 'PASS', 'records': records, 'resource': resource, 'finished_utc': now()}
    write(HERE / 'verification.json', result)
    print('FINAL_VERIFICATION_PASS', flush=True)
    return result
