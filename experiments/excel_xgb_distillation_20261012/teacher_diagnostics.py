"""Descriptive diagnostics only; never used to select or tune teachers."""
from __future__ import annotations

import numpy as np

from .common import CACHE, HERE, now, read, sha, write


def main():
    receipt = read(HERE / 'teacher/receipt.json')
    path = HERE / 'teacher/soft_targets_train.npz'
    if sha(path) != receipt['files']['soft_targets_train.npz']:
        raise RuntimeError('Teacher targets changed')
    y = np.load(CACHE / 'train_y.npy', mmap_mode='r')
    records = []
    with np.load(path) as targets:
        for fold in receipt['folds']:
            a, b = fold['start'], fold['end']
            q = targets['q'][a:b].astype(float)
            labels = np.asarray(y[a:b], dtype=float)
            clipped = np.clip(q, 1e-12, 1-1e-12)
            records.append({'fold': fold['fold'], 'rows': len(q), 'fraud_prevalence': float(labels.mean()),
                'mean_teacher_probability': float(q.mean()), 'teacher_bce': float(-(labels*np.log(clipped)+(1-labels)*np.log1p(-clipped)).mean()),
                'positive_mean_probability': float(q[labels==1].mean()), 'negative_mean_probability': float(q[labels==0].mean()),
                'quantiles': np.quantile(q, [0,.01,.5,.9,.99,1]).tolist(),
                'oof_block_metrics': fold['training_ap_diagnostic_only'],
                'rounds': fold['rounds'], 'peak_device_memory_mib_sampled': fold['peak_device_memory_mib_sampled']})
    write(HERE / 'teacher_diagnostics.json', {'status': 'PASS', 'scope': 'TRAIN temporal out-of-fold blocks; descriptive only, no teacher selection',
        'records': records, 'validation_test_labels_accessed': False, 'finished_utc': now()})
    print('TEACHER_DIAGNOSTICS', records, flush=True)


if __name__ == '__main__':
    main()
