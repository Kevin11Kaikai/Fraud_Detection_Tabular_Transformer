"""Prefix-fitted FraudSquad feature maps and chronological CUDA teachers."""
from __future__ import annotations

import gc
import subprocess
import time

import numpy as np
import pandas as pd

from .common import *
from ..kaggle_fraudsquad_reproduction_20261011.features import build, TRANSACTION_COLUMNS, STRING_COLUMNS


def raw_training():
    """Read TRAIN covariates only; labels come from the admitted TRAIN cache."""
    types = {name: 'float32' for name in TRANSACTION_COLUMNS}
    types.update({name: 'category' for name in STRING_COLUMNS if name in TRANSACTION_COLUMNS})
    types['TransactionID'] = 'int64'
    frame = pd.read_csv(PROJECT / 'data/raw/train_transaction.csv', usecols=TRANSACTION_COLUMNS,
                        dtype=types, index_col='TransactionID', nrows=413378)
    ids = np.load(CACHE / 'train_ids.npy', mmap_mode='r')
    dt = np.load(CACHE / 'train_dt.npy', mmap_mode='r')
    np.testing.assert_array_equal(frame.index.to_numpy(), ids)
    np.testing.assert_array_equal(frame.TransactionDT.to_numpy(), dt)
    identity = PROJECT / 'data/raw/train_identity.csv'
    names = pd.read_csv(identity, nrows=0).columns.tolist()
    id_types = {name: 'category' if name in STRING_COLUMNS else 'float32' for name in names}
    id_types['TransactionID'] = 'int64'
    kept = []
    for chunk in pd.read_csv(identity, dtype=id_types, chunksize=50000, index_col='TransactionID'):
        kept.append(chunk.loc[chunk.index.isin(frame.index)].copy())
    identity_frame = pd.concat(kept)
    for name in STRING_COLUMNS:
        if name in identity_frame:
            identity_frame[name] = identity_frame[name].astype('category')
    frame = frame.join(identity_frame, how='left', sort=False).reindex(ids)
    if not frame.index.is_unique or 'isFraud' in frame or np.any(np.diff(dt) < 0):
        raise RuntimeError('TRAIN raw ordering or label exclusion failed')
    return frame


def blocks(dt):
    starts = [int(np.searchsorted(dt, dt[int(len(dt)*fraction)], side='left')) for fraction in (.25, .5, .75)]
    ends = starts[1:] + [len(dt)]
    for start, end in zip(starts, ends):
        if start <= 0 or end <= start or dt[start-1] >= dt[start]:
            raise RuntimeError('Teacher timestamp boundary is not strictly past-only')
    return list(zip(starts, ends))


def prepare_features():
    root = HERE / 'teacher'
    receipt_path = root / 'features_receipt.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        for name, expected in receipt['files'].items():
            if sha(root / name) != expected:
                raise RuntimeError('Prepared prefix feature hash mismatch')
        return receipt
    root.mkdir(exist_ok=True)
    start_time = time.perf_counter()
    frame = raw_training()
    dt = np.load(CACHE / 'train_dt.npy', mmap_mode='r')
    records = []
    for fold, (start, end) in enumerate(blocks(dt), 1):
        prefix, future, diagnostics = build(frame.iloc[:start].copy(), frame.iloc[start:end].copy(), 'train_only')
        if diagnostics['fit_rows'] != start:
            raise RuntimeError('Teacher feature map fit exceeded past prefix')
        directory = root / f'fold{fold}'
        directory.mkdir(exist_ok=True)
        x = prefix.fillna(-1).to_numpy(dtype=np.float32)
        xv = future.fillna(-1).to_numpy(dtype=np.float32)
        if not np.isfinite(x).all() or not np.isfinite(xv).all():
            raise RuntimeError('Nonfinite prefix feature matrix')
        np.save(directory / 'train.npy', x)
        np.save(directory / 'block.npy', xv)
        write(directory / 'feature_scope.json', diagnostics | {'fold': fold, 'prefix_end': start, 'block_end': end,
                     'prefix_max_timestamp': float(dt[start-1]), 'block_min_timestamp': float(dt[start]),
                     'labels_used_in_feature_fit': False, 'validation_test_accessed': False})
        records.append({'fold': fold, 'start': start, 'end': end, 'features': x.shape[1], 'scope': 'past_prefix_only'})
        print('PREFIX_FEATURES_READY', records[-1], flush=True)
        del prefix, future, x, xv
        gc.collect()
    del frame
    gc.collect()
    files = {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file() and p.name in ('train.npy', 'block.npy', 'feature_scope.json')}
    receipt = {'status': 'PASS', 'folds': records, 'files': files, 'cpu_seconds': time.perf_counter()-start_time,
               'training_only': True, 'validation_test_labels_accessed': False, 'finished_utc': now()}
    write(receipt_path, receipt)
    return receipt


def gpu_memory_mib():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True, timeout=15)
    return int(raw.strip().splitlines()[0])


def cuda_teachers(budget, resource):
    import xgboost as xgb
    protocol = freeze()
    root = HERE / 'teacher'
    receipt_path = root / 'receipt.json'
    if receipt_path.exists():
        prior = read(receipt_path)
        if prior['protocol_sha256'] != sha(HERE / 'protocol.json'):
            raise RuntimeError('Teacher protocol mismatch')
        for name, expected in prior['files'].items():
            if sha(root / name) != expected:
                raise RuntimeError('Teacher completed artifact mismatch')
        print('REUSE_COMPLETE_TEACHERS', flush=True)
        return prior
    features = read(root / 'features_receipt.json')
    y = np.load(CACHE / 'train_y.npy', mmap_mode='r')
    ids = np.load(CACHE / 'train_ids.npy', mmap_mode='r')
    dt = np.load(CACHE / 'train_dt.npy', mmap_mode='r')
    q = np.full(len(y), .5, dtype=np.float32)
    available = np.zeros(len(y), dtype=bool)
    fold_ids = np.zeros(len(y), dtype=np.int8)
    records = []
    for item in features['folds']:
        fold, start, end = item['fold'], item['start'], item['end']
        directory = root / f'fold{fold}'
        completed = directory / 'result.json'
        if completed.exists():
            result = read(completed)
            for name, expected in result['files'].items():
                if sha(directory / name) != expected:
                    raise RuntimeError('Completed teacher fold hash mismatch')
            prediction = np.load(directory / 'probabilities.npy')
            records.append(result)
        else:
            budget.check(reserve=20.)
            assert_lease(resource)
            training = np.load(directory / 'train.npy', mmap_mode='r')
            prediction_input = np.load(directory / 'block.npy', mmap_mode='r')
            tick = time.perf_counter()

            class StopAtBudget(xgb.callback.TrainingCallback):
                timed_out = False
                peak_device_memory_mib = gpu_memory_mib()

                def after_iteration(self, model, epoch, evals_log):
                    if epoch % 50 == 0:
                        budget.heartbeat()
                        self.peak_device_memory_mib = max(self.peak_device_memory_mib, gpu_memory_mib())
                        write(directory / 'progress.json', {'status': 'TRAINING', 'iteration': epoch+1, 'elapsed_seconds': time.perf_counter()-tick, 'updated_utc': now()})
                    self.timed_out = time.perf_counter()-tick >= protocol['teacher_seconds_per_fold']
                    return self.timed_out or budget.remaining() < 15.

            callback = StopAtBudget()
            model = xgb.XGBClassifier(**protocol['teacher_config'], callbacks=[callback])
            model.fit(training, np.asarray(y[:start]), verbose=False)
            actual = json.loads(model.get_booster().save_config())
            if not actual['learner']['generic_param']['device'].startswith('cuda'):
                raise RuntimeError('XGBoost silently fell back from CUDA')
            budget.check(reserve=2.)
            prediction = model.predict_proba(prediction_input)[:, 1].astype(np.float32)
            if not np.isfinite(prediction).all() or ((prediction < 0) | (prediction > 1)).any():
                raise RuntimeError('Invalid chronological teacher probabilities')
            model.save_model(directory / 'model.ubj')
            write(directory / 'actual_booster_config.json', actual)
            np.save(directory / 'probabilities.npy', prediction)
            result = item | {'status': 'COMPLETE', 'actual_device': actual['learner']['generic_param']['device'],
                      'rounds': model.get_booster().num_boosted_rounds(), 'time_cap_reached': callback.timed_out,
                      'train_predict_seconds': time.perf_counter()-tick, 'peak_device_memory_mib_sampled': callback.peak_device_memory_mib,
                      'training_ap_diagnostic_only': metrics(y[start:end], prediction, ids[start:end]),
                      'predicted_block_labels_used_for_fit_or_selection': False,
                      'files': {name: sha(directory / name) for name in ('model.ubj', 'actual_booster_config.json', 'probabilities.npy', 'feature_scope.json')}, 'finished_utc': now()}
            write(completed, result)
            del model, training, prediction_input
            gc.collect()
            print('TEACHER_COMPLETE', fold, 'rounds', result['rounds'], 'seconds', result['train_predict_seconds'], flush=True)
            records.append(result)
        if len(prediction) != end-start:
            raise RuntimeError('Teacher block length mismatch')
        q[start:end] = prediction
        available[start:end] = True
        fold_ids[start:end] = fold
        budget.heartbeat()
    np.savez_compressed(root / 'soft_targets_train.npz', ids=ids, dt=dt, q=q, available=available, fold=fold_ids)
    receipt = {'status': 'PASS', 'protocol_sha256': sha(HERE / 'protocol.json'), 'folds': records,
               'available_rows': int(available.sum()), 'unavailable_rows': int((~available).sum()),
               'train_rows': len(y), 'teacher_validation_test_labels_accessed': False,
               'files': {'soft_targets_train.npz': sha(root / 'soft_targets_train.npz'),
                         'features_receipt.json': sha(root / 'features_receipt.json')}, 'finished_utc': now()}
    write(receipt_path, receipt)
    return receipt
