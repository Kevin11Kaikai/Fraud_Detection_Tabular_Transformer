from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import psutil

from ..resource_coordination import scheduler as fifo
from ..kaggle_fraudsquad_reproduction_20261011.common import metrics, sha

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
OWNER = HERE.name
SOURCE = HERE.parent / 'kaggle_fraudsquad_reproduction_20261011'
TRANSFER = HERE.parent / 'identity_uid_transformer_transfer_20261011'
PARENT = TRANSFER / 'runs/excel_B1C_seed42'
CACHE = TRANSFER / 'derived/B1C'
PARENT_SHA = 'b6d0feda8435c9d183ead350e4864f6e2cec941e21105a93fd216d622e4d5e1b'
LIMITS = {'smoke': 300., 'teacher': 600., 'bce_control': 2700., 'xgb_kd_lambda1': 2700., 'verification': 900.}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fifo.write(path, value)


def event(kind, **fields):
    with (HERE / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'utc': now(), 'event': kind, **fields}, ensure_ascii=False, allow_nan=False) + '\n')


def execution_hashes():
    names = ('common.py', 'loss.py', 'teacher.py', 'training.py', 'contracts.py', 'campaign.py')
    return {name: sha(HERE / name) for name in names}


def original_hashes():
    parent = read(PARENT / 'result.json')
    if sha(PARENT / 'best.pt') != PARENT_SHA or parent['checkpoint_sha256'] != PARENT_SHA:
        raise RuntimeError('Fixed parent checkpoint mismatch')
    paths = {PARENT / 'best.pt': PARENT_SHA, PARENT / 'result.json': sha(PARENT / 'result.json'),
             PARENT / 'predictions_validation.npz': parent['prediction_sha256']}
    paths.update({Path(name): value for name, value in parent['code_fingerprints'].items()})
    schema = read(CACHE / 'schema.json')
    receipt = read(CACHE / 'excel/receipt.json')
    paths[CACHE / 'schema.json'] = parent['input_hash'].split(':')[0]
    paths[CACHE / 'excel/receipt.json'] = parent['input_hash'].split(':')[1]
    paths.update({CACHE / name: value for name, value in schema['files'].items()})
    paths.update({CACHE / 'excel' / name: value for name, value in receipt['files'].items()})
    source = read(SOURCE / 'protocol.json')
    paths.update({PROJECT / 'data/raw/train_transaction.csv': source['raw_transaction_sha256'],
                  PROJECT / 'data/raw/train_identity.csv': source['raw_identity_sha256']})
    paths.update({SOURCE / name: sha(SOURCE / name) for name in ('features.py', 'common.py', 'protocol.json')})
    inventory = PROJECT / 'reports/model_inventory_20261012/model_cards.json'
    paths[inventory] = sha(inventory)
    for card in read(inventory):
        if any(abs(card['average_precision']-ap) < 1e-12 for ap in (.6083378935148605, .6338591015757505, .6877080964842449, .6715574513192961)):
            paths[Path(card['result_path'])] = sha(card['result_path'])
            paths[Path(card['prediction_path'])] = sha(card['prediction_path'])
            paths[Path(card['checkpoint_path'])] = card['checkpoint_expected_sha256']
    for path, expected in paths.items():
        if sha(path) != expected:
            raise RuntimeError(f'Original artifact hash mismatch: {path}')
    return {str(path): value for path, value in paths.items()}


def freeze():
    target = HERE / 'protocol.json'
    if target.exists():
        protocol = read(target)
        if protocol['execution_code'] != execution_hashes():
            raise RuntimeError('Frozen execution code changed')
        return protocol
    original = original_hashes()
    parent = read(PARENT / 'result.json')
    xgb = read(SOURCE / 'protocol.json')['xgb'].copy()
    xgb.pop('early_stopping_rounds')
    xgb['n_jobs'] = 2
    protocol = {'campaign': OWNER, 'created_utc': now(), 'authorization': '2026-10-12 user explicitly approved implementation; new 2-hour active GPU budget, AP priority.',
                'backend': 'LOCAL_ML_FRAUD_WORKFLOW', 'parent_run': str(PARENT), 'parent_checkpoint_sha256': PARENT_SHA,
                'parent_metrics': parent['metrics'], 'input_hash': parent['input_hash'], 'seed': 42,
                'train_rows': 413378, 'validation_rows': 88581, 'test_accessed': False,
                'maximum_active_gpu_seconds': 7200, 'phase_limits_seconds': LIMITS,
                'teacher_config': xgb, 'teacher_prefix_fractions': [.25, .5, .75], 'teacher_seconds_per_fold': 180,
                'student_config': parent['config'] | {'lr': 3e-5, 'epochs': 5, 'warmup': 0, 'schedule': 'constant', 'optimizer_reset': True},
                'trials': {'bce_control': 0., 'xgb_kd_lambda1': 1.}, 'temperature': 1.,
                'selection': 'Maximum development validation AP including epoch zero; ties retain earlier. Primary paired comparison uses common completed epochs.',
                'target_ap': .6083378935148605, 'recall_hard_gate': False,
                'claim_boundary': 'Reused development validation. Student has whole-TRAIN covariate maps; teachers independently fit past-only maps. Not a streaming or independent test assessment.',
                'original_artifacts': original, 'execution_code': execution_hashes()}
    write(target, protocol)
    write(HERE / 'budget.json', {'created_utc': now(), 'maximum_active_gpu_seconds': 7200., 'active_gpu_seconds': 0., 'inflight': None, 'events': []})
    write(HERE / 'historical_costs.json', {'learning_signal_budget_unchanged': read(HERE.parent / 'learning_signal_ft_20261011/budget.json'),
                                        'parent_training_seconds': parent['training_seconds'], 'new_phase_budget_is_not_a_reset': True})
    return protocol


def verify_original(protocol=None):
    protocol = protocol or freeze()
    for path, expected in protocol['original_artifacts'].items():
        if sha(path) != expected:
            raise RuntimeError(f'Immutable original changed: {path}')
    return {'status': 'PASS', 'verified_files': len(protocol['original_artifacts']), 'test_accessed': False}


class BudgetExhausted(RuntimeError):
    pass


class Budget:
    def __init__(self):
        self.path = HERE / 'budget.json'
        self.state = read(self.path)
        previous = self.state['inflight']
        if previous:
            if fifo.alive(previous):
                raise RuntimeError('A live worker already owns this campaign budget')
            # No trustworthy exit timestamp after a forced kill: charge the full
            # remaining phase reservation rather than silently reset usage.
            charged = previous['reserved_seconds']
            self.state['active_gpu_seconds'] += charged
            self.state['events'].append(previous | {'status': 'INTERRUPTION_CONSERVATIVE_CHARGE', 'seconds': charged, 'finished_utc': now()})
            self.state['inflight'] = None
            write(self.path, self.state)
        self.started = None
        self.cap = None

    def phase_used(self, name):
        return sum(e['seconds'] for e in self.state['events'] if e['name'] == name)

    def remaining(self):
        elapsed = time.monotonic() - self.started if self.started is not None else 0.
        total = self.state['maximum_active_gpu_seconds'] - self.state['active_gpu_seconds'] - elapsed
        return min(total, self.cap - elapsed) if self.cap is not None else total

    def check(self, reserve=0.):
        if self.remaining() <= reserve:
            raise BudgetExhausted('Active GPU budget or fixed phase cap exhausted')

    def heartbeat(self):
        self.check()
        self.state['inflight']['last_heartbeat_utc'] = now()
        self.state['inflight']['elapsed_seconds'] = time.monotonic() - self.started
        write(self.path, self.state)

    @contextmanager
    def phase(self, name):
        cap = min(LIMITS[name] - self.phase_used(name), self.remaining())
        if cap <= 0:
            raise BudgetExhausted(f'No remaining phase reservation: {name}')
        self.started = time.monotonic()
        self.cap = cap
        info = {'name': name, 'pid': os.getpid(), 'process_created': psutil.Process().create_time(), 'started_utc': now(), 'reserved_seconds': cap}
        self.state['inflight'] = info
        write(self.path, self.state)
        status = 'COMPLETE'
        try:
            yield self
        except BaseException:
            status = 'FAILED_OR_INTERRUPTED'
            raise
        finally:
            seconds = time.monotonic() - self.started
            self.state['active_gpu_seconds'] += seconds
            self.state['events'].append(info | {'status': status, 'seconds': seconds, 'finished_utc': now()})
            self.state['inflight'] = None
            self.started = self.cap = None
            write(self.path, self.state)


def authorize_fifo():
    with fifo.guard(fifo.ROOT):
        policy = fifo.policy_read(fifo.ROOT)
        if OWNER not in policy['authorized_owners']:
            write(HERE / 'scheduler_policy_before.json', policy)
            policy['authorized_owners'].append(OWNER)
            policy.setdefault('additional_authorizations', {})[OWNER] = {'utc': now(), 'basis': 'Direct implementation approval of the 2-hour XGBoost-to-ExcelFormer plan.'}
            fifo.write(fifo.ROOT / 'policy.json', policy)
            fifo.event(fifo.ROOT, 'AUTHORIZE_OWNER', owner=OWNER)


@contextmanager
def gpu_lease(purpose):
    authorize_fifo()
    print('QUEUE_GPU_FIFO', purpose, flush=True)
    with fifo.lease(OWNER, purpose, owner_directory=HERE) as resource:
        fifo.configure_torch(resource)
        print('GPU_ADMITTED', purpose, resource, flush=True)
        event('GPU_ADMITTED', purpose=purpose, **resource)
        yield resource


def assert_lease(resource):
    with fifo.guard(fifo.ROOT):
        state = fifo.state_read(fifo.ROOT)
        active = state.get('lease')
        if not active or active['pid'] != os.getpid() or active['lease_id'] != resource['lease_id']:
            raise RuntimeError('GPU lease identity conflict')
        blockers = fifo.gpu_blockers(state)
    if blockers:
        raise RuntimeError(f'Concurrent GPU worker conflict: {blockers}')


def cuda_cleanup():
    import torch
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def load_inputs():
    from ..identity_uid_transformer_transfer_20261011.common import load
    parts, schema = load('B1C', 'excel')
    if len(parts['train']['y']) != 413378 or len(parts['validation']['y']) != 88581 or len(schema['excel']['columns']) != 335:
        raise RuntimeError('Fixed input dimensions changed')
    return parts, schema


def artifact_hashes(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(Path(root).rglob('*'))
            if p.is_file() and p.suffix not in ('.tmp', '.pyc') and '__pycache__' not in p.parts
            and p.name not in ('artifact_manifest.json', 'worker.stdout.log', 'worker.stderr.log', 'resource_status.json')}
