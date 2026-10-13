from __future__ import annotations

import gc
import hashlib
import os
import time
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import psutil

from ..excel_xgb_distillation_20261012 import common as original
from ..resource_coordination import scheduler as fifo

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
SOURCE = HERE.parent/'excel_xgb_distillation_20261012'
OWNER = HERE.name
TRIALS = ('bce_control', 'xgb_kd_lambda1')
read, write, sha, now = original.read, original.write, original.sha, original.now
metrics, load_inputs = original.metrics, original.load_inputs


def fingerprints():
    sources = list(HERE.glob('*.py'))
    sources += [SOURCE/name for name in original.execution_hashes()]
    sources += list((SOURCE.parent/'identity_uid_transformer_transfer_20261011').glob('*.py'))
    vendor = Path(r'D:\Year3_Mao_Projects\Tabular_Transformer\benchmark\experiments\architecture_benchmark_20261010\vendor')
    sources += [vendor/'excel.py', vendor/'excel_lib.py']
    return {str(path):sha(path) for path in sorted(sources)}


def protocol():
    value = read(HERE/'protocol.json')
    if not value['execution_authorized']:
        raise RuntimeError('Additional GPU budget has not been authorized')
    if value['execution_fingerprints'] != fingerprints():
        raise RuntimeError('Frozen extension source changed')
    for path, expected in value['source_files'].items():
        if sha(path) != expected:
            raise RuntimeError('Frozen source changed: '+path)
    return value


class PairPatience:
    """Both arms stop together, after a declared minimum and shared epochs."""
    def __init__(self, initial, minimum=10, patience=8, delta=1e-4):
        self.best = dict(initial)
        self.bad = {trial:0 for trial in initial}
        self.minimum, self.patience, self.delta = minimum, patience, delta
        self.completed = 0

    def update(self, values):
        if set(values) != set(self.best) or any(not np.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
            raise ValueError('A complete finite AP pair is required')
        for trial, ap in values.items():
            if ap > self.best[trial]+self.delta:
                self.best[trial], self.bad[trial] = ap, 0
            else:
                self.bad[trial] += 1
        self.completed += 1
        return self.completed >= self.minimum and all(n >= self.patience for n in self.bad.values())

    def state(self):
        return dict(best=self.best, bad=self.bad, minimum=self.minimum, patience=self.patience,
                    delta=self.delta, completed=self.completed)

    @classmethod
    def restore(cls, state):
        value = cls(state['best'], state['minimum'], state['patience'], state['delta'])
        value.bad, value.completed = dict(state['bad']), state['completed']
        return value


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Count entire GPU lease windows; never reset earlier or interrupted usage."""
    def __init__(self):
        self.path = HERE/'budget.json'
        self.state = read(self.path)
        pending = self.state.get('inflight')
        if pending:
            if fifo.alive(pending):
                raise RuntimeError('An existing live worker owns this extension')
            self.state['active_gpu_seconds'] += pending['reserved_seconds']
            self.state['events'].append(pending | dict(status='INTERRUPTION_CONSERVATIVE_CHARGE',
                seconds=pending['reserved_seconds'], finished_utc=now()))
            self.state['inflight'] = None
            write(self.path, self.state)
        self.started = self.cap = None

    def remaining(self):
        elapsed = time.monotonic()-self.started if self.started is not None else 0.
        total = self.state['maximum_active_gpu_seconds']-self.state['active_gpu_seconds']-elapsed
        return min(total, self.cap-elapsed) if self.cap is not None else total

    def heartbeat(self):
        if self.remaining() <= 0:
            raise BudgetExceeded('Additional GPU budget exhausted')
        self.state['inflight']['elapsed_seconds'] = time.monotonic()-self.started
        self.state['inflight']['last_heartbeat_utc'] = now()
        write(self.path, self.state)

    @contextmanager
    def window(self, purpose, resource, reservation):
        if self.started is not None:
            raise RuntimeError('GPU accounting windows cannot overlap')
        self.cap = min(float(reservation), self.remaining())
        if self.cap <= 0:
            raise BudgetExceeded('No authorized GPU budget remains')
        self.started = time.monotonic()
        pending = dict(pid=os.getpid(), process_created=psutil.Process().create_time(),
            purpose=purpose, lease_id=resource['lease_id'], started_utc=now(), reserved_seconds=self.cap)
        self.state['inflight'] = pending
        write(self.path, self.state)
        status = 'COMPLETE'
        try:
            yield self
        except BaseException:
            status = 'FAILED_OR_INTERRUPTED'
            raise
        finally:
            seconds = time.monotonic()-self.started
            self.state['active_gpu_seconds'] += seconds
            self.state['events'].append(pending | dict(status=status, seconds=seconds,
                waiting_seconds=resource['waiting_seconds'], finished_utc=now()))
            self.state['queue_wait_seconds'] += resource['waiting_seconds']
            self.state['inflight'] = None
            self.started = self.cap = None
            write(self.path, self.state)


def assert_lease(resource):
    with fifo.guard(fifo.ROOT):
        state = fifo.state_read(fifo.ROOT)
        lease = state.get('lease')
        if not lease or lease['pid'] != os.getpid() or lease['lease_id'] != resource['lease_id']:
            raise RuntimeError('Extension no longer owns its GPU lease')
        blockers = fifo.gpu_blockers(state)
    if blockers:
        raise RuntimeError('Concurrent GPU worker conflict: '+str(blockers))


@contextmanager
def gpu_lease(purpose):
    plan = protocol()
    with fifo.guard(fifo.ROOT):
        policy = fifo.policy_read(fifo.ROOT)
        if OWNER not in policy['authorized_owners']:
            write(HERE/'scheduler_policy_before.json', policy)
            policy['authorized_owners'].append(OWNER)
            policy.setdefault('additional_authorizations', {})[OWNER] = dict(utc=now(),
                basis=plan['budget_authorization'])
            fifo.write(fifo.ROOT/'policy.json', policy)
            fifo.event(fifo.ROOT, 'AUTHORIZE_OWNER', owner=OWNER)
    print('QUEUE_GPU_FIFO', purpose, flush=True)
    with fifo.lease(OWNER, purpose, owner_directory=HERE) as resource:
        fifo.configure_torch(resource)
        print('GPU_ADMITTED', purpose, resource, flush=True)
        yield resource


def cuda_cleanup():
    import torch
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def array_hash(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()
