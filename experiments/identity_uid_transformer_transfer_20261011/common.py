from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np

from ..kaggle_fraudsquad_reproduction_20261011.common import metrics, now, read, sha, write

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SOURCE = HERE.parent / 'kaggle_fraudsquad_reproduction_20261011'
ABLATION = HERE.parent / 'identity_uid_ablation_20261011'
EXTERNAL = Path(r'D:\Year3_Mao_Projects\Tabular_Transformer\benchmark\experiments')
VARIANTS = ('B00', 'B10', 'B1M', 'B1C')
SEEDS = (42, 43)
ARCHITECTURES = ('ft', 'excel')


def external_module(name):
    alias = 'identity_transfer_external'
    if alias not in sys.modules:
        module = types.ModuleType(alias)
        module.__path__ = [str(EXTERNAL)]
        sys.modules[alias] = module
    return importlib.import_module(alias + '.' + name)


def fingerprints():
    paths = list(HERE.glob('*.py')) + [HERE / 'protocol.json',
        PROJECT / 'experiments/ft_upgrade_20261010/train.py',
        EXTERNAL / 'architecture_benchmark_20261010/vendor/excel.py',
        EXTERNAL / 'architecture_benchmark_20261010/vendor/excel_lib.py']
    return {str(p): sha(p) for p in paths}


def array_hash(array):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def labels():
    with np.load(SOURCE / 'derived/labels_ids.npz') as z:
        return {k: z[k].copy() for k in z.files}


def load(variant, architecture='ft'):
    root = HERE / 'derived' / variant
    schema = read(root / 'schema.json')
    parts = {}
    for part in ('train', 'validation'):
        parts[part] = {k: np.load(root / f'{part}_{k}.npy', mmap_mode='r')
                       for k in ('x_num', 'x_cat', 'ids', 'y', 'dt')}
        if architecture == 'excel':
            parts[part]['x_excel'] = np.load(root / 'excel' / f'{part}.npy', mmap_mode='r')
    if architecture == 'excel':
        schema['excel'] = read(root / 'excel/receipt.json')
    return parts, schema


def trial_name(architecture, variant, seed):
    return f'{architecture}_{variant}_seed{seed}'


def snapshot_gpu():
    return external_module('champion_dual_axis_20261011.preflight').gpu_processes()
