"""Freeze all conditions before observing new neural validation results."""
from __future__ import annotations

import importlib.metadata
import shutil

from .common import *


def freeze():
    if (HERE / 'protocol.json').exists():
        return read(HERE / 'protocol.json')
    seal = read(ABLATION / 'artifact_manifest.json')
    for name, expected in seal['files'].items():
        if sha(ABLATION / name) != expected:
            raise RuntimeError('Original ablation seal mismatch: ' + name)
    src = read(SOURCE / 'derived/train_only/manifest.json')
    for name, expected in src['array_hashes'].items():
        if sha(SOURCE / 'derived/train_only' / name) != expected:
            raise RuntimeError('Frozen source array mismatch')
    cfg = read(EXTERNAL / 'champion_dual_axis_20261011/config.yaml')
    old = read(ABLATION / 'preflight.json')
    raw_protocol = read(SOURCE / 'protocol.json')
    for name,key in [('train_transaction.csv','raw_transaction_sha256'),('train_identity.csv','raw_identity_sha256')]:
        if sha(PROJECT/'data/raw'/name)!=raw_protocol[key]:
            raise RuntimeError('Raw source changed: '+name)
    ft = cfg['FT'] | {'microbatch': 128, 'inference_batch': 128,
                      'gradient_clip': 1., 'sampler_recency_ratio': 4.}
    excel = cfg['Excel'] | {'epochs': 16, 'microbatch': 32,
                            'inference_batch': 128, 'gradient_clip': 1.}
    protocol = dict(name=HERE.name, started_utc=now(),
        authorization='User explicitly requested the complete input-adaptation, training and verification loop.',
        architectures=list(ARCHITECTURES), seeds=list(SEEDS), variants=list(VARIANTS),
        neural_conditions=16, new_lightgbm_conditions=['B1M_seed43', 'B1C_seed43'],
        primary_metric='average_precision',
        secondary_metrics=['roc_auc', 'recall_top_1pct', 'precision_top_1pct', 'recall_top_5pct', 'brier_score'],
        selection='Per trial best validation AP; FT >1e-6 improvement, Excel >0; ties retain earlier. Fixed full matrix, no added hyperparameter search.',
        feature_policy='Frozen train_only FraudSquad arrays. Whole-training-partition statistics, frozen validation maps; NOT prefix-causal or a streaming evaluation.',
        preprocessing='Train-only numeric median and quantile-normal transform. Missing flags for training-observed missing numeric fields and all identity/UID numeric fields. Categorical train vocab min_count2/max50000; unknown0/missing1 and explicit indicators. No raw UID/TransactionID input tokens.',
        excel_preprocessing='Five chronological training label blocks: first block constant0.5, later blocks encoded using only prior blocks; frozen all-training encoding for validation; training-only MI order/weights on fixed20k rows.',
        samplers='Same within architecture/seed across all four feature conditions; CPU index generation only. CUDA model, tensors and optimizer.',
        ft=ft, excel=excel, lightgbm=old['lgb_config'],
        boundaries=dict(train_rows=413378, validation_rows=88581, test_accessed=False,
                        official_test_loaded=False, independently_heldout_evaluation=False),
        resource_policy='Sequential CUDA; never terminate another worker; wait for external compute to release GPU. Preserve epoch checkpoints and resume matching config/code/data only. No CPU fallback.',
        validation_strata='Frozen previous ablation UID seen/unseen and id12 observed/missing; all rows retained; within-stratum Top-K is not global-budget recall.',
        source_hashes={str(p):sha(p) for p in [ABLATION/'artifact_manifest.json',ABLATION/'feature_groups.json',ABLATION/'strata_validation.npz',SOURCE/'derived/train_only/manifest.json',SOURCE/'derived/labels_ids.npz']},
        source_seal=seal, source_arrays=src['array_hashes'],
        input_raw_hashes={key:raw_protocol[key] for key in ['raw_transaction_sha256','raw_identity_sha256']},
        reused_baselines='Six already completed matching LightGBM conditions; historical FT/Excel records kept as context, new shared-input neural controls are necessary.',
        cost='Estimate from training-only CUDA smoke before formal fits; do all16 conditions without stopping on a negative AP result.')
    write(HERE / 'protocol.json', protocol)
    import psutil
    write(HERE / 'preflight.json', dict(status='SOURCE_PASS', verified_source_files=len(seal['files']),
        available_ram_gib=psutil.virtual_memory().available/2**30,
        free_disk_gib=shutil.disk_usage(HERE).free/2**30,
        packages={name:importlib.metadata.version(name) for name in ['torch','numpy','pandas','lightgbm','scikit-learn','rtdl-revisiting-models','rtdl-num-embeddings','libauc','category-encoders']},
        gpu=snapshot_gpu(), created_utc=now(), test_accessed=False))
    print('PROTOCOL_FROZEN', protocol['neural_conditions'], 'neural trials and2 new tree controls', flush=True)
    return protocol


if __name__ == '__main__':
    freeze()
