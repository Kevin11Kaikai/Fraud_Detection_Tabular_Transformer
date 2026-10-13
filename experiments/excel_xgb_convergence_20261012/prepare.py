"""Read-only source verification and CPU resume checks before GPU authorization."""
from __future__ import annotations

import argparse

from .common import *


def prepare():
    import torch
    source_protocol = read(SOURCE/'protocol.json')
    if source_protocol['execution_code'] != original.execution_hashes():
        raise RuntimeError('Original five-epoch implementation changed')
    immutable = original.verify_original(source_protocol)
    seal = read(SOURCE/'artifact_manifest.json')
    for relative, expected in seal['files'].items():
        if sha(SOURCE/relative) != expected:
            raise RuntimeError('Original five-epoch sealed artifact changed: '+relative)
    source_files = {str(SOURCE/'artifact_manifest.json'):sha(SOURCE/'artifact_manifest.json'),
        str(SOURCE/'protocol.json'):sha(SOURCE/'protocol.json'),
        str(SOURCE/'budget.json'):sha(SOURCE/'budget.json'),
        str(SOURCE/'teacher/receipt.json'):sha(SOURCE/'teacher/receipt.json'),
        str(SOURCE/'teacher/soft_targets_train.npz'):sha(SOURCE/'teacher/soft_targets_train.npz')}
    parents, capsules = {}, {}
    for trial in TRIALS:
        root = SOURCE/'runs'/trial
        cp = torch.load(root/'last.pt', map_location='cpu', weights_only=False)
        best = torch.load(root/'epoch_05.pt', map_location='cpu', weights_only=False)
        result = read(root/'result.json')
        if cp['epoch'] != 5 or result['best_epoch'] != 5 or result['completed_epochs'] != 5:
            raise RuntimeError('The assumed epoch-five resume point changed')
        if cp['execution_code'] != original.execution_hashes() or cp['config'] != result['config']:
            raise RuntimeError('Source checkpoint metadata changed')
        if cp['parent_sha256'] != source_protocol['parent_checkpoint_sha256']:
            raise RuntimeError('Original parent identity changed')
        if not all(torch.equal(cp['model'][name], value) for name,value in best['model'].items()):
            raise RuntimeError('Last checkpoint is not the selected epoch-five model')
        steps = [float(state['step']) for state in cp['optimizer']['state'].values() if 'step' in state]
        if not steps or min(steps) != 8075 or max(steps) != 8075:
            raise RuntimeError('AdamW continuation state is missing or has unexpected steps')
        if any(group['lr'] != 3e-5 for group in cp['optimizer']['param_groups']):
            raise RuntimeError('Source optimizer learning rate changed')
        capsules[trial] = cp['rng']
        parents[trial] = dict(source_epoch=5, checkpoint=str(root/'last.pt'), checkpoint_sha256=sha(root/'last.pt'),
            selected_epoch_checkpoint=str(root/'epoch_05.pt'), selected_prediction=str(root/'predictions_validation.npz'),
            metrics=result['metrics'], optimizer_steps=8075, optimizer_reset=False, rng_restored=True)
        for name in ('last.pt','epoch_05.pt','predictions_validation.npz','history.csv','result.json'):
            source_files[str(root/name)] = sha(root/name)
    a,b = (capsules[t] for t in TRIALS)
    if not torch.equal(a['torch'],b['torch']) or len(a['cuda']) != len(b['cuda']) or not all(torch.equal(x,y) for x,y in zip(a['cuda'],b['cuda'])):
        raise RuntimeError('Two source RNG streams no longer match')
    np.testing.assert_equal(a['numpy'],b['numpy'])
    if a['python'] != b['python']:
        raise RuntimeError('Source Python RNG streams differ')
    parts,schema = load_inputs()
    from ..identity_uid_transformer_transfer_20261011 import models
    for trial in TRIALS:
        cp = torch.load(SOURCE/'runs'/trial/'last.pt',map_location='cpu',weights_only=False)
        model = models.build('excel',schema,cp['config'])
        model.load_state_dict(cp['model'])
        opt = torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=0.)
        opt.load_state_dict(cp['optimizer'])
        restored = opt.state_dict()
        if restored['param_groups'] != cp['optimizer']['param_groups']:
            raise RuntimeError('CPU AdamW parameter groups were not preserved')
        if set(restored['state']) != set(cp['optimizer']['state']):
            raise RuntimeError('CPU AdamW parameter bindings changed')
        for key,state in cp['optimizer']['state'].items():
            for name,value in state.items():
                other = restored['state'][key][name]
                if torch.is_tensor(value):
                    if not torch.equal(value,other):
                        raise RuntimeError('CPU AdamW moment/step restoration differs')
                elif value != other:
                    raise RuntimeError('CPU AdamW state restoration differs')
        if not all(torch.equal(model.state_dict()[key],value) for key,value in cp['model'].items()):
            raise RuntimeError('CPU model reconstruction differs')
        del model,opt,cp
    strata = SOURCE.parent/'identity_uid_ablation_20261011/strata_validation.npz'
    with np.load(strata) as groups:
        np.testing.assert_array_equal(groups['ids'],parts['validation']['ids'])
    source_files[str(strata)] = sha(strata)
    teacher_receipt = read(SOURCE/'teacher/receipt.json')
    with np.load(SOURCE/'teacher/soft_targets_train.npz') as targets:
        np.testing.assert_array_equal(targets['ids'],parts['train']['ids'])
        np.testing.assert_array_equal(targets['dt'],parts['train']['dt'])
        if int(targets['available'].sum()) != 310035 or not np.isfinite(targets['q']).all():
            raise RuntimeError('Teacher coverage or probability validity changed')
    if sha(SOURCE/'teacher/soft_targets_train.npz') != teacher_receipt['files']['soft_targets_train.npz']:
        raise RuntimeError('Frozen teacher labels changed')
    if read(original.TRANSFER/'campaign_state.json')['status'] != 'PAUSED':
        raise RuntimeError('Original Excel matrix was expected to remain paused')
    value = dict(status='CPU_RESUME_CHECKS_PASS_PENDING_BUDGET_FREEZE',
        verified_original_files=immutable['verified_files'], verified_five_epoch_sealed_files=len(seal['files']),
        parents=parents, source_files=source_files, source_artifact_manifest=seal,
        source_protocol=source_protocol, teacher_rows=310035, hard_only_rows=103343,
        matching_source_cpu_cuda_rng_streams=True, train_rows=413378, validation_rows=88581,
        model_cpu_reconstruction_exact=True, optimizer_cpu_restoration_exact=True,
        input_tokens=335, input_hash=source_protocol['input_hash'],
        prior_active_gpu_seconds=read(SOURCE/'budget.json')['active_gpu_seconds'],
        source_budget_sha256=sha(SOURCE/'budget.json'), test_accessed=False, finished_utc=now())
    write(HERE/'preparation.json', value)
    write(HERE/'proposal.json', dict(status='PREPARED_PENDING_GPU_BUDGET_AUTHORIZATION',
        additional_epochs_max=20, minimum_additional_epochs=10, patience=8, min_delta=1e-4,
        stopping='Both arms finish identical new epochs; stop when both plateau after the minimum, or hit the cap/budget',
        resume_source_epoch=5, final_continuation_epoch_cap=25, final_parent_training_epoch_cap=41,
        lr=3e-5, optimizer_reset=False, teacher_lambda={'bce_control':0.,'xgb_kd_lambda1':1.},
        suggested_additional_gpu_hours=6, forecast_additional_gpu_hours=[4.5,5.],
        prior_budget_preserved=True, original_excel_matrix_remains_paused=True, created_utc=now()))
    print('CPU_RESUME_CHECKS_PASS', {k:value[k] for k in ('verified_original_files','verified_five_epoch_sealed_files','teacher_rows','input_tokens')}, flush=True)
    return value


def freeze_budget(hours, authorization):
    if (HERE/'protocol.json').exists():
        plan = protocol()
        if plan['maximum_additional_active_gpu_seconds'] != hours*3600:
            raise RuntimeError('Do not replace an existing authorized budget')
        return plan
    ready = read(HERE/'preparation.json')
    plan = dict(campaign=OWNER, execution_authorized=True, budget_authorization=authorization,
        maximum_additional_active_gpu_seconds=hours*3600, original_active_gpu_seconds=ready['prior_active_gpu_seconds'],
        old_budget_reset=False, additional_epochs_max=20, minimum_additional_epochs=10,
        patience=8, min_delta=1e-4, gpu_lease_chunk_pairs=5, verification_reserve_seconds=900,
        resume_source_epoch=5, final_continuation_epoch_cap=25, lr=3e-5, optimizer_reset=False,
        temperature=1., teacher_lambda={'bce_control':0.,'xgb_kd_lambda1':1.},
        precision='bfloat16', effective_batch=256, microbatch=32, inference_batch=128,
        source_files=ready['source_files'], source_parents=ready['parents'],
        input_hash=ready['input_hash'], train_rows=413378, validation_rows=88581, seed=42,
        stopping_rule='After at least 10 additional common epochs, both arms must each have 8 consecutive epochs without AP gain greater than 0.0001; otherwise both continue to 20 additional epochs or the authorized budget limit.',
        selection_rule='Select each arm maximum AP including its source epoch-five checkpoint over identical complete epochs; strict improvement only, ties retain the earlier checkpoint.',
        primary_metric='average_precision', strong_lightgbm_ap=0.6338591015757505,
        matched_B1C_lightgbm_ap=0.6877080964842449,
        missing_uid_rows_retained=True, final_test_accessed=False, independent_replication_performed=False,
        original_excel_matrix_remains_paused=True, created_utc=now(), execution_fingerprints=fingerprints())
    write(HERE/'protocol.json', plan)
    write(HERE/'budget.json', dict(maximum_active_gpu_seconds=hours*3600, active_gpu_seconds=0.,
        queue_wait_seconds=0., inflight=None, events=[], prior_campaign_active_gpu_seconds=ready['prior_active_gpu_seconds'],
        old_budget_file=str(SOURCE/'budget.json'), old_budget_sha256=ready['source_budget_sha256'],
        old_budget_reset=False, created_utc=now()))
    print('EXTENSION_PROTOCOL_FROZEN', hours, 'additional GPU hours', flush=True)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--approved-gpu-hours',type=float)
    parser.add_argument('--authorization')
    args = parser.parse_args()
    prepare()
    if args.approved_gpu_hours is not None:
        if not args.authorization or args.approved_gpu_hours <= 0:
            raise RuntimeError('An explicit budget authorization is required')
        freeze_budget(args.approved_gpu_hours,args.authorization)
