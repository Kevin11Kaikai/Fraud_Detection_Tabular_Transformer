"""Rescore every complete epoch and deliver matched extended-training evidence."""
from __future__ import annotations

import shutil
from datetime import datetime

import pandas as pd

from .common import *
from ..identity_uid_transformer_transfer_20261011.analyze import numpy_scores


def deliver(plan,budget):
    from .train import completed_histories,pair_rule
    rows = completed_histories()
    common = min(len(v) for v in rows.values())
    if common == 0 or any(len(v) != common for v in rows.values()):
        raise RuntimeError('Delivery requires equal nonzero new training epochs')
    rule,plateau = pair_rule(rows,plan)
    parts,_ = load_inputs()
    va = parts['validation']
    records,epoch_scores,selected_predictions = {},[],{}
    replay = read(HERE/'fresh_verification.json')
    for trial in TRIALS:
        root = HERE/'runs'/trial
        for row in rows[trial]:
            epoch = row['epoch']
            checkpoint = root/f'epoch_{epoch:02d}.pt'
            predictions = root/f'predictions_epoch_{epoch:02d}.npz'
            if sha(checkpoint) != row['checkpoint_sha256'] or sha(predictions) != row['prediction_sha256']:
                raise RuntimeError('Complete epoch artifact changed')
            with np.load(predictions) as saved:
                np.testing.assert_array_equal(saved['ids'],va['ids'])
                np.testing.assert_array_equal(saved['y'],va['y'])
                scores = metrics(saved['y'],saved['p'],saved['ids'])
                other = numpy_scores(saved['y'],saved['p'],saved['ids'])
            for key,value in other.items():
                np.testing.assert_allclose(scores[key],value,rtol=0,atol=1e-12)
                np.testing.assert_allclose(scores[key],row[key],rtol=0,atol=1e-12)
            epoch_scores.append(dict(trial=trial,**row))
        with np.load(root/'predictions_validation.npz') as saved:
            np.testing.assert_array_equal(saved['ids'],va['ids'])
            np.testing.assert_array_equal(saved['y'],va['y'])
            p = saved['p'].copy()
        selected_predictions[trial] = p
        scores = metrics(va['y'],p,va['ids'])
        for key,value in numpy_scores(va['y'],p,va['ids']).items():
            np.testing.assert_allclose(scores[key],value,rtol=0,atol=1e-12)
            np.testing.assert_allclose(scores[key],replay['records'][trial]['metrics'][key],rtol=0,atol=1e-12)
        if replay['records'][trial]['checkpoint_sha256'] != sha(root/'best.pt'):
            raise RuntimeError('Fresh replay checkpoint changed')
        result = dict(status='COMPLETE',trial=trial,metrics=scores,
            source_epoch=5,additional_completed_epochs=common,total_continuation_epochs=5+common,
            selected_epoch=replay['records'][trial]['epoch'],new_training_seconds=sum(r['training_seconds'] for r in rows[trial]),
            source_ap=plan['source_parents'][trial]['metrics']['average_precision'],
            checkpoint_sha256=sha(root/'best.pt'),prediction_sha256=sha(root/'predictions_validation.npz'),
            optimizer_reset=False,fresh_model_checkpoint_reload_pass=True,
            reload_max_abs_error=replay['records'][trial]['fresh_reload_max_abs_error'],
            execution_fingerprints=plan['execution_fingerprints'],protocol_sha256=sha(HERE/'protocol.json'),test_accessed=False,
            finished_utc=now())
        write(root/'result.json',result)
        write(root/'progress.json',dict(status='COMPLETE',trial=trial,metrics=scores,
            additional_completed_epochs=common,selected_epoch=result['selected_epoch'],updated_utc=now()))
        records[trial] = result
    pd.DataFrame(epoch_scores).to_csv(HERE/'all_epoch_metrics.csv',index=False)
    initial = [{ 'model':t+'_source_epoch5', 'selected_epoch':5, **plan['source_parents'][t]['metrics']} for t in TRIALS]
    final = [dict(model=t,selected_epoch=r['selected_epoch'],**r['metrics']) for t,r in records.items()]
    pd.DataFrame(initial+final).to_csv(HERE/'results.csv',index=False)
    original.verify_original(read(SOURCE/'protocol.json'))
    ready = read(HERE/'preparation.json')
    for relative,expected in ready['source_artifact_manifest']['files'].items():
        if sha(SOURCE/relative) != expected:
            raise RuntimeError('Original five-epoch sealed evidence changed: '+relative)
    if sha(SOURCE/'budget.json') != ready['source_budget_sha256']:
        raise RuntimeError('Original GPU ledger changed')
    if read(original.TRANSFER/'campaign_state.json')['status'] != 'PAUSED':
        raise RuntimeError('Original Excel matrix resumed unexpectedly')
    strata_path = SOURCE.parent/'identity_uid_ablation_20261011/strata_validation.npz'
    with np.load(strata_path) as strata:
        np.testing.assert_array_equal(strata['ids'],va['ids'])
        masks = dict(uid_seen=strata['uid_seen'].copy(),uid_unseen=~strata['uid_seen'],
            identity_observed=strata['identity_marker_observed'].copy(),identity_missing=~strata['identity_marker_observed'])
    order = np.argsort(va['dt'],kind='stable')
    for number,index in enumerate(np.array_split(order,5),1):
        mask = np.zeros(len(order),dtype=bool)
        mask[index] = True
        masks[f'validation_time_block_{number}'] = mask
    groups = []
    for trial,p in selected_predictions.items():
        with np.load(SOURCE/'runs'/trial/'predictions_validation.npz') as source_prediction:
            np.testing.assert_array_equal(source_prediction['ids'],va['ids'])
            source_p = source_prediction['p'].copy()
        for phase,predictions in (('source_epoch5',source_p),('extended_selected',p)):
            for name,mask in masks.items():
                groups.append(dict(trial=trial,phase=phase,stratum=name,
                    **metrics(va['y'][mask],predictions[mask],va['ids'][mask])))
    pd.DataFrame(groups).to_csv(HERE/'temporal_uid_metrics.csv',index=False)
    stopping = read(HERE/'stopping.json')
    bce,kd = (records[t]['metrics'] for t in TRIALS)
    selected = max(TRIALS,key=lambda t:records[t]['metrics']['average_precision'])
    shutil.copyfile(HERE/'runs'/selected/'best.pt',HERE/'selected_model.pt')
    decision = dict(status='COMPLETE_PAIRED_EXTENDED_TRAINING_AND_VERIFICATION',
        selected_model=selected,selected_epoch=records[selected]['selected_epoch'],
        selected_ap=records[selected]['metrics']['average_precision'],
        selected_checkpoint_sha256=sha(HERE/'selected_model.pt'),
        additional_common_epochs=common,total_continuation_epochs=5+common,
        original_parent_training_epochs=16,total_model_training_epochs=21+common,
        stopping_reason=stopping['reason'],validation_stability_rule_satisfied=plateau,
        mathematical_convergence_established=False,
        distillation_minus_bce_ap=kd['average_precision']-bce['average_precision'],
        exceeds_original_strong_lightgbm=records[selected]['metrics']['average_precision']>plan['strong_lightgbm_ap'],
        exceeds_matched_B1C_lightgbm=records[selected]['metrics']['average_precision']>plan['matched_B1C_lightgbm_ap'],
        paired_metrics={t:r['metrics'] for t,r in records.items()},
        additional_active_gpu_seconds=budget.state['active_gpu_seconds'],
        prior_active_gpu_seconds=plan['original_active_gpu_seconds'],
        total_related_active_gpu_seconds=budget.state['active_gpu_seconds']+plan['original_active_gpu_seconds'],
        additional_queue_wait_seconds=budget.state['queue_wait_seconds'],
        peak_reserved_gib=max(read(p)['peak_reserved_gib'] for p in (HERE/'windows').glob('*.json')),
        development_validation_only=True,test_accessed=False,original_excel_matrix_remains_paused=True,finished_utc=now())
    write(HERE/'FINAL_DECISION.json',decision)
    verification = dict(status='PASS',every_new_complete_epoch_numpy_rescore=True,
        full_validation_rows=88581,full_validation_fraud=3042,fresh_model_replays=2,
        paired_sampler_and_mix_streams='PASS',restored_optimizer_and_rng='PASS',
        original_five_epoch_evidence_files_unchanged=len(ready['source_artifact_manifest']['files']),
        old_budget_preserved=True,original_excel_matrix_remains_paused=True,
        independent_replication_performed=False,test_accessed=False,finished_utc=now())
    write(HERE/'verification.json',verification)
    write_report(plan,decision,records,ready,rule.state())
    plot_curves(rows,plan)
    print('EXTENDED_EVERY_EPOCH_RESCORE_PASS',len(epoch_scores),flush=True)
    return decision


def write_report(plan,decision,records,ready,patience):
    lines = ['# Paired extended training: BCE and XGBoost distillation', '',
        f"Both arms resumed from their complete epoch-5 checkpoints, adding {decision['additional_common_epochs']} matched epochs; total continuation: {decision['total_continuation_epochs']} epochs, or {decision['total_model_training_epochs']} epochs including the parent. Model weights, AdamW momentum and random states were restored without resetting the optimizer.", '',
        f"Stopping reason: {decision['stopping_reason']}. Validation stability rule satisfied: {decision['validation_stability_rule_satisfied']}; this does not establish mathematical convergence.", '',
        '| Model | Starting AP | Selected AP after extension | Selected continuation epoch | Fraud found at Top1% | Brier |',
        '|---|---:|---:|---:|---:|---:|']
    for trial in TRIALS:
        r=records[trial];m=r['metrics']
        lines.append(f"| {trial} | {r['source_ap']:.6f} | {m['average_precision']:.6f} | {r['selected_epoch']} | {m['captured_top_1pct']} | {m['brier_score']:.6f} |")
    lines += ['',f"The distillation-minus-BCE AP difference was {decision['distillation_minus_bce_ap']:+.6f}; all epochs and negative results are retained. The delivered model is {decision['selected_model']}, AP={decision['selected_ap']:.6f}.", '',
        f"Original strong LightGBM AP={plan['strong_lightgbm_ap']:.6f}; the strong B1C same-input reference AP={plan['matched_B1C_lightgbm_ap']:.6f}. Exceeded, respectively: {decision['exceeds_original_strong_lightgbm']} / {decision['exceeds_matched_B1C_lightgbm']}.", '',
        'Both arms used identical inputs, 335 tokens, seed42, effective batch256/microbatch32, BF16, AdamW at fixed learning rate3e-5, the original MI-weighted mixing, past-prefix teachers and temperature1. Only the teacher-loss coefficient differed (0 or1). Sampling and mixing hashes matched at every epoch.', '',
        'Stopping rules were frozen before new validation scores: run at least 10 additional matched epochs, then stop both arms only after each has 8 consecutive epochs without an AP gain greater than 0.0001; otherwise run to the20-epoch extension cap or GPU budget cap. Selection for each arm also includes its original epoch5.', '',
        f"Added GPU time: {decision['additional_active_gpu_seconds']/60:.2f} minutes; added queue wait: {decision['additional_queue_wait_seconds']/60:.2f} minutes; prior-stage GPU time: {decision['prior_active_gpu_seconds']/60:.2f} minutes; total: {decision['total_related_active_gpu_seconds']/60:.2f} minutes. The old ledger was preserved; the added cap was {plan['maximum_additional_active_gpu_seconds']/3600:.2f} hours.", '',
        'Fresh models reloaded both selected checkpoints and reran inference on all 88,581 validation rows. AP, ROC, Top-K and Brier for every new complete epoch were recomputed using separate NumPy formulas. Original sealed evidence and budgets were preserved.', '',
        'Neither the final test split nor the official unlabeled test was accessed in this campaign. Validation was reused for selection, so results are single-seed development evidence. Matched stopping conditions and complete execution do not establish theoretical convergence, independent generalization or deployment guarantees. Input maps fitted on the entire training partition are not strictly past-only streaming preprocessing.', '',
        'Unseen UID, missing identity and all five chronological validation blocks are retained in temporal_uid_metrics.csv. Stratified Top-K uses within-group budgets. Other Excel experiments remain paused as requested and are not resumed automatically.', '',
        '## Reproduction and checkpoint recovery', '', 'Use the existing environment from the original fraud-detection-mvp root:', '', '```powershell',
        r'& .\.venv\Scripts\python.exe -X utf8 -B -u -m experiments.excel_xgb_convergence_20261012.campaign', '```', '',
        'Reuse completed experiments. Resume incomplete work with model, AdamW and RNG states from a complete epoch. If one arm has finished a paired epoch, complete only the other arm without repeating the first update. Stop on code, input or parent-model hash mismatch; charge interrupted GPU windows conservatively without resetting budgets.']
    (HERE/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def plot_curves(rows,plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for trial,color in zip(TRIALS,('#2563a6','#bc7525')):
        original_rows=pd.read_csv(SOURCE/'runs'/trial/'history.csv').to_dict('records')
        combined=original_rows+rows[trial]
        epoch=[r['epoch'] for r in combined]
        axes[0].plot(epoch,[r['average_precision'] for r in combined],marker='o',markersize=3,color=color,label=trial)
        axes[1].plot(epoch,[r['hard_loss'] for r in combined],color=color,label=trial)
        axes[2].plot(epoch,[r['soft_loss'] for r in combined],color=color,label=trial)
    axes[0].axhline(plan['strong_lightgbm_ap'],color='#27745d',linestyle='--',label='Original strong LightGBM')
    for ax,title in zip(axes,('Development validation AP','Hard-label training loss','Teacher training loss')):
        ax.axvline(5.5,color='#64748b',linestyle=':',label='Extension starts' if ax==axes[0] else None)
        ax.set(xlabel='Continuation epoch after the 16-epoch parent',title=title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Matched continuation; optimizer and RNG preserved',fontsize=14)
    fig.text(.5,-.015,'Validation AP stability is an operational stopping rule, not mathematical convergence or independent test evidence.',ha='center',fontsize=9)
    fig.savefig(HERE/'learning_curves.png',dpi=180,bbox_inches='tight')
    fig.savefig(HERE/'learning_curves.svg',bbox_inches='tight')
    plt.close(fig)
