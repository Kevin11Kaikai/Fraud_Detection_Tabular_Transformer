from __future__ import annotations

import numpy as np
import pandas as pd

from .common import *
from .verify import METRIC_KEYS, selected_epoch


def validate_completed_stage_reuse():
    """Read-only guards run when the CLI imports reporting, before reuse."""
    if not (HERE / 'protocol.json').exists():
        return
    protocol = freeze()
    for name in ('cpu_contracts.json', 'smoke.json'):
        if (HERE/name).exists() and read(HERE/name)['execution_code'] != execution_hashes():
            raise RuntimeError('Cached correctness receipt has different execution code')
    teacher = HERE / 'teacher/receipt.json'
    if teacher.exists():
        receipt = read(teacher)
        if receipt['protocol_sha256'] != sha(HERE/'protocol.json'):
            raise RuntimeError('Cached teachers belong to a different protocol')
        for name, expected in receipt['files'].items():
            if sha(HERE/'teacher'/name) != expected:
                raise RuntimeError('Cached teacher target receipt hash mismatch')
        for fold in receipt['folds']:
            for name, expected in fold['files'].items():
                if sha(HERE/'teacher'/f"fold{fold['fold']}"/name) != expected:
                    raise RuntimeError('Cached teacher model or feature scope changed')
    for trial in protocol['trials']:
        path = HERE/'runs'/trial/'result.json'
        if path.exists():
            result = read(path)
            expected = protocol['student_config'] | {'trial': trial, 'lambda': protocol['trials'][trial], 'temperature': 1.}
            if result['execution_code'] != execution_hashes() or result['config'] != expected:
                raise RuntimeError('Cached continuation configuration mismatch')
            for name, expected_sha in result['files'].items():
                if sha(path.parent/name) != expected_sha:
                    raise RuntimeError('Cached continuation artifact mismatch')
    auxiliary = HERE/'reference_source_audit.json'
    if auxiliary.exists():
        audit = read(auxiliary)
        if sha(audit['source_notebook_path']) != audit['expected_sha256']:
            raise RuntimeError('Pinned public source notebook changed')


validate_completed_stage_reuse()


def account_complete_lease_windows():
    """Include CUDA configuration/release overhead after all GPU work ends."""
    state = read(HERE/'budget.json')
    if state['inflight'] or not (HERE/'verification.json').exists():
        return state
    coordinator = fifo.read(fifo.ROOT/'scheduler_state.json')
    if coordinator.get('lease') and coordinator['lease']['owner'] == OWNER:
        return state
    starts, windows = {}, []
    for line in (fifo.ROOT/'events.jsonl').read_text(encoding='utf-8').splitlines():
        record = json.loads(line)
        if record.get('owner') != OWNER or 'lease_id' not in record:
            continue
        if record['event'] == 'ACQUIRED':
            starts[record['lease_id']] = record
        elif record['event'] == 'RELEASED' and record['lease_id'] in starts:
            start = starts[record['lease_id']]
            seconds = (datetime.fromisoformat(record['utc'])-datetime.fromisoformat(start['utc'])).total_seconds()
            windows.append({'lease_id': record['lease_id'], 'purpose': start['purpose'], 'seconds': seconds,
                            'waiting_seconds': start['waiting_seconds'], 'started_utc': start['utc'], 'finished_utc': record['utc']})
    wall = sum(w['seconds'] for w in windows)
    overhead = max(0., wall-state['active_gpu_seconds'])
    if overhead > 0:
        state['active_gpu_seconds'] += overhead
        state['events'].append({'name': 'lease_configuration_release_overhead', 'seconds': overhead, 'status': 'ACCOUNTED', 'finished_utc': now()})
    state['lease_windows'] = windows
    state['whole_lease_wall_seconds'] = wall
    state['total_queue_wait_seconds'] = sum(w['waiting_seconds'] for w in windows)
    if state['active_gpu_seconds'] > state['maximum_active_gpu_seconds']:
        raise RuntimeError('Whole-lease accounting exceeded the approved GPU budget')
    write(HERE/'budget.json', state)
    return state


def make_report():
    protocol = freeze()
    verification_path = HERE / 'verification.json'
    verified = read(verification_path) if verification_path.exists() else {'status': 'NOT_EXECUTED'}
    validate_completed_stage_reuse()
    budget = account_complete_lease_windows()
    rows = [{'model': 'ExcelFormer B1C parent', 'epoch': 0, 'training_seconds_new': 0., **protocol['parent_metrics']}]
    results = {}
    for trial in protocol['trials']:
        path = HERE / 'runs' / trial / 'result.json'
        if path.exists():
            result = read(path)
            results[trial] = result
            rows.append({'model': trial, 'epoch': result['best_epoch'], 'training_seconds_new': result['training_seconds'],
                         'gpu_wait_seconds': result['resource_wait_seconds'], 'peak_reserved_gib': result['peak_reserved_gib'],
                         'parameter_count': result['parameter_count'], **result['metrics']})
    cards = read(PROJECT / 'reports/model_inventory_20261012/model_cards.json')
    for ap in (.6083378935148605, .6338591015757505, .6877080964842449, .6715574513192961):
        card = next(c for c in cards if abs(c['average_precision']-ap) < 1e-12)
        rows.append({'model': card['display_name'], 'epoch': card['best_epoch'], 'training_seconds_new': 0.,
                     'information': card['information'], **{key: card[key] for key in METRIC_KEYS},
                     'captured_top_1pct': card['captured_top_1pct'], 'parameter_count': card['parameter_count']})
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / 'ablation_results.csv', index=False)
    paired = []
    completed = len(results) == 2
    common_epochs = min(r['completed_epochs'] for r in results.values()) if completed else 0
    for trial in results:
        epoch = selected_epoch(trial, common_epochs)
        with np.load(HERE / 'runs' / trial / f'predictions_epoch_{epoch:02d}.npz') as p:
            paired.append({'model': trial, 'epoch': epoch, **metrics(p['y'], p['p'], p['ids'])})
    if paired:
        pd.DataFrame(paired).to_csv(HERE / 'paired_results.csv', index=False)
    outcome = 'NOT_COMPLETED'
    kd_delta = None
    if completed:
        bce, kd = paired
        kd_delta = kd['average_precision']-bce['average_precision']
        if kd['average_precision'] > protocol['target_ap'] and kd_delta > 0 and kd['average_precision'] > protocol['parent_metrics']['average_precision']:
            outcome = 'DISTILLATION_TARGET_ACHIEVED'
        elif bce['average_precision'] > protocol['target_ap']:
            outcome = 'CONTINUATION_TARGET_ACHIEVED_DISTILLATION_GAIN_NOT_ESTABLISHED'
        elif kd_delta > 0:
            outcome = 'DISTILLATION_GAIN_OBSERVED_TARGET_NOT_ACHIEVED'
        else:
            outcome = 'DISTILLATION_GAIN_NOT_OBSERVED_TARGET_NOT_ACHIEVED'
    if verified['status'] != 'PASS':
        outcome = 'UNVERIFIED_' + outcome
    summary = {'outcome': outcome, 'verified': verified['status'], 'common_completed_epochs': common_epochs,
               'paired_results': paired, 'distillation_minus_bce_ap': kd_delta,
               'active_gpu_seconds': budget['active_gpu_seconds'], 'gpu_budget_seconds': 7200,
               'queue_wait_seconds': budget.get('total_queue_wait_seconds'),
               'active_budget_inflight': budget['inflight'], 'test_accessed': False, 'independent_future_evaluation': False,
               'single_transformer_inference': True, 'finished_utc': now()}
    if completed and verified['status'] == 'PASS':
        import shutil
        candidates = [{'model': 'parent', 'epoch': 0, **protocol['parent_metrics']}] + paired
        winner = max(candidates, key=lambda row: row['average_precision'])
        source_checkpoint = PARENT/'best.pt' if winner['model'] == 'parent' else HERE/'runs'/winner['model']/f"epoch_{winner['epoch']:02d}.pt"
        shutil.copyfile(source_checkpoint, HERE/'selected_model.pt')
        model_card = {'architecture': 'ExcelFormer', 'variant': 'B1C', 'source_run': winner['model'], 'source_epoch': winner['epoch'],
                      'checkpoint': 'selected_model.pt', 'checkpoint_sha256': sha(HERE/'selected_model.pt'),
                      'source_checkpoint_path': str(source_checkpoint), 'preprocessing_cache': str(CACHE), 'input_hash': protocol['input_hash'],
                      'input_tokens': 335, 'parameter_count': 470738, 'teacher_required_for_inference': False,
                      'metrics': {key: winner[key] for key in METRIC_KEYS}, 'development_validation_only': True, 'test_accessed': False}
        write(HERE/'selected_model_card.json', model_card)
        summary['selected_single_transformer'] = model_card
    write(HERE / 'decision.json', summary)
    parts, _ = load_inputs()
    va = parts['validation']
    strata = np.load(HERE.parent / 'identity_uid_ablation_20261011/strata_validation.npz')
    np.testing.assert_array_equal(strata['ids'], va['ids'])
    cuts = [0]+[int(np.searchsorted(va['dt'], va['dt'][int(len(va['dt'])*q)], side='left')) for q in (.2,.4,.6,.8)]+[len(va['dt'])]
    masks = {f'time_block_{i+1}': (np.arange(len(va['y']))>=a)&(np.arange(len(va['y']))<b) for i,(a,b) in enumerate(zip(cuts[:-1],cuts[1:]))}
    masks.update({'uid_seen': strata['uid_seen'], 'uid_unseen': ~strata['uid_seen'],
                  'identity_observed': strata['identity_marker_observed'], 'identity_missing': ~strata['identity_marker_observed']})
    grouped = []
    for trial in ['parent']+list(results):
        path = HERE / 'parent_replay_validation.npz' if trial == 'parent' else HERE / 'runs' / trial / 'predictions_validation.npz'
        if not path.exists():
            continue
        with np.load(path) as saved:
            for name, mask in masks.items():
                if len(np.unique(saved['y'][mask])) < 2:
                    continue
                grouped.append({'model': trial, 'group': name, **metrics(saved['y'][mask], saved['p'][mask], saved['ids'][mask])})
    pd.DataFrame(grouped).to_csv(HERE / 'temporal_uid_metrics.csv', index=False)
    plots = []
    if results:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for trial in results:
            history = pd.read_csv(HERE / 'runs' / trial / 'history.csv')
            axes[0].plot([0]+history['epoch'].tolist(), [protocol['parent_metrics']['average_precision']]+history['validation_ap'].tolist(), marker='o', label=trial)
            axes[1].plot(history['epoch'], history['hard_loss'], marker='o', label=trial)
            axes[2].plot([0]+history['epoch'].tolist(), [protocol['parent_metrics']['recall_top_1pct']]+history['recall_top_1pct'].tolist(), marker='o', label=trial)
        axes[0].axhline(protocol['target_ap'], color='black', linestyle='--', label='existing rich_all LGB reference')
        for ax, title, ylabel in zip(axes, ('Development validation AP', 'Training hard-label loss', 'Recall at top 1%'),
                                     ('AP (fraction)', 'Mean BCE', 'Recall (fraction)')):
            ax.set_title(title)
            ax.set_xlabel('Continuation epoch')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        fig.text(.5, .01, 'Source: identical 88,581 development validation rows; training loss is the mean original MI-weighted Hid-Mix BCE.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0,.05,1,1))
        fig.savefig(HERE / 'training_curves.png', dpi=160)
        plt.close(fig)
        plots.append('training_curves.png')
        group_frame = pd.DataFrame(grouped)
        fig, axes = plt.subplots(1,2,figsize=(12,4.2))
        names = ['parent']+list(results)
        for name in names:
            sub = group_frame[group_frame['model']==name].set_index('group')
            axes[0].plot(range(1,6), [sub.loc[f'time_block_{i}','average_precision'] for i in range(1,6)], marker='o', label=name)
        categories = ['uid_seen','uid_unseen','identity_observed','identity_missing']
        positions = np.arange(len(categories))
        for i,name in enumerate(names):
            sub = group_frame[group_frame['model']==name].set_index('group')
            axes[1].bar(positions+(i-1)*.24, [sub.loc[g,'average_precision'] for g in categories], width=.24, label=name)
        axes[0].set_title('AP by chronological validation block')
        axes[0].set_xlabel('Time block (early to late; timestamp groups intact)')
        axes[0].set_xticks(range(1,6))
        axes[1].set_title('AP by UID and identity strata')
        axes[1].set_xticks(positions, ['UID seen','UID unseen','Identity\nobserved','Identity\nmissing'])
        axes[1].set_xlabel('Fixed validation strata')
        for ax in axes:
            ax.set_ylabel('Average precision (fraction)')
            ax.grid(axis='y',alpha=.2)
            ax.legend(fontsize=8)
        fig.text(.5,.01,'Source: fixed development validation; all rows retained. Strata differ in prevalence and must not be treated as matched causal comparisons.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.05,1,1))
        fig.savefig(HERE/'temporal_uid_diagnostics.png',dpi=160)
        plt.close(fig)
        plots.append('temporal_uid_diagnostics.png')
    table = ['| Model | AP ↑ | Recall@Top1% ↑ | Precision@Top1% ↑ | Recall@Top5% ↑ | Brier ↓ |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        table.append(f"| {row['model']} | {row['average_precision']:.6f} | {row['recall_top_1pct']:.2%} | {row['precision_top_1pct']:.2%} | {row['recall_top_5pct']:.2%} | {row['brier_score']:.6f} |")
    teacher_notes = ''
    diagnostics_path = HERE/'teacher_diagnostics.json'
    if diagnostics_path.exists():
        diagnostics = read(diagnostics_path)['records']
        teacher_notes = 'Teacher AP on the three subsequent TRAIN blocks: '+', '.join(f"{r['oof_block_metrics']['average_precision']:.6f}" for r in diagnostics)+'; mean predicted risk: '+', '.join(f"{r['mean_teacher_probability']:.2%}" for r in diagnostics)+'; observed fraud prevalence: '+', '.join(f"{r['fraud_prevalence']:.2%}" for r in diagnostics)+'. These are descriptive OOF results from different TRAIN blocks, not entries for the main validation ranking, and were not used to select teacher epochs. Low teacher risk probabilities are an observed signal limitation; the paired results assess the net impact on the student. See teacher_diagnostics.json.'
    verdict = 'The two experiments and final verification are not yet complete.'
    contrast = ''
    if completed and verified['status'] == 'PASS':
        bce = next(r for r in paired if r['model']=='bce_control')
        kd = next(r for r in paired if r['model']=='xgb_kd_lambda1')
        verdict = {'DISTILLATION_TARGET_ACHIEVED': 'Distillation improved AP and reached the predefined target.',
                   'CONTINUATION_TARGET_ACHIEVED_DISTILLATION_GAIN_NOT_ESTABLISHED': 'The single-Transformer pipeline exceeded the predefined LightGBM reference; distillation provided no additional gain over matched BCE continuation in this study.',
                   'DISTILLATION_GAIN_OBSERVED_TARGET_NOT_ACHIEVED': 'Distillation improved on BCE, but did not reach the predefined AP reference target.',
                   'DISTILLATION_GAIN_NOT_OBSERVED_TARGET_NOT_ACHIEVED': 'The predefined AP reference target was not reached, and no additional distillation gain was observed.'}[outcome]
        contrast = f"BCE AP change relative to the original B1C parent: {bce['average_precision']-protocol['parent_metrics']['average_precision']:+.6f}; change in fraud found at Top1%: {bce['captured_top_1pct']-protocol['parent_metrics']['captured_top_1pct']:+d}. Distillation AP change relative to BCE: {kd_delta:+.6f}; change in fraud found at Top1%: {kd['captured_top_1pct']-bce['captured_top_1pct']:+d}; Brier change: {kd['brier_score']-bce['brier_score']:+.6f}. Distillation had higher ROC-AUC, but AP was the predefined primary metric; AUC was not substituted to declare a win."
    temporal_notes = ''
    if completed:
        gm = pd.DataFrame(grouped).set_index(['model','group'])
        time_ap_values = [float(gm.loc[('bce_control', f'time_block_{i}'),'average_precision']) for i in range(1,6)]
        formatted_time_ap = ', '.join(f'{value:.6f}' for value in time_ap_values)
        temporal_notes = f"BCE AP over the five chronological blocks: {formatted_time_ap}; performance still declines in later blocks. UID seen/unseen AP: {gm.loc[('bce_control','uid_seen'),'average_precision']:.6f}/{gm.loc[('bce_control','uid_unseen'),'average_precision']:.6f}; identity observed/missing AP: {gm.loc[('bce_control','identity_observed'),'average_precision']:.6f}/{gm.loc[('bce_control','identity_missing'),'average_precision']:.6f}. These groups differ in prevalence and composition, so their gaps cannot be attributed entirely to one field. Distillation AP in the last block: {gm.loc[('xgb_kd_lambda1','time_block_5'),'average_precision']:.6f}, above BCE in that block at {gm.loc[('bce_control','time_block_5'),'average_precision']:.6f}, while its overall validation AP remains lower. This is exploratory heterogeneity, not grounds for changing the primary method or evaluation roles. See temporal_uid_diagnostics.png."
    report = f'''# Paired XGBoost-to-ExcelFormer B1C distillation report

Result status: **{outcome}**. Verification status: **{verified['status']}**.

**{verdict}**

## From inputs to outputs

Raw transaction and identity covariates pass through frozen B1C preprocessing to produce 335 input tokens; ExcelFormer outputs fraud probabilities for ranking. During training, three past-prefix XGBoost teachers produce soft targets for later TRAIN blocks, mixed with the original Hid-Mix weights. XGBoost is unnecessary at inference. Teacher feature maps use their own past prefixes; the student retains its original batch TRAIN maps, so the complete pipeline is not claimed to use streaming preprocessing.

## Paired results

{chr(10).join(table)}

Both arms completed {common_epochs} matched epochs. The primary attribution comparison selects checkpoints only within this common range. Distillation-minus-BCE AP: {kd_delta}. The original checkpoint remains an epoch0 candidate so a degraded model is not delivered as an improvement.

{contrast}

The original B1C peak occurred at its final training epoch, so additional training itself could help; the BCE control isolates this gain. The full-TRAIN teacher AP0.671557 is not a guarantee of student performance and is not attributed to prefix teachers. Temperature was fixed at 1 and lambda at 1; no extra teacher, architecture or seed search was performed.

{teacher_notes}

The control measures the net effect of the complete masked distillation objective. The teacher-availability mask also changes effective loss weights for the last75% of TRAIN rows, so the result cannot be isolated as teacher knowledge or calibration. Low teacher probabilities, temporal drift across prefixes and rule-representation transfer remain hypotheses; no additional experiments tested these mechanisms.

## Comparison scope and temporal generalization

The regular rich_all LightGBM AP0.608338 is the declared pipeline reference. B1C adds identity/UID representations, so exceeding it does not demonstrate architectural superiority with identical inputs. The original strong LightGBM AP0.633859 and same-input B1C LightGBM AP0.687708 are retained without weakening or retraining baselines. See temporal_uid_metrics.csv for the five chronological validation blocks and UID/identity strata. Stratified Top-K uses within-group budgets and is not a global investigation-budget gain.

{temporal_notes}

This is development validation reused for selection; neither the remaining test labels nor the official competition test was accessed. One seed, short continuation and validation-based selection do not establish statistical significance, independent future generalization or deployment guarantees. Recall changes are fully disclosed and are not new acceptance criteria.

## Resources and verification

Active GPU lease time in this stage: {budget['active_gpu_seconds']/60:.2f} minutes, within a 120-minute cap; queue wait is recorded separately. Lease setup and release overhead are included in whole-lease costs; see budget.json for original windows. Per-trial training, wait, memory and parameter counts appear in ablation_results.csv and each result.json. Teacher CPU preparation costs appear in teacher/features_receipt.json. Historical budgets were preserved, not reset.

Cumulative queue wait: {(budget.get('total_queue_wait_seconds') or 0.)/60:.2f} minutes; both models have 470738 parameters and approximately3.09GiB peak reserved training memory. All GPU work used the existing FIFO without terminating other threads.

See verification.json for code, original-model and input hashes, matched sampling/Hid-Mix, saved-prediction rescoring and fresh-model checkpoint reloads. This was author verification, not independent third-party validation. Incomplete and failed work is retained in campaign_state.json, failure records and budget.json.

See REPRODUCTION.md for entry points. Fallback LightGBM distillation was not executed.

After comparison within the common training range, the highest-AP single Transformer is saved as selected_model.pt; its source, input cache, metrics and hashes appear in selected_model_card.json. Retaining the original checkpoint or selecting BCE continuation is not described as a distillation result.
'''
    (HERE / 'FINAL_REPORT.md').write_text(report, encoding='utf-8')
    (HERE / 'REPRODUCTION.md').write_text('''# Reproduction

From the existing fraud-detection-mvp root, reuse its existing .venv. No installation or download.

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 -B -m experiments.excel_xgb_distillation_20261012.campaign all
.\\.venv\\Scripts\\python.exe -X utf8 -B -m experiments.excel_xgb_distillation_20261012.campaign verify
.\\.venv\\Scripts\\python.exe -X utf8 -B -m experiments.excel_xgb_distillation_20261012.campaign report
```

The all command verifies fingerprints before reusing completed stages; it never resets consumed budget. GPU stages join the existing exclusive FIFO. A forced-kill recovery conservatively charges the interrupted phase reservation if its end timestamp is unknown. The fixed protocol.json lists every setting and immutable original hash. Parent checkpoints and original baselines are never modified.
''', encoding='utf-8')
    if completed:
        kd = next(r for r in paired if r['model']=='xgb_kd_lambda1')
        bce = next(r for r in paired if r['model']=='bce_control')
        selected = summary.get('selected_single_transformer', {})
        selected_metrics = selected.get('metrics', bce)
        selected_name = selected.get('source_run', 'NOT_VERIFIED')
        english = f'''Completed a controlled XGBoost-to-ExcelFormer study on chronological IEEE-CIS data. Distillation achieved development AP {kd['average_precision']:.6f}, versus {bce['average_precision']:.6f} for matched BCE continuation and 0.601916 for the parent; the distillation increment was {kd_delta:+.6f}. Three expanding TRAIN prefixes independently fitted teacher feature maps, and inference uses one 470,738-parameter Transformer. The delivered checkpoint is {selected_name}, selected for AP. Both new pipelines exceed the existing rich_all LightGBM reference (AP 0.608338) with different input features; the matched strong B1C LightGBM remains higher at AP 0.687708. Results are exploratory on reused development validation, without independent future-period evaluation.

- Built reproducible chronological tree-to-Transformer distillation with past-prefix teacher targets, paired Hid-Mix training, immutable checkpoint/data hashes, and deterministic prediction replay across 88,581 validation transactions.
- Delivered a single ExcelFormer ({selected_name}) at development AP {selected_metrics['average_precision']:.4f} and Recall@Top1% {selected_metrics['recall_top_1pct']:.2%}; completed the paired study in {budget['active_gpu_seconds']/60:.1f} active GPU minutes, preserving the negative distillation result and strong matched tree comparison.
'''
    else:
        english = 'Study not completed. No new performance or resume achievement claim is supported. See failure records.\n'
    (HERE / 'ENGLISH_PORTFOLIO.md').write_text(english, encoding='utf-8')
    write(HERE / 'artifact_manifest.json', {'created_utc': now(), 'files': artifact_hashes(HERE), 'plots': plots})
    print('REPORT_COMPLETE', summary, flush=True)
    return summary
