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
    table = ['| 模型 | AP ↑ | Recall@Top1% ↑ | Precision@Top1% ↑ | Recall@Top5% ↑ | Brier ↓ |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        table.append(f"| {row['model']} | {row['average_precision']:.6f} | {row['recall_top_1pct']:.2%} | {row['precision_top_1pct']:.2%} | {row['recall_top_5pct']:.2%} | {row['brier_score']:.6f} |")
    teacher_notes = ''
    diagnostics_path = HERE/'teacher_diagnostics.json'
    if diagnostics_path.exists():
        diagnostics = read(diagnostics_path)['records']
        teacher_notes = '三个TRAIN后续块的教师AP分别为'+', '.join(f"{r['oof_block_metrics']['average_precision']:.6f}" for r in diagnostics)+'；平均预测风险分别为'+', '.join(f"{r['mean_teacher_probability']:.2%}" for r in diagnostics)+'，真实欺诈率分别为'+', '.join(f"{r['fraud_prevalence']:.2%}" for r in diagnostics)+'。这些是不同TRAIN块的描述性OOF结果，不能并入主验证表排名，也没有用于选择教师轮数。教师风险概率偏低是观察到的信号限制；它是否造成学生损失仍由配对结果评估。详见teacher_diagnostics.json。'
    verdict = '尚未完成两条实验与最终验证。'
    contrast = ''
    if completed and verified['status'] == 'PASS':
        bce = next(r for r in paired if r['model']=='bce_control')
        kd = next(r for r in paired if r['model']=='xgb_kd_lambda1')
        verdict = {'DISTILLATION_TARGET_ACHIEVED': '蒸馏增加AP并达到既定目标。',
                   'CONTINUATION_TARGET_ACHIEVED_DISTILLATION_GAIN_NOT_ESTABLISHED': '单Transformer管线超过既定LightGBM参考；本轮没有观察到蒸馏相对匹配BCE续训的额外优势。',
                   'DISTILLATION_GAIN_OBSERVED_TARGET_NOT_ACHIEVED': '观察到蒸馏相对BCE的增益，但既定AP参考目标未达到。',
                   'DISTILLATION_GAIN_NOT_OBSERVED_TARGET_NOT_ACHIEVED': '既定AP参考目标未达到，也未观察到蒸馏额外增益。'}[outcome]
        contrast = f"BCE相对原B1C的AP变化为{bce['average_precision']-protocol['parent_metrics']['average_precision']:+.6f}，Top1%抓获数变化为{bce['captured_top_1pct']-protocol['parent_metrics']['captured_top_1pct']:+d}。蒸馏相对BCE的AP变化为{kd_delta:+.6f}，Top1%抓获数变化为{kd['captured_top_1pct']-bce['captured_top_1pct']:+d}，Brier变化为{kd['brier_score']-bce['brier_score']:+.6f}。蒸馏ROC-AUC较高，但预先固定的主指标是AP，没有改用AUC宣布获胜。"
    temporal_notes = ''
    if completed:
        gm = pd.DataFrame(grouped).set_index(['model','group'])
        time_ap_values = [float(gm.loc[('bce_control', f'time_block_{i}'),'average_precision']) for i in range(1,6)]
        formatted_time_ap = ', '.join(f'{value:.6f}' for value in time_ap_values)
        temporal_notes = f"BCE的五个时间块AP依次为{formatted_time_ap}，晚期下降仍明显。UID seen/unseen的AP分别为{gm.loc[('bce_control','uid_seen'),'average_precision']:.6f}/{gm.loc[('bce_control','uid_unseen'),'average_precision']:.6f}；identity observed/missing分别为{gm.loc[('bce_control','identity_observed'),'average_precision']:.6f}/{gm.loc[('bce_control','identity_missing'),'average_precision']:.6f}。这些组的欺诈率和样本构成不同，不能将差异全部归因于某一字段。蒸馏在最后时间块AP为{gm.loc[('xgb_kd_lambda1','time_block_5'),'average_precision']:.6f}，高于BCE该块{gm.loc[('bce_control','time_block_5'),'average_precision']:.6f}，但全验证AP仍较低；这是探索性异质性观察，没有据此改换主方法或评价角色。图见temporal_uid_diagnostics.png。"
    report = f'''# XGBoost→ExcelFormer B1C 配对蒸馏报告

结果状态：**{outcome}**。验证状态：**{verified['status']}**。

**{verdict}**

## 从输入到输出

原始交易及identity协变量进入已冻结B1C预处理，生成335个token输入；ExcelFormer输出欺诈概率并排序。训练时，三个过去前缀XGBoost分别产生后续TRAIN块的软标签，随原Hid-Mix同权重混合；推理不需要XGBoost。教师的特征映射只拟合各自过去前缀，学生保留原TRAIN批量映射，因此整条管线不宣称流式预处理。

## 配对结果

{chr(10).join(table)}

两条训练共同完成{common_epochs}轮；主要归因比较只在该共同范围内选择checkpoint。蒸馏减BCE的AP差值：{kd_delta}。原checkpoint保留为epoch0候选，避免将退化模型交付为提升。

{contrast}

原B1C最优出现在原训练最后一轮，继续训练本身可能有益；BCE对照用于区分这部分收益。教师全TRAIN模型的AP0.671557不是学生性能保证，前缀教师也不冒用该指标。温度固定1，λ固定1，未搜索额外教师、架构或seed。

{teacher_notes}

本轮对照识别的是完整masked distillation目标的净效果；可用教师掩码也会改变后75%TRAIN样本的有效损失权重，不能将结果进一步归因于教师知识或校准的某个独立机制。教师概率偏低、前缀时间漂移和规则表示迁移是后续假设，本轮没有追加试验验证这些机制。

## 比较边界与时间泛化

常规rich_all LightGBM AP0.608338是已声明的管线参考，B1C增加identity/UID表示，超过该参考不等于同输入架构优越。保留原强LightGBM AP0.633859和B1C同输入LightGBM AP0.687708，未削弱或重训基线。固定五个连续验证时间块和UID/identity分层结果见temporal_uid_metrics.csv；分层Top-K使用组内预算，不能当作全局调查预算收益。

{temporal_notes}

本数据是反复用于选择的开发validation；没有访问剩余test标签或官方竞赛test。单seed、短续训和validation选择不能建立统计显著性、独立未来泛化或部署保证。召回变化完整披露，不作为新增硬验收门槛。

## 资源与验证

本轮累计GPU租约内用时{budget['active_gpu_seconds']/60:.2f}分钟，上限120分钟；排队等待另计。租约配置和释放开销也计入whole-lease成本，原始窗口见budget.json。逐trial训练/等待/显存/参数量见ablation_results.csv与各result.json。教师CPU准备成本见teacher/features_receipt.json。历史预算保持原样，未将旧训练清零。

当前累计排队等待{(budget.get('total_queue_wait_seconds') or 0.)/60:.2f}分钟；两条模型均470738个参数，训练峰值reserved显存均约3.09GiB。所有GPU工作使用现有FIFO，未终止其他线程。

代码、原模型与输入哈希、同采样及Hid-Mix配对、保存预测复算、fresh-model checkpoint重载证据见verification.json；本轮为作者自查，不称独立第三方验证。未完成和失败记录保存在campaign_state.json、failure记录及budget.json。

复现入口见REPRODUCTION.md。后备LightGBM蒸馏没有执行。

经过共同训练范围比较后，AP最高的单Transformer保存在selected_model.pt，其来源、输入缓存、指标与哈希见selected_model_card.json。如果最终保留原checkpoint或选择BCE续训，不将其称为蒸馏成果。
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
