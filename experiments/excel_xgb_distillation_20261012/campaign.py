from __future__ import annotations

import argparse
import traceback

from .common import *


def prepare():
    from .contracts import cpu_checks
    from .teacher import prepare_features
    protocol = freeze()
    original = verify_original(protocol)
    cpu_checks()
    prepare_features()
    text = '''# Diagnosis before continuation

The frozen ExcelFormer B1C parent reaches AP 0.601916 and 822/3042 frauds at the top-886 investigation budget. Its best checkpoint occurs at the last original epoch (16); additional training is a plausible competing explanation for any gain. The experiment therefore starts both arms at the same parent with fresh AdamW, matched uniform sampling, original MI-weighted Hid-Mix, and identical five-epoch limits. Only the Bernoulli teacher-loss coefficient differs.

The existing full-TRAIN XGBoost AP 0.671557 motivates a teacher recipe but is not reused for in-sample TRAIN supervision. Three past-prefix teachers independently fit FraudSquad feature maps; the first TRAIN quarter retains hard supervision without a teacher. Student features retain whole-TRAIN unsupervised maps, so this remains TRAIN-only batch evaluation rather than strict streaming preprocessing.

The rich_all LightGBM reference AP 0.608338 uses different features. The original stronger tree and B1C matched tree remain reported. Main selection uses reused validation AP; recall changes, negative results, GPU costs and hashes are preserved. No heldout test is accessed and no claim is made about guaranteed superiority.
'''
    (HERE / 'diagnosis.md').write_text(text, encoding='utf-8')
    write(HERE / 'preparation.json', {'status': 'PASS', 'original': original, 'execution_code': execution_hashes(), 'finished_utc': now()})
    print('PREPARATION_PASS', flush=True)


def all_stages():
    from .contracts import cuda_smoke
    from .teacher import cuda_teachers
    from .training import run
    from .verify import cuda_verify
    from .report import make_report
    prepare()
    budget = Budget()
    if not (HERE/'smoke.json').exists() or not (HERE/'teacher/receipt.json').exists():
        # Smoke and teacher preparation are one short admission block; release
        # before either complete student trial so existing owners rotate fairly.
        with gpu_lease('preflight_and_past_prefix_teachers') as resource:
            if not (HERE/'smoke.json').exists():
                with budget.phase('smoke'):
                    cuda_smoke(budget, resource)
            if not (HERE/'teacher/receipt.json').exists():
                with budget.phase('teacher'):
                    cuda_teachers(budget, resource)
            cuda_cleanup()
    for trial in ('bce_control', 'xgb_kd_lambda1'):
        if (HERE/'runs'/trial/'result.json').exists():
            continue
        write(HERE/'campaign_state.json', {'status': 'RUNNING', 'current': trial, 'updated_utc': now()})
        with gpu_lease(trial) as resource:
            with budget.phase(trial):
                run(trial, budget, resource)
            if trial == 'xgb_kd_lambda1' and not (HERE/'verification.json').exists():
                with budget.phase('verification'):
                    cuda_verify(budget, resource)
            cuda_cleanup()
    if not (HERE/'verification.json').exists():
        with gpu_lease('final_checkpoint_replay') as resource:
            with budget.phase('verification'):
                cuda_verify(budget, resource)
    decision = make_report()
    write(HERE/'campaign_state.json', {'status': 'COMPLETE', 'decision': decision, 'finished_utc': now()})
    # The final manifest includes the completion state.
    write(HERE/'artifact_manifest.json', {'created_utc': now(), 'files': artifact_hashes(HERE)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['all', 'prepare', 'verify', 'report'])
    action = parser.parse_args().action
    try:
        if action == 'all':
            all_stages()
        elif action == 'prepare':
            prepare()
        elif action == 'verify':
            from .verify import cuda_verify
            verify_original()
            if (HERE/'verification.json').exists():
                from .verify import cpu_verify
                cpu_verify()
                print('REUSE_FRESH_REPLAY_WITH_CPU_REVERIFICATION', flush=True)
            else:
                budget = Budget()
                with gpu_lease('final_checkpoint_replay') as resource:
                    with budget.phase('verification'):
                        cuda_verify(budget, resource)
        else:
            from .report import make_report
            make_report()
    except BaseException as error:
        record = {'status': 'FAILED', 'action': action, 'reason': str(error), 'traceback': traceback.format_exc(), 'finished_utc': now()}
        directory = HERE/'failures'
        directory.mkdir(exist_ok=True)
        write(directory / (now().replace(':','-')+'.json'), record)
        write(HERE/'campaign_state.json', record)
        print('CAMPAIGN_FAILED', record['reason'], flush=True)
        raise


if __name__ == '__main__':
    main()
