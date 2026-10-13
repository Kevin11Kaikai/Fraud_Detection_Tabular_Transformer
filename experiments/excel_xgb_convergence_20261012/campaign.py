"""Run authorized paired extensions, fresh replays, and complete result delivery."""
from __future__ import annotations

import traceback

from .common import *


def main():
    from .train import completed_histories,pair_rule,train_chunk,fresh_verify
    from .report import deliver
    plan = protocol()
    if (HERE/'campaign_state.json').exists():
        state = read(HERE/'campaign_state.json')
        if state['status'] == 'COMPLETE':
            print('REUSE_COMPLETE_EXTENSION',state['decision'],flush=True)
            return
        if state.get('pid') != os.getpid() and fifo.alive(state):
            raise RuntimeError('Another extension worker is already alive')
    budget = Budget()
    reason = 'ADDITIONAL_EPOCH_CAP'
    try:
        while True:
            rows = completed_histories()
            rule,stop = pair_rule(rows,plan)
            if stop:
                reason = 'BOTH_ARMS_AP_PLATEAU_RULE'
                break
            if rule.completed >= plan['additional_epochs_max']:
                reason = 'ADDITIONAL_EPOCH_CAP'
                break
            estimate = sum(float(np.median([r['epoch_seconds'] for r in v[-5:]])) if v else 435. for v in rows.values())
            if budget.remaining() < 1.5*estimate+plan['verification_reserve_seconds']:
                reason = 'GPU_BUDGET_STOP_BEFORE_PAIRED_EPOCH'
                break
            purpose = f'paired_extension_epochs_{rule.completed+6:02d}_to_{min(25,rule.completed+10):02d}'
            reservation = min(budget.remaining()-plan['verification_reserve_seconds'],
                1.4*estimate*plan['gpu_lease_chunk_pairs']+300.)
            write(HERE/'campaign_state.json',dict(status='WAITING_GPU_FIFO',current=purpose,
                additional_common_epochs=rule.completed,pid=os.getpid(),process_created=psutil.Process().create_time(),updated_utc=now()))
            with gpu_lease(purpose) as resource:
                with budget.window(purpose,resource,reservation):
                    reason = train_chunk(plan,budget,resource)
            if reason != 'NEXT_CHUNK':
                break
        rows = completed_histories()
        common = min(len(v) for v in rows.values())
        if common == 0:
            raise RuntimeError('Budget does not permit a complete new paired epoch')
        if len(rows[TRIALS[0]]) != len(rows[TRIALS[1]]):
            raise RuntimeError('Cannot complete delivery with unequal new epoch counts')
        write(HERE/'stopping.json',dict(reason=reason,additional_common_epochs=common,
            validation_stability_rule_satisfied=reason=='BOTH_ARMS_AP_PLATEAU_RULE',
            mathematical_convergence_established=False,finished_utc=now()))
        if not (HERE/'fresh_verification.json').exists():
            write(HERE/'campaign_state.json',dict(status='VERIFYING',current='fresh_full_validation_replays',
                additional_common_epochs=common,pid=os.getpid(),process_created=psutil.Process().create_time(),updated_utc=now()))
            with gpu_lease('fresh_full_validation_replays') as resource:
                with budget.window('fresh_full_validation_replays',resource,plan['verification_reserve_seconds']):
                    fresh_verify(plan,budget,resource)
        decision = deliver(plan,budget)
        write(HERE/'campaign_state.json',dict(status='COMPLETE',decision=decision,finished_utc=now()))
        seal = {str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file()
            and p.name not in ('artifact_manifest.json','worker.stdout.log','worker.stderr.log','resource_status.json')
            and p.suffix not in ('.tmp','.pyc') and '__pycache__' not in p.parts}
        write(HERE/'artifact_manifest.json',dict(status='SEALED',files=seal,test_accessed=False,finished_utc=now()))
        print('PAIRED_EXTENSION_COMPLETE',decision,flush=True)
    except BaseException as error:
        record = dict(status='FAILED',reason=str(error),traceback=traceback.format_exc(),
            completed_history_counts={t:len(v) for t,v in completed_histories().items()},finished_utc=now())
        write(HERE/'failure.json',record)
        write(HERE/'campaign_state.json',record)
        print('PAIRED_EXTENSION_FAILED',record['reason'],flush=True)
        raise


if __name__ == '__main__':
    main()
