"""Run every frozen condition, recover fingerprints, then verify and deliver."""
import subprocess
import sys
import time

from .common import *


def execute(module,args=()):
    command=[sys.executable,'-X','utf8','-B','-u','-m',f'experiments.{HERE.name}.{module}',*map(str,args)]
    with (HERE/'commands.jsonl').open('a',encoding='utf-8') as log:
        import json
        log.write(json.dumps(dict(utc=now(),command=command,cwd=str(PROJECT)))+'\n')
    code=subprocess.call(command,cwd=PROJECT)
    if code:raise RuntimeError(f'{module} exited{code}; completed checkpoints retained')


def campaign():
    if not (HERE/'adaptation_verification.json').exists():execute('prepare')
    if not (HERE/'cpu_contracts.json').exists():execute('smoke')
    execute('train_lightgbm')
    if not (HERE/'schedule.json').exists():
        schedule={}
        for architecture,options in [('ft',[128,64]),('excel',[32,16,8])]:
            for micro in options:
                dest=HERE/f'smoke_{architecture}_{micro}.json'
                if not dest.exists():
                    try:execute('smoke',['--architecture',architecture,'--micro',micro])
                    except RuntimeError as error:
                        failure=HERE/f'smoke_failure_{architecture}_{micro}.json'
                        if not failure.exists() or read(failure).get('classification')!='RESOURCE_LIMIT':raise
                        continue
                record=read(dest)
                if record['status']=='PASS':
                    schedule[architecture]=dict(microbatch=micro,resource_smoke=dest.name,
                        maximum_epoch_training_seconds_estimate=max(r['epoch_training_seconds_estimate'] for r in record['records']))
                    break
            if architecture not in schedule:raise RuntimeError('No finite admitted CUDA configuration: '+architecture)
        forecast=8*20*schedule['ft']['maximum_epoch_training_seconds_estimate']+8*16*schedule['excel']['maximum_epoch_training_seconds_estimate']
        schedule.update(status='FROZEN_BEFORE_FORMAL_NEURAL_VALIDATION',full_matrix_forecast_training_hours=forecast/3600,
            queue_time_excluded=True,created_utc=now(),code_fingerprints=fingerprints())
        write(HERE/'schedule.json',schedule)
        print('FULL_MATRIX_FORECAST_GPU_HOURS',round(forecast/3600,2),flush=True)
    schedule=read(HERE/'schedule.json')
    if schedule['code_fingerprints']!=fingerprints():raise RuntimeError('Code changed after neural schedule freeze')
    start=time.time();state=dict(status='RUNNING',started_utc=now(),conditions={})
    if (HERE/'campaign_state.json').exists():state=read(HERE/'campaign_state.json')
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            for variant in VARIANTS:
                name=trial_name(architecture,variant,seed)
                state['current']=name;write(HERE/'campaign_state.json',state)
                execute('train',['--architecture',architecture,'--variant',variant,'--seed',seed])
                result=read(HERE/'runs'/name/'result.json')
                state['conditions'][name]=dict(status='COMPLETE',average_precision=result['metrics']['average_precision'],finished_utc=result['finished_utc'])
                write(HERE/'campaign_state.json',state)
    execute('analyze')
    state.update(status='COMPLETE',current=None,verification='PASS',finished_utc=now(),last_invocation_wall_seconds=time.time()-start)
    write(HERE/'campaign_state.json',state)
    print('COMPLETE_INPUT_TRAIN_VERIFY_LOOP',len(state['conditions']),'neural conditions',flush=True)


if __name__=='__main__':campaign()
