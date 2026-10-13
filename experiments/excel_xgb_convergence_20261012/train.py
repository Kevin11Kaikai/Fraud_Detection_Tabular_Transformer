"""Resume both objectives without resetting AdamW, RNG, inputs, or teachers."""
from __future__ import annotations

import csv
import hashlib
import math
import shutil
import time

from .common import *

import pandas as pd
import torch

from ..excel_xgb_distillation_20261012.training import update
from ..identity_uid_transformer_transfer_20261011 import models


def completed_histories():
    rows = {}
    for trial in TRIALS:
        path = HERE/'runs'/trial/'last.pt'
        if path.exists():
            cp = torch.load(path,map_location='cpu',weights_only=False)
            if cp['execution_fingerprints'] != fingerprints() or cp['protocol_sha256'] != sha(HERE/'protocol.json'):
                raise RuntimeError('Extension checkpoint fingerprint mismatch')
            rows[trial] = cp['history']
        else:
            rows[trial] = []
    return rows


def pair_rule(rows, plan):
    initial = {t:plan['source_parents'][t]['metrics']['average_precision'] for t in TRIALS}
    rule = PairPatience(initial,plan['minimum_additional_epochs'],plan['patience'],plan['min_delta'])
    common = min(len(v) for v in rows.values())
    stop = False
    for index in range(common):
        a,b = (rows[t][index] for t in TRIALS)
        if a['epoch'] != b['epoch'] or a['epoch'] != index+6:
            raise RuntimeError('Common epoch alignment mismatch')
        if a['sampler_order_sha256'] != b['sampler_order_sha256'] or a['mix_trace_sha256'] != b['mix_trace_sha256']:
            raise RuntimeError('Paired sampler or mix stream mismatch')
        stop = rule.update({t:rows[t][index]['average_precision'] for t in TRIALS})
    return rule,stop


def load_arm(trial, schema, plan):
    root = HERE/'runs'/trial
    root.mkdir(parents=True,exist_ok=True)
    source = SOURCE/'runs'/trial
    config = read(source/'config.json')
    model = models.build('excel',schema,config).cuda()
    opt = torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=0.)
    if (root/'last.pt').exists():
        cp = torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        if cp['execution_fingerprints'] != fingerprints() or cp['protocol_sha256'] != sha(HERE/'protocol.json'):
            raise RuntimeError('Extension resume source changed')
        history, best_ap, best_epoch = cp['history'],cp['best_ap'],cp['best_epoch']
        if cp['source_checkpoint_sha256'] != plan['source_parents'][trial]['checkpoint_sha256']:
            raise RuntimeError('Original epoch-five parent changed')
    else:
        cp = torch.load(source/'last.pt',map_location='cpu',weights_only=False)
        if sha(source/'last.pt') != plan['source_parents'][trial]['checkpoint_sha256'] or cp['epoch'] != 5:
            raise RuntimeError('Frozen epoch-five checkpoint changed')
        history, best_ap, best_epoch = [],cp['best_ap'],cp['best_epoch']
        shutil.copyfile(source/'epoch_05.pt',root/'best.pt')
        shutil.copyfile(source/'predictions_validation.npz',root/'predictions_validation.npz')
    model.load_state_dict(cp['model'])
    opt.load_state_dict(cp['optimizer'])
    steps = [float(s['step']) for s in opt.state.values() if 'step' in s]
    expected_steps = (5+len(history))*1615
    if not steps or min(steps) != expected_steps or max(steps) != expected_steps:
        raise RuntimeError('AdamW state was reset or does not match the completed epochs')
    if any(group['lr'] != 3e-5 for group in opt.param_groups):
        raise RuntimeError('Frozen learning rate changed')
    for state in opt.state.values():
        for name in ('exp_avg','exp_avg_sq'):
            if state[name].device.type != 'cuda':
                raise RuntimeError('AdamW moment tensors were not restored to CUDA')
    write(root/'config.json',dict(source_config=config,optimizer_reset=False,source_epoch=5,
        resume_optimizer_steps=8075,additional_epochs_cap=20,protocol_sha256=sha(HERE/'protocol.json')))
    return dict(model=model,optimizer=opt,rng=cp['rng'],history=history,
                best_ap=best_ap,best_epoch=best_epoch,root=root,config=config)


def one_epoch(trial,arm,epoch,data,plan,budget,resource):
    assert_lease(resource)
    models.restore_rng(arm['rng'])
    tick = time.perf_counter()
    model,opt = arm['model'],arm['optimizer']
    model.train()
    order = models.order('excel','pre',len(data['y']),data['train_y_cpu'],42,16+epoch)
    order_hash,trace = array_hash(order.numpy()),hashlib.sha256()
    totals,count = np.zeros(3,dtype=float),0
    train_tick = heartbeat = time.perf_counter()
    for off in range(0,len(order),256):
        indices = order[off:off+256].to('cuda')
        totals += update(model,opt,data['x'],data['y'],data['q'],data['available'],indices,
                         data['mi'],plan['teacher_lambda'][trial],trace)
        count += len(indices)
        if time.perf_counter()-heartbeat > 10:
            budget.heartbeat()
            heartbeat = time.perf_counter()
        if (off//256)%128 == 0:
            write(arm['root']/'progress.json',dict(status='TRAINING',trial=trial,epoch=epoch,
                additional_epoch=epoch-5,additional_epoch_cap=20,processed=count,total=len(order),
                best_ap=arm['best_ap'],remaining_gpu_seconds=budget.remaining(),updated_utc=now()))
        if (off//256)%256 == 0:
            assert_lease(resource)
    torch.cuda.synchronize()
    training_seconds = time.perf_counter()-train_tick
    p,inference_seconds = models.predict(model,data['vn'],None,128)
    measured = metrics(data['validation_y'],p,data['validation_ids'])
    root = arm['root']
    epoch_file = root/f'epoch_{epoch:02d}.pt'
    models.checkpoint(epoch_file,dict(model=model.state_dict(),config=arm['config'],epoch=epoch,
        trial=trial,execution_fingerprints=fingerprints(),protocol_sha256=sha(HERE/'protocol.json')))
    predictions = root/f'predictions_epoch_{epoch:02d}.npz'
    np.savez_compressed(predictions,ids=data['validation_ids'],y=data['validation_y'],p=p)
    if measured['average_precision'] > arm['best_ap']:
        arm['best_ap'],arm['best_epoch'] = measured['average_precision'],epoch
        shutil.copyfile(epoch_file,root/'best.pt')
        shutil.copyfile(predictions,root/'predictions_validation.npz')
    row = dict(epoch=epoch,additional_epoch=epoch-5,parent_epoch_offset=16+epoch,
        training_loss=float(totals[0]/count),hard_loss=float(totals[1]/count),soft_loss=float(totals[2]/count),
        training_seconds=training_seconds,inference_seconds=inference_seconds,
        epoch_seconds=time.perf_counter()-tick,sampler_order_sha256=order_hash,
        mix_trace_sha256=trace.hexdigest(),samples=count,optimizer_steps=math.ceil(count/256),
        cumulative_optimizer_steps=epoch*1615,learning_rate=3e-5,
        checkpoint_sha256=sha(epoch_file),prediction_sha256=sha(predictions),**measured)
    arm['history'].append(row)
    arm['rng'] = models.rng_state()
    # last.pt is the complete-epoch authority; regenerate the CSV on every write.
    models.checkpoint(root/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=arm['rng'],
        epoch=epoch,history=arm['history'],best_ap=arm['best_ap'],best_epoch=arm['best_epoch'],
        config=arm['config'],source_checkpoint_sha256=plan['source_parents'][trial]['checkpoint_sha256'],
        execution_fingerprints=fingerprints(),protocol_sha256=sha(HERE/'protocol.json')))
    pd.DataFrame(arm['history']).to_csv(root/'history.csv',index=False)
    write(root/'progress.json',dict(status='EPOCH_COMPLETE',trial=trial,epoch=epoch,
        additional_epoch=epoch-5,best_ap=arm['best_ap'],metrics=measured,updated_utc=now()))
    budget.heartbeat()
    if torch.cuda.max_memory_reserved() > 12*2**30:
        raise RuntimeError('Existing 12GiB allocation cap exceeded')
    print(trial,'epoch',epoch,'AP',round(measured['average_precision'],6),
          'best',round(arm['best_ap'],6),'seconds',round(row['epoch_seconds'],1),flush=True)


def train_chunk(plan,budget,resource):
    models.seed_all(42)
    torch.cuda.reset_peak_memory_stats()
    parts,schema = load_inputs()
    tr,va = parts['train'],parts['validation']
    teacher_path = SOURCE/'teacher/soft_targets_train.npz'
    with np.load(teacher_path) as targets:
        np.testing.assert_array_equal(targets['ids'],tr['ids'])
        q = torch.as_tensor(targets['q'].copy(),device='cuda')
        available = torch.as_tensor(targets['available'].copy(),device='cuda')
    data = dict(x=torch.as_tensor(np.array(tr['x_excel'],copy=True),device='cuda'),
        vn=torch.as_tensor(np.array(va['x_excel'],copy=True),device='cuda'),
        y=torch.as_tensor(np.array(tr['y'],copy=True),device='cuda',dtype=torch.float32),
        q=q,available=available,mi=torch.tensor(schema['excel']['mi_weights'],device='cuda',dtype=torch.float32),
        train_y_cpu=np.asarray(tr['y']),validation_y=va['y'],validation_ids=va['ids'])
    arms = {t:load_arm(t,schema,plan) for t in TRIALS}
    rows = {t:arm['history'] for t,arm in arms.items()}
    rule,stop = pair_rule(rows,plan)
    start = rule.completed+6
    maximum = min(26,start+plan['gpu_lease_chunk_pairs'])
    reason = 'NEXT_CHUNK'
    try:
        for epoch in range(start,maximum):
            pair_estimate = sum(float(np.median([r['epoch_seconds'] for r in arm['history'][-5:]]))
                if arm['history'] else 435. for arm in arms.values())
            if budget.remaining() < 1.4*pair_estimate+120:
                reason = 'GPU_BUDGET_STOP_BEFORE_PAIRED_EPOCH'
                break
            for trial in TRIALS:
                arm = arms[trial]
                # A partial pair after interruption resumes only its unfinished arm.
                if arm['history'] and arm['history'][-1]['epoch'] >= epoch:
                    continue
                write(HERE/'campaign_state.json',dict(status='RUNNING',current=trial,epoch=epoch,
                    additional_common_epochs=rule.completed,process_created=psutil.Process().create_time(),pid=os.getpid(),updated_utc=now()))
                one_epoch(trial,arm,epoch,data,plan,budget,resource)
            values = {t:arms[t]['history'][epoch-6]['average_precision'] for t in TRIALS}
            a,b = (arms[t]['history'][epoch-6] for t in TRIALS)
            if a['sampler_order_sha256'] != b['sampler_order_sha256'] or a['mix_trace_sha256'] != b['mix_trace_sha256']:
                raise RuntimeError('New paired sampling or mix stream mismatch')
            stop = rule.update(values)
            write(HERE/'pair_progress.json',dict(status='PAIRED_EPOCH_COMPLETE',epoch=epoch,
                additional_common_epochs=rule.completed,patience=rule.state(),ap=values,updated_utc=now()))
            if stop:
                reason = 'BOTH_ARMS_AP_PLATEAU_RULE'
                break
            if epoch == 25:
                reason = 'ADDITIONAL_EPOCH_CAP'
                break
        window = dict(stop_reason=reason,patience=rule.state(),
            common_additional_epochs=rule.completed,gpu=torch.cuda.get_device_name(0),precision='bfloat16',
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,finished_utc=now())
        write(HERE/'latest_training_window.json',window)
        write(HERE/'windows'/f'{start:02d}_to_{rule.completed+5:02d}.json',window)
        return reason
    finally:
        arms = data = q = available = None
        cuda_cleanup()


def fresh_verify(plan,budget,resource):
    parts,schema = load_inputs()
    va = parts['validation']
    models.seed_all(42)
    vn = torch.as_tensor(np.array(va['x_excel'],copy=True),device='cuda')
    records = {}
    try:
        for trial in TRIALS:
            root = HERE/'runs'/trial
            cp = torch.load(root/'best.pt',map_location='cpu',weights_only=False)
            model = models.build('excel',schema,read(SOURCE/'runs'/trial/'config.json')).cuda()
            model.load_state_dict(cp['model'])
            replay,seconds = models.predict(model,vn,None,128)
            with np.load(root/'predictions_validation.npz') as saved:
                np.testing.assert_array_equal(saved['ids'],va['ids'])
                np.testing.assert_array_equal(saved['y'],va['y'])
                error = float(np.max(np.abs(replay-saved['p'])))
                np.testing.assert_allclose(replay,saved['p'],rtol=1e-5,atol=1e-6)
            records[trial] = dict(status='PASS',epoch=cp['epoch'],fresh_reload_max_abs_error=error,
                inference_seconds=seconds,checkpoint_sha256=sha(root/'best.pt'),
                prediction_sha256=sha(root/'predictions_validation.npz'),metrics=metrics(va['y'],replay,va['ids']))
            model = None
            cuda_cleanup()
            budget.heartbeat()
        write(HERE/'fresh_verification.json',dict(status='PASS',records=records,test_accessed=False,
            fresh_models_constructed=2,finished_utc=now()))
        print('EXTENDED_FRESH_REPLAY_PASS',flush=True)
    finally:
        vn = model = None
        cuda_cleanup()
