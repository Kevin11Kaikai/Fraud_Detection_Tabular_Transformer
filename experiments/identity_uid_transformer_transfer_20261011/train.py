"""Full fixed-epoch CUDA fits with complete-epoch recovery and fresh reload."""
from __future__ import annotations

import argparse
import gc
import math
import time
import traceback

import pandas as pd
import torch

from .common import *
from . import models
from .resource import gpu_lease


def stage(architecture,variant,seed,resource):
    name=trial_name(architecture,variant,seed);root=HERE/'runs'/name;root.mkdir(parents=True,exist_ok=True)
    protocol=read(HERE/'protocol.json');cfg=protocol[architecture]|{'seed':seed,'variant':variant,'architecture':architecture}
    if (HERE/'schedule.json').exists():cfg['microbatch']=read(HERE/'schedule.json')[architecture]['microbatch']
    parts,schema=load(variant,architecture);input_hash=sha(HERE/'derived'/variant/'schema.json')
    if architecture=='excel':input_hash += ':'+sha(HERE/'derived'/variant/'excel/receipt.json')
    code=fingerprints()
    if (root/'result.json').exists():
        result=read(root/'result.json')
        if result['config']!=cfg or result['input_hash']!=input_hash or result['code_fingerprints']!=code or sha(root/'best.pt')!=result['checkpoint_sha256']:
            raise RuntimeError('Completed artifact mismatch; do not overwrite')
        print('REUSE_COMPLETE',name,flush=True);return result
    if (root/'config.json').exists() and read(root/'config.json')!=cfg:raise RuntimeError('Resume config mismatch')
    write(root/'config.json',cfg)
    models.seed_all(seed);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    tr,va=parts['train'],parts['validation'];mi=None
    key='x_num' if architecture=='ft' else 'x_excel'
    xn=torch.as_tensor(np.array(tr[key],copy=True),device='cuda');vn=torch.as_tensor(np.array(va[key],copy=True),device='cuda')
    xc=torch.as_tensor(np.array(tr['x_cat'],copy=True),device='cuda') if architecture=='ft' else None
    vc=torch.as_tensor(np.array(va['x_cat'],copy=True),device='cuda') if architecture=='ft' else None
    y=torch.as_tensor(np.array(tr['y'],copy=True),device='cuda',dtype=torch.float32)
    if architecture=='excel':mi=torch.tensor(schema['excel']['mi_weights'],device='cuda')
    model=models.build(architecture,schema,cfg).cuda();params=sum(p.numel() for p in model.parameters())
    (root/'model_structure.txt').write_text(str(model),encoding='utf-8')
    best=-1.;best_epoch=0;best_stage=None;history=[];elapsed=0.;stage='pre';next_epoch=1;patbest=-1.;bad=0
    opt=models.optimizer(model,architecture,stage,cfg)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=4,min_lr=1e-6) if architecture=='ft' else None
    if (root/'last.pt').exists():
        cp=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        if cp['config']!=cfg or cp['code']!=code or cp['input_hash']!=input_hash:raise RuntimeError('Checkpoint fingerprint mismatch')
        stage=cp['stage'];opt=models.optimizer(model,architecture,stage,cfg)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=4,min_lr=1e-6) if architecture=='ft' and stage=='pre' else None
        model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer'])
        for state in opt.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.cuda()
        if scheduler:scheduler.load_state_dict(cp['scheduler'])
        if stage=='fine':opt.T=cp.get('soap_T',0);opt.steps=cp.get('soap_steps',0)
        models.restore_rng(cp['rng']);history=cp['history'];elapsed=cp['elapsed'];best=cp['best'];best_epoch=cp['best_epoch'];best_stage=cp['best_stage'];patbest=cp['patbest'];bad=cp['bad'];next_epoch=cp['epoch']+1
        print('RESUMED',name,stage,next_epoch,flush=True)
    stop='MAX_EPOCHS'
    stages=[('pre',cfg['epochs'])]+([('fine',1)] if architecture=='ft' else [])
    for target,cap in stages:
        if target=='pre' and stage=='fine':continue
        if target=='fine' and stage=='pre':
            if sum(r['stage']=='pre' for r in history)!=cfg['epochs']:raise RuntimeError('Incomplete frozen FT pretraining')
            cp=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
            models.seed_all(seed);del model,opt;gc.collect();torch.cuda.empty_cache()
            model=models.build(architecture,schema,cfg).cuda();model.load_state_dict(cp['model'])
            opt=models.optimizer(model,architecture,'fine',cfg);scheduler=None;stage='fine';next_epoch=1
        for epoch in range(next_epoch,cap+1):
            if architecture=='excel':
                lr=cfg['lr']*min(1.,epoch/cfg['warmup'])*(.5+.5*math.cos(math.pi*max(0,epoch-2)/38))
                for group in opt.param_groups:group['lr']=lr
            tick=time.perf_counter();model.train()
            order=models.order(architecture,stage,len(y),np.asarray(tr['y']),seed,epoch)
            order_hash=array_hash(order.numpy());total=0.;count=0;bs=128 if architecture=='ft' else 256
            train_tick=time.perf_counter()
            for off in range(0,len(order),bs):
                idx=order[off:off+bs].to('cuda')
                total+=models.update(model,architecture,stage,opt,xn,xc,y,idx,cfg['microbatch'],mi);count+=len(idx)
                if (off//bs)%128==0:
                    write(root/'progress.json',dict(status='TRAINING',trial=name,stage=stage,epoch=epoch,processed=count,total=len(order),
                        best_ap=best,elapsed_training_seconds=elapsed+time.perf_counter()-tick,updated_utc=now()))
            torch.cuda.synchronize();training_seconds=time.perf_counter()-train_tick
            p,inference=models.predict(model,vn,vc,cfg['inference_batch']);m=metrics(va['y'],p,va['ids']);ap=m['average_precision']
            if scheduler:scheduler.step(ap)
            if ap>best+(1e-6 if architecture=='ft' else 0.):
                best=ap;best_epoch=epoch;best_stage=stage
                models.checkpoint(root/'best.pt',dict(model=model.state_dict(),config=cfg,stage=stage,epoch=epoch,input_hash=input_hash,code=code))
                np.savez_compressed(root/'predictions_validation.npz',ids=va['ids'],y=va['y'],p=p)
            if ap>patbest+cfg.get('min_delta',1e-4):patbest=ap;bad=0
            else:bad+=1
            seconds=time.perf_counter()-tick;elapsed+=seconds
            row=dict(stage=stage,epoch=epoch,validation_ap=ap,training_loss=total/count,epoch_seconds=seconds,training_seconds=training_seconds,
                inference_seconds=inference,sampler_order_sha256=order_hash,samples=count,optimizer_steps=math.ceil(len(order)/bs),learning_rate=opt.param_groups[0]['lr'])
            history.append(row);pd.DataFrame(history).to_csv(root/'history.csv',index=False)
            models.checkpoint(root/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict() if scheduler else None,
                config=cfg,code=code,input_hash=input_hash,stage=stage,epoch=epoch,best=best,best_epoch=best_epoch,best_stage=best_stage,
                patbest=patbest,bad=bad,history=history,elapsed=elapsed,rng=models.rng_state(),soap_T=getattr(opt,'T',0),soap_steps=getattr(opt,'steps',0)))
            print(name,stage,'epoch',epoch,'AP',round(ap,6),'best',round(best,6),'seconds',round(seconds,1),flush=True)
            if torch.cuda.max_memory_reserved()>12*2**30:raise RuntimeError('12GiB CUDA allocation cap exceeded')
            if architecture=='excel' and epoch>=cfg['min_epochs'] and bad>=cfg['patience']:
                stop='EARLY_STOPPING';break
        if stop!='MAX_EPOCHS':break
        next_epoch=1
    cp=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    del model,opt;gc.collect();torch.cuda.empty_cache()
    model=models.build(architecture,schema,cfg).cuda();model.load_state_dict(cp['model'])
    replay,inference=models.predict(model,vn,vc,cfg['inference_batch'])
    with np.load(root/'predictions_validation.npz') as z:
        if not np.array_equal(z['ids'],va['ids']) or not np.array_equal(z['y'],va['y']):raise RuntimeError('Prediction row mismatch')
        error=float(np.max(np.abs(replay-z['p'])))
        np.testing.assert_allclose(replay,z['p'],rtol=1e-5,atol=1e-6)
    result=dict(status='COMPLETE',trial=name,architecture=architecture,variant=variant,seed=seed,config=cfg,
        metrics=metrics(va['y'],replay,va['ids']),best_epoch=best_epoch,best_stage=best_stage,stop_reason=stop,
        completed_pre_epochs=sum(r['stage']=='pre' for r in history),completed_fine_epochs=sum(r['stage']=='fine' for r in history),
        training_seconds=sum(r['training_seconds'] for r in history),fit_elapsed_seconds=elapsed,
        resource_wait_seconds=resource['waiting_seconds'],parameter_count=params,gpu=torch.cuda.get_device_name(0),precision='bfloat16',
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,fresh_model_checkpoint_reload_pass=True,reload_max_abs_error=error,
        checkpoint_sha256=sha(root/'best.pt'),prediction_sha256=sha(root/'predictions_validation.npz'),
        input_hash=input_hash,code_fingerprints=code,test_accessed=False,finished_utc=now())
    write(root/'result.json',result);write(root/'progress.json',dict(status='COMPLETE',trial=name,metrics=result['metrics'],updated_utc=now()))
    del model,xn,vn,xc,vc,y;gc.collect();torch.cuda.empty_cache()
    return result


def run(architecture,variant,seed):
    name=trial_name(architecture,variant,seed)
    if (HERE/'runs'/name/'result.json').exists():return stage(architecture,variant,seed,None)
    with gpu_lease(name) as resource:
        return stage(architecture,variant,seed,resource)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--architecture',choices=ARCHITECTURES,required=True)
    parser.add_argument('--variant',choices=VARIANTS,required=True);parser.add_argument('--seed',choices=SEEDS,type=int,required=True)
    args=parser.parse_args();name=trial_name(args.architecture,args.variant,args.seed)
    try:run(args.architecture,args.variant,args.seed)
    except BaseException as e:
        write(HERE/'runs'/name/'failure.json',dict(status='FAILED',reason=str(e),traceback=traceback.format_exc(),finished_utc=now()))
        raise


if __name__=='__main__':main()
