"""Training-only resource estimate and genuine saved-state reload contracts."""
import argparse
import gc
import time
import traceback

import joblib
import torch

from .common import *
from . import models
from .prepare import historical_encode
from .resource import gpu_lease


def cpu_checks():
    # Loading from a fresh interpreter is an acceptance condition for adapters.
    for variant in VARIANTS:
        proc=joblib.load(HERE/'derived'/variant/'adapter.joblib')
        if proc.__class__.__module__!='experiments.identity_uid_transformer_transfer_20261011.prepare':
            raise RuntimeError('Adapter is not importable across processes')
    base,bs=load('B00');comparisons={}
    for variant in VARIANTS[1:]:
        parts,schema=load(variant)
        for key,names in [('x_num',bs['numeric_output_names']),('x_cat',bs['categorical'])]:
            target=schema['numeric_output_names'] if key=='x_num' else schema['categorical']
            ix=[target.index(n) for n in names]
            for part in ['train','validation']:
                for off in range(0,len(base[part][key]),32768):
                    np.testing.assert_allclose(base[part][key][off:off+32768],parts[part][key][off:off+32768,ix],rtol=0,atol=0)
        comparisons[variant]='All common tensor values exactly preserved'
    # Perturb future training labels; earlier block encodings must not change.
    n=1000;dt=np.arange(n);cat=(np.arange(n)%11)[:,None];y=(np.arange(n)%7==0).astype('float32')
    a,_,_,cuts=historical_encode(cat,y,cat[:10],dt)
    perturbed=y.copy();perturbed[cuts[-2]:]=1-perturbed[cuts[-2]:]
    b,_,_,_=historical_encode(cat,perturbed,cat[:10],dt)
    np.testing.assert_array_equal(a,b)
    write(HERE/'cpu_contracts.json',dict(status='PASS',importable_adapters=4,
        common_tensor_equivalence=comparisons,historical_future_label_invariance=True,
        raw_uid_or_transactionid_features=False,final_test_accessed=False,finished_utc=now()))
    print('CPU_CONTRACTS_PASS',flush=True)


def cuda_stage(architecture,micro,resource):
    protocol=read(HERE/'protocol.json');cfg=protocol[architecture]|{'microbatch':micro}
    records=[]
    for variant in ['B00','B1M']:
        models.seed_all(42);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        parts,schema=load(variant,architecture);tr,va=parts['train'],parts['validation'];key='x_num' if architecture=='ft' else 'x_excel'
        xn=torch.as_tensor(np.array(tr[key],copy=True),device='cuda');vn=torch.as_tensor(np.array(va[key],copy=True),device='cuda')
        xc=torch.as_tensor(np.array(tr['x_cat'],copy=True),device='cuda') if architecture=='ft' else None
        y=torch.as_tensor(np.array(tr['y'],copy=True),device='cuda',dtype=torch.float32)
        mi=torch.tensor(schema['excel']['mi_weights'],device='cuda') if architecture=='excel' else None
        model=models.build(architecture,schema,cfg).cuda();opt=models.optimizer(model,architecture,'pre',cfg)
        bs=128 if architecture=='ft' else 256;idx=torch.arange(bs,device='cuda');times=[]
        for step in range(10):
            model.train();torch.cuda.synchronize();tick=time.perf_counter()
            models.update(model,architecture,'pre',opt,xn,xc,y,idx,micro,mi)
            torch.cuda.synchronize()
            if step>=3:times.append(time.perf_counter()-tick)
        fine=None
        if architecture=='ft':
            del opt;opt=models.optimizer(model,architecture,'fine',cfg)
            torch.cuda.synchronize();tick=time.perf_counter()
            for step in range(3):models.update(model,architecture,'fine',opt,xn,xc,y,idx,micro,mi)
            torch.cuda.synchronize();fine=(time.perf_counter()-tick)/3
        p,_=models.predict(model,xn[:256],None if xc is None else xc[:256],128)
        path=HERE/'runs'/f'smoke_{architecture}_{micro}_{variant}.pt'
        models.checkpoint(path,{'model':model.state_dict()})
        state=torch.load(path,map_location='cpu',weights_only=False)['model']
        del model,opt;gc.collect();torch.cuda.empty_cache()
        model=models.build(architecture,schema,cfg).cuda();model.load_state_dict(state)
        q,_=models.predict(model,xn[:256],None if xc is None else xc[:256],128)
        np.testing.assert_allclose(p,q,rtol=1e-5,atol=1e-6)
        peak=torch.cuda.max_memory_reserved()/2**30
        if peak>12:raise RuntimeError(f'Resource smoke exceeds12GiB: {peak}')
        records.append(dict(architecture=architecture,variant=variant,microbatch=micro,
            optimizer_update_seconds=float(np.median(times)),fine_update_seconds=fine,
            epoch_training_seconds_estimate=float(np.median(times))*int(np.ceil(len(y)/bs)),
            peak_reserved_gib=peak,fresh_reload_max_abs_error=float(np.max(np.abs(p-q))),
            gpu=torch.cuda.get_device_name(0),training_only_fixture=True))
        print('CUDA_SMOKE',records[-1],flush=True)
        del model,xn,vn,xc,y,mi,state;gc.collect();torch.cuda.empty_cache()
    write(HERE/f'smoke_{architecture}_{micro}.json',dict(status='PASS',records=records,resource=resource,finished_utc=now()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--architecture',choices=ARCHITECTURES);parser.add_argument('--micro',type=int)
    args=parser.parse_args()
    if args.architecture:
        try:
            with gpu_lease(f'pre_admission_smoke_{args.architecture}_{args.micro}') as resource:
                cuda_stage(args.architecture,args.micro,resource)
        except BaseException as error:
            classification='RESOURCE_LIMIT' if isinstance(error,torch.OutOfMemoryError) or 'Resource smoke exceeds' in str(error) else 'IMPLEMENTATION_ERROR'
            write(HERE/f'smoke_failure_{args.architecture}_{args.micro}.json',dict(reason=str(error),classification=classification,
                traceback=traceback.format_exc(),finished_utc=now()))
            raise
    else:cpu_checks()


if __name__=='__main__':main()
