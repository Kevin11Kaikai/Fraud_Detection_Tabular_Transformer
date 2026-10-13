"""Reuse the installed official FT backbone and pinned ExcelFormer vendor."""
from __future__ import annotations

import random
import time

import torch
from torch.nn import functional as F

from .common import *
from ..ft_upgrade_20261010.train import build as build_ft


def seed_all(seed):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_bf16_supported():raise RuntimeError('Frozen BF16 protocol unsupported')


def build(architecture,schema,config):
    if architecture=='ft':return build_ft(schema,config)
    cls=external_module('architecture_benchmark_20261010.vendor.excel').ExcelFormer
    return cls(d_numerical=len(schema['excel']['columns']),categories=None,token_bias=True,
        n_layers=3,d_token=128,n_heads=8,attention_dropout=.3,ffn_dropout=0.,residual_dropout=0.,
        prenormalization=True,kv_compression=None,kv_compression_sharing=None,d_out=1,init_scale=.01)


def optimizer(model,architecture,stage,config):
    if architecture=='ft' and stage=='fine':
        from libauc.optimizers import SOAP
        return SOAP(model.make_parameter_groups(),lr=config['fine_lr'],mode='adam',weight_decay=config['weight_decay'],
                    epoch_decay=0,verbose=False,device=torch.device('cuda'))
    groups=model.make_parameter_groups() if architecture=='ft' else model.parameters()
    return torch.optim.AdamW(groups,lr=config['lr'],weight_decay=config['weight_decay'])


def loss(model,architecture,xn,xc,y,mi):
    if architecture=='ft':
        z=model(xn,xc).reshape(-1).float()
        return F.binary_cross_entropy_with_logits(z,y,pos_weight=torch.tensor(3.,device='cuda'))
    z,mask,perm=model(xn,None,mixup=True,beta=.5,mtype='feat_mix')
    lam=(mask.float()*mi).sum(1)
    a=F.binary_cross_entropy_with_logits(z.reshape(-1).float(),y,reduction='none')
    b=F.binary_cross_entropy_with_logits(z.reshape(-1).float(),y[perm],reduction='none')
    return (lam*a+(1-lam)*b).mean()


def update(model,architecture,stage,opt,xn,xc,y,idx,micro,mi):
    opt.zero_grad(set_to_none=True);total=0.
    if stage=='fine':
        from torch.utils.checkpoint import checkpoint
        z=[]
        for chunk in idx.split(micro):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z.append(checkpoint(model,xn[chunk],xc[chunk],use_reentrant=False).reshape(-1))
        value=F.binary_cross_entropy_with_logits(torch.cat(z).float(),y[idx],pos_weight=torch.tensor(3.,device='cuda'))
        if not torch.isfinite(value):raise RuntimeError('Nonfinite SOAP loss')
        value.backward();total=float(value.detach())*len(idx)
    else:
        for chunk in idx.split(micro):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                value=loss(model,architecture,xn[chunk],None if xc is None else xc[chunk],y[chunk],mi)
            if not torch.isfinite(value):raise RuntimeError('Nonfinite training loss')
            (value*(len(chunk)/len(idx))).backward();total+=float(value.detach())*len(chunk)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    opt.step()
    return total


def order(architecture,stage,n,y,seed,epoch):
    if stage=='fine':
        from libauc.sampler import DualSampler
        sampler=DualSampler(np.arange(n),batch_size=128,labels=y,num_pos=32,sampling_rate=None,random_seed=seed*1000+epoch)
        return torch.tensor(list(iter(sampler)),dtype=torch.int64)
    generator=torch.Generator(device='cpu').manual_seed(seed*1000+epoch)
    if architecture=='ft':return torch.multinomial(torch.linspace(1.,4.,n),n,replacement=True,generator=generator)
    return torch.randperm(n,generator=generator)


@torch.inference_mode()
def predict(model,xn,xc,batch):
    model.eval();out=[];torch.cuda.synchronize();tick=time.perf_counter()
    for off in range(0,len(xn),batch):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=model(xn[off:off+batch],None if xc is None else xc[off:off+batch]).reshape(-1)
        out.append(z.float().sigmoid().cpu().numpy())
    torch.cuda.synchronize();p=np.concatenate(out)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise RuntimeError('Invalid CUDA probabilities')
    return p,time.perf_counter()-tick


def rng_state():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy']);random.setstate(state['python'])


def checkpoint(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)
