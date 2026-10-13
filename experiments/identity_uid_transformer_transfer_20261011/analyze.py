"""Rescore all full validation predictions and preserve subgroup tradeoffs."""
from __future__ import annotations

import math
import time
from datetime import datetime

import pandas as pd

from .common import *


def numpy_scores(y,p,ids):
    y=np.asarray(y,dtype='float64');p=np.asarray(p,dtype='float64');ids=np.asarray(ids)
    descending=np.lexsort((ids,-p));ys=y[descending];ps=p[descending]
    endpoints=np.r_[np.flatnonzero(np.diff(ps)!=0),len(p)-1]
    true=np.cumsum(ys)[endpoints];precision=true/(endpoints+1)
    ap=np.sum(np.diff(np.r_[0.,true])*precision)/y.sum()
    order=np.argsort(p,kind='stable');sortedp=p[order]
    starts=np.r_[0,np.flatnonzero(np.diff(sortedp)!=0)+1];ends=np.r_[starts[1:],len(p)]
    ranks=np.empty(len(p),dtype='float64')
    for start,end in zip(starts,ends):ranks[order[start:end]]=(start+1+end)/2
    nf=y.sum();auc=(ranks[y==1].sum()-nf*(nf+1)/2)/(nf*(len(y)-nf))
    k1=math.ceil(.01*len(y));k5=math.ceil(.05*len(y));c1=int(ys[:k1].sum());c5=int(ys[:k5].sum())
    return dict(average_precision=float(ap),roc_auc=float(auc),recall_top_1pct=c1/nf,
        precision_top_1pct=c1/k1,recall_top_5pct=c5/nf,brier_score=float(np.mean((p-y)**2)),captured_top_1pct=c1,captured_top_5pct=c5)


def tree_reference(variant,seed):
    if seed==43 and variant in ['B1M','B1C']:
        root=HERE/'runs'/f'lgb_{variant}_seed43';return root,read(root/'result.json')
    root=ABLATION/'runs'/f'seed{seed}_{variant}';return root,read(root/'result.json')


def analyze():
    tick=time.perf_counter();lab=labels();rows=[];verification={};strata=[]
    input_hashes={}
    for variant in VARIANTS:
        root=HERE/'derived'/variant;schema=read(root/'schema.json')
        for name,h in schema['files'].items():
            if sha(root/name)!=h:raise RuntimeError('Adapted tensor or transform changed: '+variant+'/'+name)
        receipt=read(root/'excel/receipt.json')
        for name,h in receipt['files'].items():
            if sha(root/'excel'/name)!=h:raise RuntimeError('Excel adapted input changed')
        input_hashes[('ft',variant)]=sha(root/'schema.json')
        input_hashes[('excel',variant)]=sha(root/'schema.json')+':'+sha(root/'excel/receipt.json')
    with np.load(ABLATION/'strata_validation.npz') as z:
        if not np.array_equal(z['ids'],lab['ids_validation']):raise RuntimeError('Frozen stratum IDs changed')
        masks={'uid_seen':z['uid_seen'].copy(),'uid_unseen':~z['uid_seen'],
               'identity_id12_observed':z['identity_marker_observed'].copy(),'identity_id12_missing':~z['identity_marker_observed']}
    references=[]
    for architecture in (*ARCHITECTURES,'lgb'):
        for seed in SEEDS:
            for variant in VARIANTS:
                name=trial_name(architecture,variant,seed)
                if architecture=='lgb':root,r=tree_reference(variant,seed)
                else:root=HERE/'runs'/name;r=read(root/'result.json')
                if r['status'] not in ['COMPLETE','COMPLETED']:raise RuntimeError('Condition incomplete: '+name)
                pred=root/'predictions_validation.npz'
                if r.get('reused') or r.get('reused_source'):
                    source=SOURCE/'runs/lgb_train_only';pred=source/'predictions_validation.npz'
                with np.load(pred) as z:
                    if not np.array_equal(z['ids'],lab['ids_validation']) or not np.array_equal(z['y'],lab['y_validation']):raise RuntimeError('Full-validation identity mismatch: '+name)
                    p=z['p'].copy()
                score=metrics(lab['y_validation'],p,lab['ids_validation']);other=numpy_scores(lab['y_validation'],p,lab['ids_validation'])
                for key,value in other.items():
                    np.testing.assert_allclose(score[key],value,rtol=0,atol=1e-12)
                    np.testing.assert_allclose(score[key],r['metrics'][key],rtol=0,atol=1e-12)
                if architecture!='lgb':
                    if r['input_hash']!=input_hashes[(architecture,variant)]:raise RuntimeError('Model/input schema mismatch')
                    if r['code_fingerprints']!=fingerprints() or sha(root/'best.pt')!=r['checkpoint_sha256'] or sha(pred)!=r['prediction_sha256']:
                        raise RuntimeError('Neural result fingerprint changed')
                    if not r['fresh_model_checkpoint_reload_pass']:raise RuntimeError('Missing fresh checkpoint replay')
                rows.append(dict(trial=name,architecture=architecture,variant=variant,seed=seed,**score,
                                 training_seconds_new=r.get('training_seconds',r.get('training_seconds_new',0)) if root.is_relative_to(HERE) else 0,
                                 best_epoch=r.get('best_epoch'),best_stage=r.get('best_stage'),result_path=str(root/'result.json')))
                verification[name]=dict(status='PASS',full_validation_metrics_recomputed=True,numpy_formula_max_error=max(abs(score[k]-other[k]) for k in other),
                    fresh_model_reload=r.get('fresh_model_checkpoint_reload_pass','Reused sealed previous tree replay'),prediction_sha256=sha(pred))
                references.append(dict(trial=name,prediction_path=str(pred),result_path=str(root/'result.json')))
                for group,mask in masks.items():
                    m=metrics(lab['y_validation'][mask],p[mask],lab['ids_validation'][mask])
                    strata.append(dict(trial=name,architecture=architecture,seed=seed,variant=variant,stratum=group,**m))
    frame=pd.DataFrame(rows);frame.to_csv(HERE/'results.csv',index=False);sf=pd.DataFrame(strata);sf.to_csv(HERE/'strata_metrics.csv',index=False)
    effects=[]
    for architecture in (*ARCHITECTURES,'lgb'):
        for seed in SEEDS:
            subset=frame[(frame.architecture==architecture)&(frame.seed==seed)].set_index('variant')
            for label,a,b in [('identity','B10','B00'),('UID_mean_std_given_identity','B1M','B10'),('UID_frequency_nunique_given_identity','B1C','B10')]:
                effects.append(dict(architecture=architecture,seed=seed,effect=label,delta_ap=float(subset.loc[a,'average_precision']-subset.loc[b,'average_precision']),
                    delta_top1_captured=int(subset.loc[a,'captured_top_1pct']-subset.loc[b,'captured_top_1pct']),delta_brier=float(subset.loc[a,'brier_score']-subset.loc[b,'brier_score'])))
    ef=pd.DataFrame(effects);ef.to_csv(HERE/'effects.csv',index=False)
    sampler_equivalence={}
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            histories={v:pd.read_csv(HERE/'runs'/trial_name(architecture,v,seed)/'history.csv') for v in VARIANTS}
            reference=histories['B00'].set_index(['stage','epoch'])
            for variant,history in histories.items():
                for (stage,epoch),row in history.set_index(['stage','epoch']).iterrows():
                    if (stage,epoch) in reference.index and row.sampler_order_sha256!=reference.loc[(stage,epoch),'sampler_order_sha256']:
                        raise RuntimeError('Matched sampler stream mismatch')
            sampler_equivalence[f'{architecture}_seed{seed}']='PASS for all common completed stages/epochs'
    preserved=0
    protocol=read(HERE/'protocol.json')
    for name,h in protocol['source_seal']['files'].items():
        if sha(ABLATION/name)!=h:raise RuntimeError('Original ablation changed: '+name)
        preserved+=1
    for name,h in protocol['source_campaign_seal']['files'].items():
        if sha(SOURCE/name)!=h:raise RuntimeError('Original champion packet changed: '+name)
    for path,h in protocol['source_hashes'].items():
        if sha(path)!=h:raise RuntimeError('Frozen dependency changed: '+path)
    for name,h in protocol['source_arrays'].items():
        if sha(SOURCE/'derived/train_only'/name)!=h:raise RuntimeError('Frozen input array changed')
    for name,key in [('train_transaction.csv','raw_transaction_sha256'),('train_identity.csv','raw_identity_sha256')]:
        if sha(PROJECT/'data/raw'/name)!=protocol['input_raw_hashes'][key]:raise RuntimeError('Raw input changed')
    checks=dict(status='PASS',conditions=verification,source_ablation_files_unchanged=preserved,
                prepared_tensor_hashes_verified=True,matched_sampler_streams=sampler_equivalence,
                source_champion_files_unchanged=len(protocol['source_campaign_seal']['files']),
                raw_and_source_arrays_unchanged=True,test_accessed=False,independent_replication_performed=False,
                elapsed_seconds=time.perf_counter()-tick,finished_utc=now())
    write(HERE/'verification.json',checks);write(HERE/'prediction_references.json',references)
    neural=frame[frame.architecture!='lgb'];best=neural.loc[neural.average_precision.idxmax()]
    tree=frame[frame.architecture=='lgb'];besttree=tree.loc[tree.average_precision.idxmax()]
    decisions={}
    for architecture in ARCHITECTURES:
        detail={}
        for variant in VARIANTS:
            ap=neural[(neural.architecture==architecture)&(neural.variant==variant)].average_precision
            detail[variant]=dict(mean_ap=float(ap.mean()),minimum_ap=float(ap.min()),seed_values=ap.tolist())
        selected=max(VARIANTS,key=lambda v:detail[v]['mean_ap'])
        differences=ef[ef.architecture==architecture].to_dict('records')
        detail_seen=[]
        for seed in SEEDS:
            for group in ['uid_seen','uid_unseen']:
                a=sf[(sf.architecture==architecture)&(sf.seed==seed)&(sf.stratum==group)].set_index('variant')
                detail_seen.append(dict(seed=seed,stratum=group,identity_ap=float(a.loc['B10','average_precision']),
                    uid_mean_std_ap=float(a.loc['B1M','average_precision']),uid_count_ap=float(a.loc['B1C','average_precision'])))
        decisions[architecture]=dict(selected_by_two_seed_mean_ap=selected,variants=detail,matched_effects=differences,uid_strata=detail_seen)
    decision=dict(status='COMPLETE_INPUT_TRAIN_VERIFY_LOOP',architectures=decisions,
        strongest_neural_trial=best.trial,strongest_neural_ap=float(best.average_precision),strongest_tree_trial=besttree.trial,strongest_tree_ap=float(besttree.average_precision),
        exceeds_strongest_tree_on_reused_validation=bool(best.average_precision>besttree.average_precision),
        classification='Development evidence only; tree and neural adapters differ, no pure architecture causality claim.',
        test_accessed=False,finished_utc=now())
    tree_means=tree.groupby('variant').average_precision.mean()
    best_tree_mean_variant=str(tree_means.idxmax());best_tree_mean=float(tree_means.max())
    decision['two_seed_comparison']=dict(strongest_tree_variant=best_tree_mean_variant,strongest_tree_mean_ap=best_tree_mean,
        neural={a:dict(selected_variant=decisions[a]['selected_by_two_seed_mean_ap'],
            selected_mean_ap=decisions[a]['variants'][decisions[a]['selected_by_two_seed_mean_ap']]['mean_ap'],
            delta_to_strongest_tree_mean=decisions[a]['variants'][decisions[a]['selected_by_two_seed_mean_ap']]['mean_ap']-best_tree_mean)
            for a in ARCHITECTURES})
    decision['historical_lightgbm_context']=protocol.get('historical_lightgbm_context',[])
    write(HERE/'FINAL_DECISION.json',decision)
    write_report(frame,ef,sf,decision,checks,protocol)
    seal={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['artifact_manifest.json','campaign_state.json','resource_status.json'] and not p.name.endswith('.tmp')}
    write(HERE/'artifact_manifest.json',dict(status='SEALED',files=seal,finished_utc=now(),test_accessed=False))
    print('ALL24_CONDITIONS_VERIFIED','best_neural',best.trial,float(best.average_precision),'best_tree',float(besttree.average_precision),flush=True)


def write_report(frame,effects,strata,decision,checks,protocol):
    elapsed=(datetime.now().astimezone()-datetime.fromisoformat(protocol['started_utc'])).total_seconds()/3600
    hours=float(frame.training_seconds_new.sum())/3600
    lines=['# identity/UID → FT-Transformer 与 ExcelFormer 完整闭环', '',
        f"完整16个神经网络条件和8个树模型对照均有有效产物，其中6个树对照复用、2个补齐。新增训练{hours:.2f}小时；总墙钟{elapsed:.2f}小时（含工程、排队、恢复与核验）。", '',
        f"最佳神经网络：{decision['strongest_neural_trial']}，AP={decision['strongest_neural_ap']:.6f}。最强树对照AP={decision['strongest_tree_ap']:.6f}。是否超过：{decision['exceeds_strongest_tree_on_reused_validation']}。", '',
        '输入为冻结的191个共同字段，分别增加26个identity、33个UID均值/标准差或13个UID频率/唯一值数量。FT类别embedding与训练集数值变换；Excel采用历史目标编码、训练集MI排序及原feat_mix训练。', '',
        '## 主结果', '', '| 模型 | 输入 | Seed42 AP | Seed43 AP | 两seed平均 |', '|---|---|---:|---:|---:|']
    for arch in ['lgb','ft','excel']:
        for variant in VARIANTS:
            a=frame[(frame.architecture==arch)&(frame.variant==variant)].set_index('seed')
            lines.append(f"| {arch} | {variant} | {a.loc[42,'average_precision']:.6f} | {a.loc[43,'average_precision']:.6f} | {a.average_precision.mean():.6f} |")
    lines+=['','B00=共同；B10=共同+identity；B1M=共同+identity+UID均值/标准差；B1C=共同+identity+UID频率/唯一值数量。','','## 匹配增量与移植结论','', '| 模型 | 信息增量 | Seed42 ΔAP | Seed43 ΔAP |','|---|---|---:|---:|']
    for arch in ['ft','excel','lgb']:
        for effect in effects.effect.unique():
            a=effects[(effects.architecture==arch)&(effects.effect==effect)].set_index('seed')
            lines.append(f"| {arch} | {effect} | {a.loc[42,'delta_ap']:+.6f} | {a.loc[43,'delta_ap']:+.6f} |")
    for arch in ARCHITECTURES:lines+=['',f"{arch} 两seed平均AP所选输入为 {decision['architectures'][arch]['selected_by_two_seed_mean_ap']}。这是使用开发验证集的方案选择，不是独立测试认证。"]
    lines+=['',f"最强树输入的两seed平均AP={decision['two_seed_comparison']['strongest_tree_mean_ap']:.6f}。"]
    for arch in ARCHITECTURES:
        d=decision['two_seed_comparison']['neural'][arch]
        lines+=[f"{arch} 所选输入两seed平均AP={d['selected_mean_ap']:.6f}，相对最强树平均ΔAP={d['delta_to_strongest_tree_mean']:+.6f}。"]
    for context in decision['historical_lightgbm_context']:
        lines+=[f"历史{context['name']} AP={context['average_precision']:.6f}；本轮最佳神经单trial ΔAP={decision['strongest_neural_ap']-context['average_precision']:+.6f}。历史输入及时间语义不同，保留作原目标背景。"]
    lines+=['','## UID分层的保留代价','', '| 模型 | Seed | 输入 | 已见UID AP | 未见UID AP |','|---|---:|---|---:|---:|']
    for arch in ['ft','excel','lgb']:
        for seed in SEEDS:
            for variant in VARIANTS:
                a=strata[(strata.architecture==arch)&(strata.seed==seed)&(strata.variant==variant)].set_index('stratum')
                lines.append(f"| {arch} | {seed} | {variant} | {a.loc['uid_seen','average_precision']:.6f} | {a.loc['uid_unseen','average_precision']:.6f} |")
    lines+=['','UID为预定义伪UID；不能视为经过核实的真实客户身份。未见组47256行、2091个欺诈全部保留。分层Top-K在组内计算，不等于统一审核预算下的子组召回。','','## 核验、边界与复现','',
        f"24组全验证预测的AP、ROC、Top-K和Brier经另一套NumPy公式复算。16个神经模型在训练结束后新建模型、重载最佳检查点并重新推理完整验证集；原消融{checks['source_ablation_files_unchanged']}个文件、原始数据与源缓存保持不变。", '',
        '时间切分为原70/15/15；本轮仅使用413378训练行和88581开发验证行。最终测试与官方无标签测试均未访问。重复使用验证集进行早停/选择属于开发证据；没有独立研究者复现、显著性认证或部署保证。', '',
        '本轮UID及其他组统计为完整训练分区拟合、验证映射冻结，不能宣称严格过去历史流式性能。对比原横纵特征主线时，时间语义、输入与预处理均有差异，不能只归因于模型架构。', '',
        'FT固定20轮+1轮SOAP/BCE；Excel固定最多16轮、原warmup/cosine与早停配方。完整执行冻结配方不等于证明已收敛。微批由训练集CUDA预检决定，有效batch保持不变。', '',
        '所有负增量、未见UID退化、Top-K与Brier代价均保留在effects.csv、strata_metrics.csv和results.csv。不存在依据结果剔除样本、修改目标或追加超参数搜索。', '',
        '在fraud-detection-mvp根目录运行：', '', '```powershell',
        '& .\\.venv\\Scripts\\python.exe -X utf8 -B -u -m experiments.identity_uid_transformer_transfer_20261011.campaign', '```', '',
        '已完成条件只核验并复用，未完成条件由匹配指纹的完整epoch检查点续跑；哈希不匹配时直接停止。协议、代码、命令和文件指纹均保存在本实验目录。']
    (HERE/'FINAL_REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':analyze()
