"""Fill only the two missing seed43 tree controls; do not rerun baselines."""
import time

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score

from .common import *


def run(variant):
    root=HERE/'runs'/f'lgb_{variant}_seed43';root.mkdir(parents=True,exist_ok=True)
    cfg=read(HERE/'protocol.json')['lightgbm']|{'random_state':43}
    if (root/'result.json').exists():
        r=read(root/'result.json')
        if r['config']!=cfg or sha(root/'model.txt')!=r['checkpoint_sha256']:raise RuntimeError('Tree reuse mismatch')
        return r
    source=read(SOURCE/'derived/train_only/manifest.json');names=read(ABLATION/'feature_groups.json')['variants'][variant]
    ix=[source['column_names'].index(n) for n in names];lab=labels()
    arrays={part:np.load(SOURCE/f'derived/train_only/{part}.npy',mmap_mode='r')[:,ix].copy() for part in ['train','validation']}
    for a in arrays.values():a[a==-1]=np.nan
    history={}
    def ap(y,p):return 'ap',float(average_precision_score(y,p)),True
    tick=time.perf_counter();model=lgb.LGBMClassifier(**cfg)
    model.fit(arrays['train'],lab['y_train'],eval_set=[(arrays['validation'],lab['y_validation'])],eval_metric=ap,
              callbacks=[lgb.early_stopping(150,first_metric_only=True,verbose=False),lgb.record_evaluation(history),lgb.log_evaluation(500)])
    training=time.perf_counter()-tick;model.booster_.save_model(str(root/'model.txt'))
    p=model.predict_proba(arrays['validation'])[:,1]
    replay=lgb.Booster(model_file=str(root/'model.txt')).predict(arrays['validation'])
    np.testing.assert_allclose(p,replay,rtol=0,atol=0)
    np.savez_compressed(root/'predictions_validation.npz',ids=lab['ids_validation'],y=lab['y_validation'],p=p)
    pd.DataFrame({'iteration':np.arange(1,len(history['valid_0']['ap'])+1),'validation_ap':history['valid_0']['ap']}).to_csv(root/'history.csv',index=False)
    result=dict(status='COMPLETE',trial=f'lgb_{variant}_seed43',variant=variant,architecture='lgb',seed=43,config=cfg,
        metrics=metrics(lab['y_validation'],p,lab['ids_validation']),training_seconds=training,best_iteration=model.best_iteration_,
        fresh_model_checkpoint_reload_pass=True,reload_max_abs_error=0.,checkpoint_sha256=sha(root/'model.txt'),prediction_sha256=sha(root/'predictions_validation.npz'),
        source_information='Frozen original train_only matrix with NaN handling; neural adapters have different representations.',test_accessed=False,finished_utc=now())
    write(root/'result.json',result);print('TREE_COMPLETE',variant,43,result['metrics']['average_precision'],flush=True)
    return result


if __name__=='__main__':
    for variant in ['B1M','B1C']:run(variant)
