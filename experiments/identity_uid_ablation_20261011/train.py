"""Sequential fixed-recipe fits, with every required matched condition retained."""
from __future__ import annotations

import argparse
import gc
import time
import traceback

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .common import HERE,SOURCE,fingerprints,load,metrics,now,read,sha,write


def condition(trial):
    path=HERE/"runs"/trial
    path.mkdir(parents=True,exist_ok=True)
    receipt=read(HERE/"preflight.json")
    if trial not in receipt["trials"] or receipt["code_fingerprints"]!=fingerprints():
        raise RuntimeError("Unfrozen condition or code mismatch")
    if (path/"result.json").exists():
        done=read(path/"result.json")
        artifact_path=SOURCE/"runs/lgb_train_only" if done["status"]=="REUSED_VERIFIED" else path
        for name,expected in done["artifact_hashes"].items():
            if sha(artifact_path/name)!=expected:
                raise RuntimeError("Completed ablation result changed")
        print("REUSE",trial,flush=True)
        return done
    seed_name,variant=trial.split("_",1)
    seed=int(seed_name[4:])
    config=receipt["lgb_config"]|{"random_state":seed}
    if trial=="seed42_B11":
        original=read(SOURCE/"runs/lgb_train_only/result.json")
        if config!=original["config"]:
            raise RuntimeError("Reused model has different parameters")
        done={**original,"trial":trial,"variant":variant,"seed":seed,
              "status":"REUSED_VERIFIED","training_seconds_new":0.,"training_seconds_original":original["training_seconds"],
              "source_run":str(SOURCE/"runs/lgb_train_only"),"code_fingerprints_ablation":fingerprints(),
              "feature_columns":read(HERE/"feature_groups.json")["variants"][variant]}
        # Hashes refer to original files and are checked there, not rewritten locally.
        write(path/"result.json",done)
        print("REUSE SOURCE FULL",trial,done["metrics"]["average_precision"],flush=True)
        return done
    tr,va,labels,names=load(variant)
    history={}
    started=now()
    tick=time.perf_counter()
    print("START",trial,len(names),flush=True)
    write(path/"progress.json",{"status":"RUNNING","trial":trial,"started_utc":started,"feature_count":len(names)})
    def ap(y,p):
        return "average_precision",float(average_precision_score(y,p)),True
    def progress(env):
        if env.iteration%50==0:
            write(path/"progress.json",{"status":"RUNNING","trial":trial,"iteration":env.iteration+1,
                  "ap":float(env.evaluation_result_list[0][2]),"elapsed_seconds":time.perf_counter()-tick,"updated_utc":now()})
        if (env.iteration+1)%500==0:
            env.model.save_model(str(path/"latest_checkpoint.txt"))
    model=lgb.LGBMClassifier(**config)
    model.fit(tr,labels["y_train"],eval_set=[(va,labels["y_validation"])],eval_metric=ap,
              callbacks=[lgb.early_stopping(150,first_metric_only=True,verbose=False),
                         lgb.record_evaluation(history),lgb.log_evaluation(500),progress])
    seconds=time.perf_counter()-tick
    t=time.perf_counter()
    p=model.predict_proba(va)[:,1]
    infer_seconds=time.perf_counter()-t
    model.booster_.save_model(str(path/"model.txt"))
    np.savez_compressed(path/"predictions_validation.npz",p=p,ids=labels["ids_validation"],y=labels["y_validation"])
    values=history["valid_0"]["average_precision"]
    pd.DataFrame({"iteration":range(1,len(values)+1),"validation_ap":values}).to_csv(path/"history.csv",index=False)
    pd.DataFrame({"feature":names,"importance":model.feature_importances_}).sort_values("importance",ascending=False).to_csv(path/"feature_importance.csv",index=False)
    done={"status":"COMPLETED","trial":trial,"variant":variant,"seed":seed,"config":config,
          "metrics":metrics(labels["y_validation"],p,labels["ids_validation"]),
          "best_iteration":int(model.best_iteration_),"iterations_executed":len(values),
          "training_seconds_new":seconds,"prediction_seconds":infer_seconds,"actual_device":"CPU, 8 threads",
          "feature_columns":names,"code_fingerprints_ablation":fingerprints(),"preflight_sha256":sha(HERE/"preflight.json"),
          "source_feature_groups_sha256":sha(HERE/"feature_groups.json"),
          "started_utc":started,"finished_utc":now(),
          "artifact_hashes":{name:sha(path/name) for name in ["model.txt","predictions_validation.npz","history.csv","feature_importance.csv"]}}
    write(path/"result.json",done)
    write(path/"progress.json",{"status":"COMPLETED","metrics":done["metrics"],"finished_utc":done["finished_utc"]})
    print("DONE",trial,done["metrics"],"seconds",seconds,flush=True)
    del tr,va,model
    gc.collect()
    return done


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--trial")
    args=parser.parse_args()
    trials=[args.trial] if args.trial else read(HERE/"preflight.json")["trials"]
    for trial in trials:
        try:
            condition(trial)
        except Exception as error:
            write(HERE/"runs"/trial/"failure.json",{"status":"FAILED","exception":str(error),"traceback":traceback.format_exc(),"finished_utc":now()})
            raise


if __name__=="__main__":
    main()
