"""One final model replay, paired effects, strata and Transformer candidate contract."""
from __future__ import annotations

import gc
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..kaggle_fraudsquad_reproduction_20261011.features import STRING_COLUMNS
from .common import HERE,PROJECT,SOURCE,fingerprints,metrics,now,read,sha,write


def effect(records,seed,a,b):
    left,right=records[f"seed{seed}_{a}"],records[f"seed{seed}_{b}"]
    return {name:float(left["metrics"][name]-right["metrics"][name])
            for name in ["average_precision","roc_auc","recall_top_1pct","recall_top_5pct","brier_score","captured_top_1pct","captured_top_5pct"]}


def analyze():
    preflight=read(HERE/"preflight.json")
    if preflight["code_fingerprints"]!=fingerprints():
        raise RuntimeError("Frozen analysis or training code changed")
    schema=read(SOURCE/"derived/train_only/manifest.json")
    if sha(SOURCE/"derived/train_only/manifest.json")!=preflight["source_schema_sha256"]:
        raise RuntimeError("Source schema changed")
    with np.load(SOURCE/"derived/labels_ids.npz") as archive:
        labels={k:archive[k] for k in archive.files}
    feature_groups=read(HERE/"feature_groups.json")
    if sha(HERE/"feature_groups.json")!=preflight["feature_groups_sha256"] or sha(HERE/"strata_validation.npz")!=preflight["strata_sha256"]:
        raise RuntimeError("Column definitions or strata changed")
    with np.load(HERE/"strata_validation.npz") as archive:
        uid_seen=archive["uid_seen"]
        identity_marker=archive["identity_marker_observed"]
        np.testing.assert_array_equal(archive["ids"],labels["ids_validation"])
    strata={"all":np.ones(len(uid_seen),dtype=bool),"uid_seen":uid_seen,"uid_unseen":~uid_seen,
            "identity_id12_observed":identity_marker,"identity_id12_missing":~identity_marker}
    for u in [False,True]:
        for i in [False,True]:
            strata[f"uid_seen={u},id12_observed={i}"]=(uid_seen==u)&(identity_marker==i)
    records,checks,probabilities,strata_rows={},{},{},[]
    raw=np.load(SOURCE/"derived/train_only/validation.npy",mmap_mode="r")
    for trial in preflight["trials"]:
        result=read(HERE/"runs"/trial/"result.json")
        if result["status"] not in ["COMPLETED","REUSED_VERIFIED"] or result["code_fingerprints_ablation"]!=fingerprints():
            raise RuntimeError("Incomplete or stale ablation run")
        if result["feature_columns"]!=feature_groups["variants"][result["variant"]]:
            raise RuntimeError("Model feature list differs from frozen variant")
        if result["config"]!=(preflight["lgb_config"]|{"random_state":result["seed"]}):
            raise RuntimeError("Model recipe differs from frozen recipe")
        path=SOURCE/"runs/lgb_train_only" if result["status"]=="REUSED_VERIFIED" else HERE/"runs"/trial
        for file,expected in result["artifact_hashes"].items():
            if sha(path/file)!=expected:
                raise RuntimeError("Model/prediction/history artifact mismatch")
        with np.load(path/"predictions_validation.npz") as archive:
            p,y,ids=archive["p"],archive["y"],archive["ids"]
        np.testing.assert_array_equal(ids,labels["ids_validation"])
        np.testing.assert_array_equal(y,labels["y_validation"])
        recomputed=metrics(y,p,ids)
        for name,value in result["metrics"].items():
            np.testing.assert_allclose(recomputed[name],value,rtol=0,atol=1e-12)
        if result["status"]=="REUSED_VERIFIED":
            # Reuse the already completed replay whose exact files remain sealed.
            previous=read(SOURCE/"verification.json")
            check=next(c for c in previous["checks"] if c["condition"]=="lgb_train_only")
            if check["status"]!="PASS":
                raise RuntimeError("Source replay failed")
            reload_error=check["model_reload_max_abs_error"]
            replay_kind="Reused sealed source replay, not repeated"
        else:
            indices=[schema["column_names"].index(n) for n in result["feature_columns"]]
            valid=raw[:,indices].copy()
            valid[valid==-1]=np.nan
            model=lgb.Booster(model_file=str(path/"model.txt"))
            if model.num_feature()!=len(indices):
                raise RuntimeError("Reloaded model feature count mismatch")
            replay=model.predict(valid,num_threads=8)
            np.testing.assert_allclose(replay,p,rtol=0,atol=1e-12)
            reload_error=float(np.max(np.abs(replay-p)))
            replay_kind="New saved-model replay"
            del valid,model,replay
            gc.collect()
        checks[trial]={"status":"PASS","model_reload_max_abs_error":reload_error,"replay":replay_kind,
                       "metrics_recomputed":True,"prediction_ids_aligned":True}
        records[trial]=result
        probabilities[trial]=p
        for group,mask in strata.items():
            if len(np.unique(y[mask]))==2:
                m=metrics(y[mask],p[mask],ids[mask])
                strata_rows.append({"trial":trial,"stratum":group,**m})
            else:
                strata_rows.append({"trial":trial,"stratum":group,"n_rows":int(mask.sum()),"n_fraud":int(y[mask].sum()),"status":"UNASSESSABLE_SINGLE_CLASS"})
    effects=[]
    for seed in [42,43]:
        for name,a,b in [("identity|noUID","B10","B00"),("identity|UID","B11","B01"),
                         ("UID|noIdentity","B01","B00"),("UID|identity","B11","B10")]:
            effects.append({"seed":seed,"effect":name,"with":a,"without":b,**effect(records,seed,a,b)})
        interaction=records[f"seed{seed}_B11"]["metrics"]["average_precision"]-records[f"seed{seed}_B10"]["metrics"]["average_precision"]-records[f"seed{seed}_B01"]["metrics"]["average_precision"]+records[f"seed{seed}_B00"]["metrics"]["average_precision"]
        effects.append({"seed":seed,"effect":"identity_UID_interaction","average_precision":float(interaction)})
    for name,a,b in [("mean_std_only|identity","B1M","B10"),("frequency_nunique_only|identity","B1C","B10"),
                     ("mean_std_added_to_counts|identity","B11","B1C"),("frequency_nunique_added_to_mean_std|identity","B11","B1M")]:
        effects.append({"seed":42,"effect":name,"with":a,"without":b,**effect(records,42,a,b)})
    write(HERE/"effects.json",effects)
    pd.DataFrame(effects).to_csv(HERE/"effects.csv",index=False)
    pd.DataFrame(strata_rows).to_csv(HERE/"strata_metrics.csv",index=False)
    flat=[]
    for trial,r in records.items():
        flat.append({"trial":trial,"seed":r["seed"],"variant":r["variant"],"feature_count":len(r["feature_columns"]),
                     "status":r["status"],"best_iteration":r["best_iteration"],"training_seconds_new":r["training_seconds_new"],**r["metrics"]})
    pd.DataFrame(flat).to_csv(HERE/"ablation_results.csv",index=False)
    def increments(name):
        return [e["average_precision"] for e in effects if e["effect"]==name]
    identity_gain=increments("identity|UID")
    uid_gain=increments("UID|identity")
    uid_no_identity_gain=increments("UID|noIdentity")
    uid_supported=all(x>0 for x in uid_gain) and all(x>0 for x in uid_no_identity_gain)
    identity_supported=all(x>0 for x in identity_gain)
    detail={e["effect"]:e["average_precision"] for e in effects if e["seed"]==42}
    mean_supported=uid_supported and detail["mean_std_added_to_counts|identity"]>0 and detail["mean_std_only|identity"]>0
    count_supported=uid_supported and detail["frequency_nunique_added_to_mean_std|identity"]>0 and detail["frequency_nunique_only|identity"]>0
    candidates=[]
    if mean_supported:
        candidates.append({"family":"UID mean/std","priority":1,"columns":feature_groups["uid_mean_std"],
                           "evidence":"UID gain has same positive sign across seeds; mean/std ablation in seed42 supports both standalone and conditional gain"})
    if count_supported:
        candidates.append({"family":"UID frequency/nunique","priority":2 if mean_supported else 1,
                           "columns":feature_groups["uid_frequency_nunique"],"evidence":"UID gain has same positive sign across seeds; count/frequency ablation in seed42 supports standalone and conditional gain"})
    if identity_supported:
        candidates.append({"family":"raw identity","priority":len(candidates)+1,"columns":feature_groups["identity"],
                           "evidence":"Identity conditional on UID has positive AP increment in both paired seeds"})
    categorical_identity=[n for n in feature_groups["identity"] if n in STRING_COLUMNS]
    decision={"status":"TREE_ABLATION_DECISION_COMPLETE_TRANSFORMER_NOT_TRAINED","uid_supported_across_seeds":uid_supported,
              "identity_supported_given_uid_across_seeds":identity_supported,"identity_given_uid_ap":identity_gain,
              "uid_given_identity_ap":uid_gain,"uid_without_identity_ap":uid_no_identity_gain,
              "uid_detail_seed42_ap":{k:v for k,v in detail.items() if "mean_std" in k or "frequency_nunique" in k},
              "prioritized_candidates":candidates,"identity_categorical_fields":categorical_identity,
              "identity_numeric_fields":[n for n in feature_groups["identity"] if n not in categorical_identity],
              "joint_dependency":{"uid_id_02_ct":"Requires identity and UID together; omit from a transaction-only UID pack"},
              "input_contract":{"no_literal_uid_or_transactionid_tokens":True,"numeric":"Training-only imputation/scaling plus missing/unseen masks",
                                "categorical_identity":"Training-only vocabulary and embeddings; preserve missing/unknown semantics, avoid ordinal scalar use",
                                "information_policy":"For reproducing this benchmark: whole-train fitted statistics, frozen validation mapping. For streaming claim: rebuild strictly past-only stats; do not reuse batch validation stats.",
                                "claim":"Tree evidence only supports prioritization. Matching FT/ExcelTransformer training has NOT EXECUTED."},
              "notes":"UID subgroup evidence uses seed42 only; family interactions and optimizer randomness limit attribution. Negative/discordant families retained in effects.csv, no threshold or test-based selection added.",
              "finished_utc":now()}
    write(HERE/"TRANSFORMER_DECISION.json",decision)
    # Check source preservation once at final delivery; no training is repeated.
    sealed=read(SOURCE/"artifact_manifest.json")
    for name,expected in sealed["files"].items():
        if sha(SOURCE/name)!=expected:
            raise RuntimeError(f"Sealed source changed after campaign: {name}")
    for name,expected in sealed["raw_data_final"].items():
        if sha(PROJECT/"data/raw"/name)!=expected:
            raise RuntimeError("Raw source changed after campaign")
    verified={"status":"PASS","checks":checks,"source_files_unchanged":len(sealed["files"]),
              "raw_data_unchanged":True,"code_fingerprints":fingerprints(),"test_accessed":False,
              "independent_replication_performed":False,"finished_utc":now()}
    write(HERE/"verification.json",verified)
    write_report(records,effects,decision,flat,verified)
    files={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob("*")) if p.is_file() and p.name not in ["artifact_manifest.json","artifact_manifest.json.tmp"] and not p.name.endswith(".tmp") and "__pycache__" not in p.parts}
    write(HERE/"artifact_manifest.json",{"status":"SEALED","files":files,"completed_utc":now(),"test_accessed":False})
    print("ABLATION VERIFIED",json_summary(decision),flush=True)


def json_summary(decision):
    return {"identity_given_uid_ap":decision["identity_given_uid_ap"],"uid_given_identity_ap":decision["uid_given_identity_ap"],
            "uid_detail":decision["uid_detail_seed42_ap"],"priorities":[x["family"] for x in decision["prioritized_candidates"]]}


def write_report(records,effects,decision,flat,verified):
    elapsed=(datetime.fromisoformat(verified["finished_utc"])-datetime.fromisoformat(read(HERE/"protocol.json")["started_utc"])).total_seconds()/60
    trained=sum(r["training_seconds_new"] for r in records.values())/60
    labels={"B00":"Common features","B10":"Common + identity","B01":"Common + UID, identity disabled","B11":"Common + identity + UID","B1M":"Common + identity + UID mean/standard deviation","B1C":"Common + identity + UID frequency/unique count"}
    rows=["| Seed | Feature condition | Columns | AP ↑ | ROC-AUC ↑ | Recall@1% ↑ | Recall@5% ↑ | Brier ↓ |",
          "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in flat:
        rows.append(f"| {r['seed']} | {labels[r['variant']]} | {r['feature_count']} | {r['average_precision']:.6f} | {r['roc_auc']:.6f} | {r['recall_top_1pct']:.6f} | {r['recall_top_5pct']:.6f} | {r['brier_score']:.6f} |")
    effect_rows=["| Paired AP effect | Seed42 | Seed43 |","|---|---:|---:|"]
    for name in ["identity|noUID","identity|UID","UID|noIdentity","UID|identity","identity_UID_interaction"]:
        values=[e["average_precision"] for e in effects if e["effect"]==name]
        effect_rows.append(f"| {name} | {values[0]:+.6f} | {values[1]:+.6f} |")
    priorities=", ".join(x["family"] for x in decision["prioritized_candidates"]) or "No complete information family received stable support in this study"
    details="\n".join(f"- {e['effect']}: AP {e['average_precision']:+.6f}." for e in effects if e["seed"]==42 and ("mean_std" in e["effect"] or "frequency_nunique" in e["effect"]))
    text=f"""# Matched ablations of identity and UID aggregates

All ten frozen conditions completed: nine new runs and one reused full seed42 model; unified verification PASS.
Transformer trial priorities supported by tree ablations: **{priorities}**.
Wall-clock time: {elapsed:.2f} minutes; total new training: {trained:.2f} minutes. Transformer training status: **NOT EXECUTED**.

## From inputs to predictions

This stage reused the prior 263-column TRAIN-fitted cache: common transformations included D time normalization, selected V columns and card/address grouped statistics.
Identity comprises device, browser and id_* fields. UID is a pseudo-customer key formed from card1, addr1 and day-D1, mapped to customer frequency, mean, standard deviation and unique-value counts.
Conditions differ only in retained/dropped columns. LightGBM outputs transaction fraud probabilities; AP, ROC-AUC, Top-K and Brier use the same frozen validation set.
All191 common fields remain, including existing card/address statistics. B00 is not the historical rich_all/HV baseline.

## Complete results and paired effects

{chr(10).join(rows)}

{chr(10).join(effect_rows)}

The identity|UID effect includes the cross-field uid_id_02_ct, so it is not an isolated contribution from raw identity fields.
Disabling identity also removes this cross-field. Interaction is AP(B11)-AP(B10)-AP(B01)+AP(B00),
an empirical model interaction under this protocol, not causal certification of information contributions.

UID subgroups (seed42; all other conditions and the recipe matched):

{details}

## Transformer transfer decision

Prioritized candidates and all original field names appear in TRANSFORMER_DECISION.json, including categorical/numeric identity groups.
UID mean/standard deviation uses 33 columns; frequency/unique counts use 13. A standalone UID transaction bundle should exclude uid_id_02_ct.
Numeric candidates use train-fitted imputation and normalization, retaining missing/unseen-customer flags; categorical identity uses train vocabularies and embeddings.
Literal UID and TransactionID are not input tokens. At this stage only tree evidence exists; improvements to FTTransformer or ExcelTransformer are not yet established.

Information with inconsistent effect signs across seeds remains a conditional candidate, not a stable gain. Report a negative result when adding it reduces AP.
UID subgroups currently have only seed42 evidence, so priorities specify the next controlled Transformer trial order, not a final Transformer selection.
Streaming applications require customer history with strictly t_j<t_i. This stage retains the prior whole-TRAIN fitting and frozen-validation mapping strategy.

## Reproducible evidence and limitations

- Seeds were42 and 43; model settings matched the prior stage except for seed. AP early-stop150, maximum 4000 trees; no search.
- The same413378 training/88581 validation split, field values, preprocessing and missing-sentinel interpretation were used; only feature columns changed.
- The source directory's {verified['source_files_unchanged']} sealed files and raw-data hashes matched before and after. The completed full seed42 baseline was not rerun.
- All nine new models were reloaded from disk for predictions and full metric recomputation. Prior full-baseline replay evidence was reused only with unchanged hashes.
- strata_metrics.csv retains every condition for seen/unseen UID, observed/missing id_12 and four joint subgroups. id_12 is a covariate proxy, not verified customer identity.
- Validation was reused across research selections, so effects are not independent final-test conclusions. Two seeds reflect training randomness, not statistical certification.
- Column sampling and early stopping can change with subsets; AP differences are empirical effects under the fixed training strategy.
- Recomputation and engineering checks were performed by the implementer, without independent subagent replication. All negative results remain; final test and official unlabeled test were not used.

Run in D:\\Year3_Mao_Projects\\sleep_loop\\fraud-detection-mvp using the existing environment:

```powershell
& .\\.venv\\Scripts\\python.exe -X utf8 -B -m experiments.identity_uid_ablation_20261011.preflight
& .\\.venv\\Scripts\\python.exe -X utf8 -B -m experiments.identity_uid_ablation_20261011.train
& .\\.venv\\Scripts\\python.exe -X utf8 -B -m experiments.identity_uid_ablation_20261011.analyze
```

protocol.json, feature_groups.json, preflight.json store frozen rules, field dependencies and input hashes;
runs/ stores models, epoch histories and validation predictions;ablation_results.csv, effects.csv, strata_metrics.csv store actual results;
verification.json and artifact_manifest.json store verification and final file fingerprints.
"""
    (HERE/"REPORT_ZH.md").write_text(text,encoding="utf-8")


if __name__=="__main__":
    analyze()
