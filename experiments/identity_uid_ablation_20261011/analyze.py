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
    labels={"B00":"共同特征","B10":"共同 + identity","B01":"共同 + UID，关闭 identity","B11":"共同 + identity + UID","B1M":"共同 + identity + UID均值/标准差","B1C":"共同 + identity + UID频率/唯一值数量"}
    rows=["| Seed | 特征条件 | 列数 | AP ↑ | ROC-AUC ↑ | Recall@1% ↑ | Recall@5% ↑ | Brier ↓ |",
          "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in flat:
        rows.append(f"| {r['seed']} | {labels[r['variant']]} | {r['feature_count']} | {r['average_precision']:.6f} | {r['roc_auc']:.6f} | {r['recall_top_1pct']:.6f} | {r['recall_top_5pct']:.6f} | {r['brier_score']:.6f} |")
    effect_rows=["| AP 配对增量 | Seed42 | Seed43 |","|---|---:|---:|"]
    for name in ["identity|noUID","identity|UID","UID|noIdentity","UID|identity","identity_UID_interaction"]:
        values=[e["average_precision"] for e in effects if e["effect"]==name]
        effect_rows.append(f"| {name} | {values[0]:+.6f} | {values[1]:+.6f} |")
    priorities="、".join(x["family"] for x in decision["prioritized_candidates"]) or "本次没有获得稳定支持的整体信息族"
    details="\n".join(f"- {e['effect']}：AP {e['average_precision']:+.6f}。" for e in effects if e["seed"]==42 and ("mean_std" in e["effect"] or "frequency_nunique" in e["effect"]))
    text=f"""# Identity 与 UID 聚合的匹配消融

十个冻结条件均完成：九个新训练、一个完整 seed42 模型复用；统一核验 PASS。
由树模型消融支持的 Transformer 试验优先级：**{priorities}**。
墙钟耗时 {elapsed:.2f} 分钟，新训练合计 {trained:.2f} 分钟。Transformer 训练状态：**NOT EXECUTED**。

## 从输入到预测

本轮直接复用上轮 263 列严格训练集拟合缓存：交易记录经过 D 时间归一化、精选 V 列、卡号/地址组合统计等共同转换；
identity 是设备、浏览器及 id_* 字段；UID 是 card1、addr1 与 day-D1 组合出的伪客户键，转为客户频率、均值、标准差和唯一值数量。
不同条件只改变列的保留与删除，LightGBM 输出交易欺诈概率，再在同一冻结验证集计算 AP、ROC-AUC、Top-K 与 Brier。
191 个共同字段始终保留；它们已含卡号/地址级别统计，B00 不是过去的 rich_all/HV 基线。

## 完整结果与配对增量

{chr(10).join(rows)}

{chr(10).join(effect_rows)}

identity|UID 的增量包含唯一交叉字段 uid_id_02_ct，它不能被解释为纯原始 identity 字段的独立贡献。
关闭 identity 时已同时移除该交叉字段。interaction 为 AP(B11)-AP(B10)-AP(B01)+AP(B00)，
是本协议下的经验模型交互量，不是信息贡献的因果认证。

UID 的细分（seed42；其他条件和配方相同）：

{details}

## Transformer 移植决策

优先候选和全部原始字段名已保存在 TRANSFORMER_DECISION.json，含类别 identity 与数值 identity 的字段划分。
UID 均值/标准差有33列，频率/唯一值数量13列；单独的 UID 交易包应删除 uid_id_02_ct。
数值候选采用训练集拟合的插补与归一化，并保留缺失/未见客户标记；identity 类别字段采用训练词表和 embedding。
不把 literal UID 或 TransactionID 当作 token 输入。仅有树模型证据，尚未证实这些字段能改善 FTTransformer 或 ExcelTransformer。

如果某类信息跨 seed 增量符号不一致，应作为条件候选保留，不声称稳定增益；如果增加后 AP 降低，报告负结果。
UID 细分目前只有 seed42，因此优先级是下一轮受控 Transformer 试验顺序，未完成 Transformer 的最终选择。
对于流式应用，应重新计算严格 t_j<t_i 的客户历史统计；本轮忠实沿用整个训练分区拟合、验证冻结映射的上轮策略。

## 可复现证据与限制

- 两个 seed 是42与43，模型参数与上轮一致，仅 seed 改变；AP early-stop150、最多4000棵树，无搜索。
- 同一413378训练/88581验证切分、同一字段值与预处理、同一缺失 sentinel 解释；只切换特征列。
- 源目录 {verified['source_files_unchanged']} 个封存文件及原始数据哈希在前后核验一致；未重新运行完成的 seed42 完整基线。
- 九个新模型逐一从磁盘重载预测，并复算全部指标；完整基线的先前重载核验在哈希未变的条件下复用。
- strata_metrics.csv 记录已见/未见UID、id_12 是否观测及四个联合子组的全部条件表现；id_12标记是协变量代理，不代表已验证客户身份。
- 同一验证集已用于多轮研究选择，增量不是独立最终测试结论。两种 seed 反映训练随机性，不构成统计认证。
- 列采样与 early-stop 会随子集改变；AP 差表示固定训练策略下的经验增量。
- 这是主实现者的复算与工程核验，没有进行独立子 agent 复算。全部负结果保留，最终测试和官方未标注 test 未使用。

在 D:\\Year3_Mao_Projects\\sleep_loop\\fraud-detection-mvp 中使用既有环境：

```powershell
& .\\.venv\\Scripts\\python.exe -X utf8 -B -m experiments.identity_uid_ablation_20261011.preflight
& .\\.venv\\Scripts\\python.exe -X utf8 -B -m experiments.identity_uid_ablation_20261011.train
& .\\.venv\\Scripts\\python.exe -X utf8 -B -m experiments.identity_uid_ablation_20261011.analyze
```

protocol.json、feature_groups.json、preflight.json 保存冻结规则、字段依赖和输入哈希；
runs/ 保存模型、逐轮历史与验证预测；ablation_results.csv、effects.csv、strata_metrics.csv 保存实际结果；
verification.json 和 artifact_manifest.json 保存核验与最终文件指纹。
"""
    (HERE/"REPORT_ZH.md").write_text(text,encoding="utf-8")


if __name__=="__main__":
    analyze()
