"""One input-integrity, family-dependency and low-cost CPU smoke gate."""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import psutil

from .common import HERE,PROJECT,SOURCE,fingerprints,groups,metrics,now,read,sha,write


def run():
    start=time.perf_counter()
    if (HERE/"preflight.json").exists():
        prior=read(HERE/"preflight.json")
        if prior["code_fingerprints"]!=fingerprints():
            raise RuntimeError("Frozen code changed; cannot silently overwrite preflight")
        return prior
    sealed=read(SOURCE/"artifact_manifest.json")
    checked=[]
    for name,expected in sealed["files"].items():
        if sha(SOURCE/name)!=expected:
            raise RuntimeError(f"Sealed source file changed: {name}")
        checked.append(name)
    for name,expected in sealed["raw_data_final"].items():
        if sha(PROJECT/"data/raw"/name)!=expected:
            raise RuntimeError("Raw data changed")
    schema=read(SOURCE/"derived/train_only/manifest.json")
    feature_groups=groups(schema["column_names"])
    write(HERE/"feature_groups.json",feature_groups)
    g=feature_groups["variants"]
    assert set(g["B00"]).issubset(g["B10"])
    assert set(g["B00"]).issubset(g["B01"])
    assert set(g["B11"])==set(g["B10"])|set(g["B01"])|{"uid_id_02_ct"}
    assert set(g["B1M"])|set(g["B1C"])==set(g["B11"])
    assert set(g["B1M"])&set(g["B1C"])==set(g["B10"])
    for variant in ["B00","B01"]:
        assert not set(feature_groups["identity"]).intersection(g[variant])
        assert "uid_id_02_ct" not in g[variant]
    for variant in ["B00","B10"]:
        assert not set(feature_groups["uid"]).intersection(g[variant])
    with np.load(SOURCE/"derived/labels_ids.npz") as archive:
        labels={k:archive[k] for k in archive.files}
    if (len(labels["y_train"]),len(labels["y_validation"]))!=(413378,88581):
        raise RuntimeError("Frozen split lengths mismatch")
    source_config=read(SOURCE/"protocol.json")["lgb"]
    reused=read(SOURCE/"runs/lgb_train_only/result.json")
    if reused["config"]!=source_config:
        raise RuntimeError("Reused baseline recipe mismatch")
    train=np.load(SOURCE/"derived/train_only/train.npy",mmap_mode="r")
    indices=[schema["column_names"].index(n) for n in g["B00"]]
    smoke_x=train[:40000,indices].copy()
    smoke_x[smoke_x==-1]=np.nan
    smoke_cfg=source_config|{"n_estimators":25}
    model=lgb.LGBMClassifier(**smoke_cfg)
    tick=time.perf_counter()
    model.fit(smoke_x[:32000],labels["y_train"][:32000])
    p=model.predict_proba(smoke_x[32000:])[:,1]
    if not np.isfinite(p).all():
        raise RuntimeError("CPU smoke invalid probabilities")
    smoke_seconds=time.perf_counter()-tick
    # Verify source full predictions without retraining.
    with np.load(SOURCE/"runs/lgb_train_only/predictions_validation.npz") as predictions:
        np.testing.assert_array_equal(predictions["ids"],labels["ids_validation"])
        np.testing.assert_array_equal(predictions["y"],labels["y_validation"])
        computed=metrics(predictions["y"],predictions["p"],predictions["ids"])
        for name,value in reused["metrics"].items():
            np.testing.assert_allclose(computed[name],value,rtol=0,atol=1e-12)
    for filename,digest in reused["artifact_hashes"].items():
        if sha(SOURCE/"runs/lgb_train_only"/filename)!=digest:
            raise RuntimeError("Reusable full model changed")
    # Strata use only cached covariates, never labels.
    valid=np.load(SOURCE/"derived/train_only/validation.npy",mmap_mode="r")
    uid_seen=valid[:,schema["column_names"].index("uid_FE")]!=-1
    identity_marker_observed=valid[:,schema["column_names"].index("id_12")]!=-1
    np.testing.assert_allclose(uid_seen.mean(),schema["validation_uid_seen_in_train_fraction"],rtol=0,atol=1e-12)
    np.savez_compressed(HERE/"strata_validation.npz",uid_seen=uid_seen,identity_marker_observed=identity_marker_observed,ids=labels["ids_validation"])
    trials=[f"seed{seed}_{variant}" for seed in [42,43] for variant in ["B00","B10","B01","B11"]]
    trials.extend(["seed42_B1M","seed42_B1C"])
    receipt={"status":"PASS","completed_utc":now(),"elapsed_seconds":time.perf_counter()-start,
             "sealed_source_manifest_sha256":sha(SOURCE/"artifact_manifest.json"),
             "sealed_source_files_verified":len(checked),"source_array_hashes":schema["array_hashes"],
             "source_schema_sha256":sha(SOURCE/"derived/train_only/manifest.json"),
             "source_labels_ids_sha256":sha(SOURCE/"derived/labels_ids.npz"),
             "feature_groups_sha256":sha(HERE/"feature_groups.json"),
             "strata_sha256":sha(HERE/"strata_validation.npz"),
             "variant_widths":{k:len(v) for k,v in g.items()},"dependency_checks":"PASS",
             "cpu_smoke":{"status":"PASS","rounds":25,"train_rows":32000,"holdout_rows":8000,
                          "holdout_from":"training partition only","seconds":smoke_seconds,"actual_device":"CPU, 8 threads"},
             "environment":{"lightgbm":lgb.__version__,"available_ram_gib":psutil.virtual_memory().available/2**30},
             "trials":trials,"new_trials":9,"reused_trials":1,"lgb_config":source_config,
             "uid_seen_fraction":float(uid_seen.mean()),"identity_id12_marker_fraction":float(identity_marker_observed.mean()),
             "code_fingerprints":fingerprints(),"test_accessed":False}
    write(HERE/"preflight.json",receipt)
    print("PREFLIGHT PASS",receipt["variant_widths"],"smoke",smoke_seconds,"seconds",flush=True)
    return receipt


if __name__=="__main__":
    run()
