from __future__ import annotations

import json
from pathlib import Path

from ..kaggle_fraudsquad_reproduction_20261011.common import metrics, now, read, sha, write

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SOURCE = HERE.parent/"kaggle_fraudsquad_reproduction_20261011"


def fingerprints():
    return {p.name:sha(p) for p in sorted(HERE.glob("*.py"))} | {"protocol.json":sha(HERE/"protocol.json")}


def groups(names):
    identity = [n for n in names if n.startswith("id_") or n in {"DeviceType", "DeviceInfo"}]
    uid = [n for n in names if n.startswith("uid_") or "_uid_" in n]
    joint = ["uid_id_02_ct"]
    statistics = [n for n in uid if n.endswith("_mean") or n.endswith("_std")]
    counts = [n for n in uid if n not in statistics]
    common = [n for n in names if n not in set(identity+uid)]
    if (len(common),len(identity),len(uid),len(statistics),len(counts)) != (191,26,46,33,13):
        raise RuntimeError("Unexpected source feature family sizes")
    if not set(joint).issubset(counts) or len(set(common+identity+uid))!=len(names):
        raise RuntimeError("Unexpected dependency/schema")
    sets = {"B00":set(common),"B10":set(common+identity),
            "B01":set(common+[n for n in uid if n not in joint]),
            "B11":set(names),"B1M":set(common+identity+statistics),
            "B1C":set(common+identity+counts)}
    ordered = {k:[n for n in names if n in wanted] for k,wanted in sets.items()}
    return {"common":common,"identity":identity,"uid":uid,"identity_uid_joint":joint,
            "uid_mean_std":statistics,"uid_frequency_nunique":counts,"variants":ordered}


def load(variant):
    manifest=read(HERE/"preflight.json")
    if manifest["code_fingerprints"]!=fingerprints():
        raise RuntimeError("Ablation code changed after freeze")
    schema=read(SOURCE/"derived/train_only/manifest.json")
    for path,expected in [(SOURCE/"derived/train_only/manifest.json",manifest["source_schema_sha256"]),
                          (SOURCE/"derived/labels_ids.npz",manifest["source_labels_ids_sha256"]),
                          (HERE/"feature_groups.json",manifest["feature_groups_sha256"])]:
        if sha(path)!=expected:
            raise RuntimeError("Frozen schema, labels or column definitions changed")
    for filename,expected in schema["array_hashes"].items():
        if sha(SOURCE/"derived/train_only"/filename)!=expected:
            raise RuntimeError("Source representation changed")
    feature_groups=read(HERE/"feature_groups.json")
    names=feature_groups["variants"][variant]
    indices=[schema["column_names"].index(n) for n in names]
    tr=np_load(SOURCE/"derived/train_only/train.npy")[:,indices].copy()
    va=np_load(SOURCE/"derived/train_only/validation.npy")[:,indices].copy()
    import numpy as np
    tr[tr==-1]=np.nan
    va[va==-1]=np.nan
    with np.load(SOURCE/"derived/labels_ids.npz") as archive:
        labels={k:archive[k] for k in archive.files}
    return tr,va,labels,names


def np_load(path):
    import numpy as np
    return np.load(path,mmap_mode="r")
