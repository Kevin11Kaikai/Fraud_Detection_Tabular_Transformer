"""Immutable input checks and local campaign artifacts."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OLD = PROJECT / "experiments/ft_upgrade_20261010"
DUAL = Path(r"D:\Year3_Mao_Projects\Tabular_Transformer\benchmark\experiments\champion_dual_axis_20261011")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprints():
    return {p.name: sha(p) for p in sorted(HERE.glob("*.py"))} | {"protocol.json": sha(HERE / "protocol.json")}


def metrics(y, p, ids):
    y, p, ids = np.asarray(y), np.asarray(p, dtype=float), np.asarray(ids)
    if not len(y) == len(p) == len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("Prediction alignment or uniqueness failed")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probabilities")
    order = np.lexsort((ids, -p))
    k1, k5 = math.ceil(.01 * len(y)), math.ceil(.05 * len(y))
    c1, c5, nf = int(y[order[:k1]].sum()), int(y[order[:k5]].sum()), int(y.sum())
    return {"average_precision": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "recall_top_1pct": c1/nf, "precision_top_1pct": c1/k1,
            "recall_top_5pct": c5/nf, "brier_score": float(np.mean((p-y)**2)),
            "captured_top_1pct": c1, "captured_top_5pct": c5,
            "n_rows": len(y), "n_fraud": nf, "top_1pct_count": k1}
