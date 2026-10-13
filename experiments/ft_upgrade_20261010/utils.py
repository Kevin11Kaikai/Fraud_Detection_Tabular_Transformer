"""Local campaign helpers; never modify original benchmark artifacts."""
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


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprints():
    return {p.name: digest(p) for p in sorted(HERE.glob("*.py"))} | {
        "protocol.json": digest(HERE / "protocol.json")}


def scores(y, p, ids):
    y, p, ids = np.asarray(y), np.asarray(p, dtype=float), np.asarray(ids)
    if not (len(y) == len(p) == len(ids)) or len(np.unique(ids)) != len(ids):
        raise ValueError("Unaligned or duplicate prediction rows")
    if set(np.unique(y)) != {0, 1} or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid labels/probabilities")
    ranked = np.lexsort((ids, -p))
    k1, k5 = math.ceil(.01 * len(y)), math.ceil(.05 * len(y))
    c1, c5 = int(y[ranked[:k1]].sum()), int(y[ranked[:k5]].sum())
    return {"roc_auc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "recall_top_1pct": c1 / int(y.sum()), "precision_top_1pct": c1 / k1,
            "recall_top_5pct": c5 / int(y.sum()), "brier_score": float(np.mean((p-y)**2)),
            "captured_top_1pct": c1, "captured_top_5pct": c5,
            "n_rows": len(y), "n_fraud": int(y.sum()), "top_1pct_count": k1}


def verify_baseline():
    manifest = read(PROJECT / "reports/artifact_manifest.json")
    mismatches = [name for name, expected in manifest["files"].items()
                  if not (PROJECT / name).is_file() or digest(PROJECT / name) != expected]
    if mismatches:
        raise RuntimeError(f"Original baseline changed: {mismatches}")
    return {"status": "PASS", "file_count": len(manifest["files"]),
            "manifest_sha256": digest(PROJECT / "reports/artifact_manifest.json"),
            "source_data_sha256": manifest["raw_data_sha256"]}
