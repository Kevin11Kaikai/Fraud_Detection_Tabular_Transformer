"""Read admitted cached-data rows, audit against source, then prepare features."""
from __future__ import annotations

import contextlib
import gc
import io
import time
import warnings

import numpy as np
import pandas as pd

from .common import HERE, OLD, PROJECT, fingerprints, now, read, sha, write
from .features import STRING_COLUMNS, TRANSACTION_COLUMNS, build


def load_raw():
    protocol = read(HERE / "protocol.json")
    transaction = PROJECT / "data/raw/train_transaction.csv"
    identity = PROJECT / "data/raw/train_identity.csv"
    for path, expected in [(transaction, protocol["raw_transaction_sha256"]), (identity, protocol["raw_identity_sha256"])]:
        if sha(path) != expected:
            raise RuntimeError(f"Raw input fingerprint mismatch: {path}")
    n_train = protocol["train_rows"]
    n_total = n_train + protocol["validation_rows"]
    numeric = {c: "float32" for c in TRANSACTION_COLUMNS if c != "TransactionID"}
    numeric.update({c: "category" for c in STRING_COLUMNS})
    numeric["TransactionID"] = "int64"
    numeric["isFraud"] = "uint8"
    frame = pd.read_csv(transaction, index_col="TransactionID", usecols=TRANSACTION_COLUMNS + ["isFraud"], dtype=numeric, nrows=n_total)
    ids = frame.index.to_numpy().copy()
    y = frame.pop("isFraud").to_numpy()
    dt = frame.TransactionDT.to_numpy().copy()
    if len(frame) != n_total or not frame.index.is_unique or np.any(np.diff(dt) < 0):
        raise RuntimeError("Frozen chronological order failed")
    identity_columns = pd.read_csv(identity, nrows=0).columns.tolist()
    dtype_id = {c: "category" if c in STRING_COLUMNS else "float32" for c in identity_columns}
    dtype_id["TransactionID"] = "int64"
    parts = []
    # Discard other identity covariates before joining; no identity labels exist.
    for part in pd.read_csv(identity, dtype=dtype_id, chunksize=50000, index_col="TransactionID"):
        parts.append(part.loc[part.index.isin(frame.index)].copy())
    ident = pd.concat(parts)
    for c in STRING_COLUMNS:
        if c in ident:
            ident[c] = ident[c].astype("category")
    frame = frame.join(ident, how="left", sort=False).reindex(ids)
    if not np.array_equal(frame.index.to_numpy(), ids):
        raise RuntimeError("Identity join changed row order")
    for name, section in [("train", slice(0, n_train)), ("validation", slice(n_train, n_total))]:
        with np.load(OLD / f"cache/rich_all/{name}.npz") as cached:
            if not np.array_equal(cached["ids"], ids[section]) or not np.array_equal(cached["y"], y[section]) or not np.array_equal(cached["dt"], dt[section]):
                raise RuntimeError(f"Existing split/label alignment failed: {name}")
    split = {"train_rows": n_train, "validation_rows": n_total-n_train,
             "train_fraud": int(y[:n_train].sum()), "validation_fraud": int(y[n_train:].sum()),
             "train_id_first_last": [int(ids[0]), int(ids[n_train-1])],
             "validation_id_first_last": [int(ids[n_train]), int(ids[-1])],
             "train_dt_range": [float(dt[0]), float(dt[n_train-1])],
             "validation_dt_range": [float(dt[n_train]), float(dt[-1])],
             "raw_rows_with_identity": int(frame.id_01.notna().sum()),
             "ids_sha256": __import__("hashlib").sha256(ids.tobytes()).hexdigest(),
             "test_accessed": False, "frozen_cache_match": True,
             "input_hashes": {"transaction": sha(transaction), "identity": sha(identity)}}
    return frame.iloc[:n_train].copy(), frame.iloc[n_train:].copy(), ids, y, split


def run_audits(train, valid):
    # Include rare identity fields and real validation customers in a bounded fixture.
    a = train.iloc[:4096].copy()
    b = valid.iloc[:2048].copy()
    nb = read(HERE / "references/cdeotte__xgb-fraud-with-magic-0-9600.ipynb")
    expected_a, expected_b = a.copy(), b.copy()
    env = {"np": np, "pd": pd, "gc": gc, "X_train": expected_a, "X_test": expected_b}
    output = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(output):
        warnings.simplefilter("ignore")
        for cell in [7, 9, 11, 13, 22, 29, 31, 32]:
            if cell == 32:
                env["X_train"]["oof"] = 0
                env["X_test"]["oof"] = 0
            src = "".join(nb["cells"][cell]["source"])
            src = "\n".join(line for line in src.splitlines() if not line.startswith("%"))
            src = src.replace("np.str(", "str(")
            exec(compile(src, f"audited-author-feature-cell-{cell}", "exec"), env)
    actual_a, actual_b, _ = build(a, b, "batch")
    cols = env["cols"]
    if list(actual_a.columns) != cols:
        raise AssertionError("Author feature order mismatch")
    maximum = 0.
    for expected, actual in [(env["X_train"][cols], actual_a), (env["X_test"][cols], actual_b)]:
        lhs, rhs = expected.to_numpy(dtype=np.float32), actual.to_numpy(dtype=np.float32)
        np.testing.assert_allclose(lhs, rhs, rtol=2e-6, atol=2e-6, equal_nan=True)
        delta = np.abs(lhs-rhs)
        maximum = max(maximum, float(np.nanmax(delta)))
    # A training-only transformation must not depend on other validation rows.
    strict_a, strict_b, _ = build(a, b, "train_only")
    c = b.iloc[:128].copy()
    c["TransactionAmt"] = c.TransactionAmt + 99999
    c["D1"] = 0
    c.index = c.index + 99999999
    changed_a, changed_b, _ = build(a, pd.concat([b, c]), "train_only")
    np.testing.assert_array_equal(strict_a.to_numpy(), changed_a.to_numpy())
    np.testing.assert_array_equal(strict_b.to_numpy(), changed_b.iloc[:len(b)].to_numpy())
    try:
        build(a.assign(isFraud=0), b.assign(isFraud=1), "batch")
    except ValueError:
        label_rejected = True
    else:
        raise AssertionError("Feature constructor accepted labels")
    report = {"status": "PASS", "source_parity_rows": len(a)+len(b), "source_parity_columns": len(cols),
              "source_max_abs_error": maximum, "validation_perturbation_invariance": "PASS",
              "label_input_rejected": label_rejected, "executed_author_cells": [7,9,11,13,22,29,31,32],
              "source_notebook_sha256": sha(HERE / "references/cdeotte__xgb-fraud-with-magic-0-9600.ipynb"),
              "finished_utc": now()}
    write(HERE / "audit.json", report)
    print("AUDIT", report, flush=True)
    return report


def prepare():
    start = time.perf_counter()
    train, valid, ids, y, split = load_raw()
    run_audits(train, valid)
    write(HERE / "split_receipt.json", split)
    n_train = len(train)
    dest = HERE / "derived"
    dest.mkdir(exist_ok=True)
    np.savez_compressed(dest / "labels_ids.npz", ids_train=ids[:n_train], ids_validation=ids[n_train:], y_train=y[:n_train], y_validation=y[n_train:])
    for scope in ["train_only", "batch"]:
        path = dest / scope
        if (path / "manifest.json").exists():
            receipt = read(path / "manifest.json")
            if receipt["code_fingerprints"] != fingerprints():
                raise RuntimeError("Existing representation was made with different code; do not silently overwrite")
            for name, expected in receipt["array_hashes"].items():
                if sha(path/name) != expected:
                    raise RuntimeError("Existing feature fingerprint mismatch")
            continue
        path.mkdir(exist_ok=True)
        tick = time.perf_counter()
        tr, va, info = build(train, valid, scope)
        for name, frame in [("train", tr), ("validation", va)]:
            array = frame.to_numpy(dtype="float32")
            if np.isinf(array).any():
                raise ValueError("Infinite feature value")
            np.save(path/f"{name}.npy", array)
            del array
        info.update({"status": "PREPARED", "preparation_seconds": time.perf_counter()-tick,
                     "array_hashes": {f"{name}.npy": sha(path/f"{name}.npy") for name in ["train", "validation"]},
                     "split_receipt_sha256": sha(HERE/"split_receipt.json"),
                     "code_fingerprints": fingerprints(), "finished_utc": now()})
        write(path / "manifest.json", info)
        print("PREPARED", scope, info["output_columns"], info["preparation_seconds"], flush=True)
        del tr, va
        gc.collect()
    write(HERE/"preparation.json", {"status": "PASS", "elapsed_seconds": time.perf_counter()-start, "finished_utc": now()})


if __name__ == "__main__":
    prepare()
