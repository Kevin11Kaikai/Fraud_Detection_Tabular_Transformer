"""Chronological partitions and training-only preprocessing. No test selection."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, mode_paths, read_json, sha256, utc_now, write_json

CATEGORICAL = ["ProductCD", *[f"card{i}" for i in range(1, 7)], "addr1", "addr2",
               "P_emaildomain", "R_emaildomain", *[f"M{i}" for i in range(1, 10)]]
NUMERIC = ["TransactionAmt", "dist1", "dist2", *[f"C{i}" for i in range(1, 15)],
           *[f"D{i}" for i in range(1, 16)]]
FEATURES = NUMERIC + CATEGORICAL


def chronological_split(frame):
    """Stable time order; CSV order breaks equal-time ties, never the label."""
    if frame["TransactionDT"].isna().any():
        raise ValueError("TransactionDT has missing values")
    frame = frame.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    n = len(frame)
    boundaries = (int(n * 0.70), int(n * 0.85))
    if not (0 < boundaries[0] < boundaries[1] < n):
        raise ValueError("Insufficient rows for 70/15/15 split")
    return [frame.iloc[:boundaries[0]].copy(),
            frame.iloc[boundaries[0]:boundaries[1]].copy(),
            frame.iloc[boundaries[1]:].copy()]


def numeric_values(frame, names, log_features):
    values = frame[names].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    values[~np.isfinite(values)] = np.nan
    for j, name in enumerate(names):
        if name in log_features:
            if np.any(values[:, j] < 0):
                raise ValueError(f"{name} contains negative values; log1p invalid")
            values[:, j] = np.log1p(values[:, j])
    return values


class Preprocessor:
    """Serializes plain JSON state; fitted once using training covariates only."""
    def __init__(self, config, state=None):
        self.config = config
        self.state = state

    def fit(self, train):
        numeric = [name for name in NUMERIC if train[name].nunique(dropna=True) > 1]
        categorical = [name for name in CATEGORICAL if train[name].nunique(dropna=True) > 1]
        if not numeric or not categorical:
            raise ValueError("No usable numeric or categorical training features")
        values = numeric_values(train, numeric, self.config["numeric_log1p"])
        medians = np.nanmedian(values, axis=0)
        rates = np.isnan(values).mean(axis=0)
        indicator_indices = [j for j, rate in enumerate(rates)
                             if self.config["missing_indicator_min_rate"] <= rate
                             <= self.config["missing_indicator_max_rate"]]
        imputed = np.where(np.isnan(values), medians, values)
        means = imputed.mean(axis=0)
        stds = imputed.std(axis=0)
        stds[stds < 1e-8] = 1.0
        vocabularies = {}
        for name in categorical:
            counts = train[name].astype("string").value_counts(dropna=True)
            eligible = [(str(value), int(count)) for value, count in counts.items()
                        if count >= self.config["categorical_min_count"]]
            eligible.sort(key=lambda entry: (-entry[1], entry[0]))
            vocabularies[name] = {value: j + 2 for j, (value, _) in
                                  enumerate(eligible[:self.config["categorical_max_values"]])}
        self.state = {"fit_rows": len(train), "fit_partition": "train",
                      "numeric": numeric, "categorical": categorical,
                      "dropped_training_constant_or_missing": [x for x in FEATURES
                                                               if x not in numeric + categorical],
                      "numeric_log1p": self.config["numeric_log1p"],
                      "medians": medians.tolist(), "means": means.tolist(), "stds": stds.tolist(),
                      "indicator_indices": indicator_indices, "missing_rates": rates.tolist(),
                      "numeric_output_names": numeric + [f"{numeric[j]}__missing" for j in indicator_indices],
                      "vocabularies": vocabularies,
                      "cat_cardinalities": [len(vocabularies[name]) + 2 for name in categorical]}
        return self

    def transform(self, frame):
        if self.state is None:
            raise RuntimeError("Preprocessor is not fitted")
        state = self.state
        values = numeric_values(frame, state["numeric"], state["numeric_log1p"])
        missing = np.isnan(values)
        imputed = np.where(missing, np.asarray(state["medians"]), values)
        indicators = missing[:, state["indicator_indices"]].astype(np.float32)
        x_lgb_num = np.column_stack([imputed, indicators]).astype(np.float32)
        x_num = np.column_stack([(imputed - state["means"]) / state["stds"], indicators]).astype(np.float32)
        codes = []
        for name in state["categorical"]:
            series = frame[name].astype("string")
            code = series.map(state["vocabularies"][name]).fillna(0).to_numpy(dtype=np.int64)
            code[series.isna().to_numpy()] = 1
            codes.append(code)
        x_cat = np.column_stack(codes)
        if not np.isfinite(x_num).all() or not np.isfinite(x_lgb_num).all():
            raise ValueError("Nonfinite preprocessed numeric values")
        return {"x_num": x_num, "x_lgb_num": x_lgb_num, "x_cat": x_cat}


def prepare(csv_path, mode, config):
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Real Kaggle training data required: {csv_path}")
    header = pd.read_csv(csv_path, nrows=0).columns
    required = ["TransactionID", "TransactionDT", "isFraud", *FEATURES]
    absent = [name for name in required if name not in header]
    if absent:
        raise ValueError(f"Training file missing expected fields: {absent}")
    frame = pd.read_csv(csv_path, usecols=required, dtype={name: "string" for name in CATEGORICAL})
    if frame["TransactionID"].isna().any() or frame["TransactionID"].duplicated().any():
        raise ValueError("TransactionID must be unique and nonmissing")
    if frame["isFraud"].isna().any() or not set(frame["isFraud"].unique()) <= {0, 1}:
        raise ValueError("isFraud must contain only binary nonmissing labels")
    frame = frame.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    if mode == "smoke":
        frame = frame.iloc[:config["smoke_rows"]].copy()
    parts = chronological_split(frame)
    preprocessor = Preprocessor(config).fit(parts[0])
    prepared, _, _ = mode_paths(mode)
    if (prepared / "manifest.json").exists():
        raise FileExistsError(f"Prepared data exists; reuse it or choose a fresh output directory: {prepared}")
    write_json(prepared / "preprocessor.json", preprocessor.state)
    split_info = {}
    for name, part in zip(["train", "validation", "test"], parts):
        arrays = preprocessor.transform(part)
        arrays.update({"transaction_id": part["TransactionID"].to_numpy(np.int64),
                       "transaction_dt": part["TransactionDT"].to_numpy(np.int64)})
        labels = part["isFraud"].to_numpy(np.int8)
        if name != "test":
            if len(np.unique(labels)) != 2:
                raise ValueError(f"{name} partition lacks both classes")
            arrays["y"] = labels
            np.savez_compressed(prepared / f"{name}.npz", **arrays)
        else:
            np.savez_compressed(prepared / "test_features.npz", **arrays)
            np.savez_compressed(prepared / "test_labels.npz", y=labels,
                                transaction_id=arrays["transaction_id"])
        split_info[name] = {"rows": len(part), "dt_min": int(part["TransactionDT"].min()),
                            "dt_max": int(part["TransactionDT"].max())}
    artifacts = {path.name: sha256(path) for path in prepared.glob("*.npz")}
    manifest = {"created_utc": utc_now(), "mode": mode, "source_csv": str(csv_path),
                "source_sha256": sha256(csv_path), "rows_used": len(frame),
                "splits": split_info, "tie_rule": "stable CSV order within TransactionDT ties",
                "boundary_time_ties": [split_info["train"]["dt_max"] == split_info["validation"]["dt_min"],
                                       split_info["validation"]["dt_max"] == split_info["test"]["dt_min"]],
                "config_sha256": sha256(ROOT / "configs" / "mvp.json"),
                "preprocessor_sha256": sha256(prepared / "preprocessor.json"),
                "artifact_sha256": artifacts}
    write_json(prepared / "manifest.json", manifest)
    write_json(ROOT / "results" / mode / "data_manifest.json", manifest)
    write_json(ROOT / "results" / mode / "selected_features.json",
               {key: preprocessor.state[key] for key in ["numeric", "categorical",
                "numeric_output_names", "dropped_training_constant_or_missing", "cat_cardinalities"]})
    print(f"PREPARED {mode}: {split_info}; {len(preprocessor.state['numeric']) + len(preprocessor.state['categorical'])} raw features", flush=True)
    return manifest


def lightgbm_frame(arrays, state):
    frame = pd.DataFrame(arrays["x_lgb_num"], columns=state["numeric_output_names"])
    for j, (name, count) in enumerate(zip(state["categorical"], state["cat_cardinalities"])):
        frame[name] = pd.Categorical(arrays["x_cat"][:, j], categories=range(count))
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "raw" / "train_transaction.csv"))
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    args = parser.parse_args()
    prepare(args.data, args.mode, read_json(ROOT / "configs" / "mvp.json"))
