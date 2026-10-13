"""Audited pandas port of the author's public XGB96 features.

No labels are accepted. Fit-scope is explicit; all maps are frozen at fitting.
The batch mode must match the actual archived source on a real-data fixture.
"""
from __future__ import annotations

import ast
import warnings

import numpy as np
import pandas as pd

from .common import HERE, read


def source_constants():
    source = "".join(read(HERE / "references/cdeotte__xgb-fraud-with-magic-0-9600.ipynb")["cells"][3]["source"])
    tree = ast.parse(source)
    allowed = {"str_type", "cols", "v"}
    selected = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AugAssign) else []
        if targets and all(isinstance(t, ast.Name) and t.id in allowed for t in targets):
            selected.append(node)
    env = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "audited-source-constants", "exec"), env)
    return env["cols"], env["str_type"], env["v"]


TRANSACTION_COLUMNS, STRING_COLUMNS, V_COLUMNS = source_constants()
DROP = ["TransactionDT", "D6", "D7", "D8", "D9", "D12", "D13", "D14",
        "DT_M", "day", "uid", "C3", "M5", "id_08", "id_33", "card4",
        "id_07", "id_14", "id_21", "id_30", "id_32", "id_34"] + [f"id_{i}" for i in range(22, 28)]


def build(train, validation, scope):
    if scope not in {"train_only", "batch"}:
        raise ValueError(scope)
    forbidden = {"isFraud", "y", "target", "oof"}
    if forbidden.intersection(train.columns) or forbidden.intersection(validation.columns):
        raise ValueError("Target/prediction columns cannot enter feature construction")
    if list(train.columns) != list(validation.columns):
        raise ValueError("Raw schema mismatch")
    n_train = len(train)
    frame = pd.concat([train, validation], axis=0, copy=True)
    # Canonical author input dtype prevents concatenated fixture rows from
    # promoting all earlier float32 features and changing their arithmetic.
    for name in frame.select_dtypes(include=["number"]).columns:
        frame[name] = frame[name].astype("float32")
    fit_n = n_train if scope == "train_only" else len(frame)
    diagnostics = {"scope": scope, "fit_rows": fit_n, "raw_columns": len(frame.columns),
                   "v_columns": len(V_COLUMNS), "unknown_static_categories": {}}

    def fit_series(name):
        return frame[name].iloc[:fit_n]

    def label_encode(name, initial=False):
        _, levels = fit_series(name).factorize(sort=True)
        codes = levels.get_indexer(frame[name])
        frame[name] = codes.astype("int32" if codes.max(initial=0) > 32000 else "int16")
        if initial:
            diagnostics["unknown_static_categories"][name] = int((codes[n_train:] == -1).sum())

    def frequency(names):
        for name in names:
            counts = fit_series(name).value_counts(dropna=True, normalize=True).to_dict()
            counts[-1] = -1
            frame[name + "_FE"] = frame[name].map(counts).astype("float32").fillna(-1)

    def combine(a, b):
        name = a + "_" + b
        frame[name] = frame[a].astype(str) + "_" + frame[b].astype(str)
        label_encode(name)

    def aggregate(values, groups, stats):
        for value in values:
            for group in groups:
                fit_values = fit_series(value).mask(fit_series(value) == -1)
                grouped = fit_values.groupby(fit_series(group), sort=True)
                for stat in stats:
                    mapping = grouped.agg(stat).to_dict()
                    name = value + "_" + group + "_" + stat
                    frame[name] = frame[group].map(mapping).astype("float32").fillna(-1)

    def unique(values):
        for value in values:
            mapping = fit_series(value).groupby(fit_series("uid"), sort=True).nunique().to_dict()
            frame["uid_" + value + "_ct"] = frame["uid"].map(mapping).astype("float32").fillna(-1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        for i in range(1, 16):
            if i not in [1, 2, 3, 5, 9]:
                frame[f"D{i}"] = frame[f"D{i}"] - frame.TransactionDT / np.float32(86400)
        for name in list(frame.columns):
            if isinstance(frame[name].dtype, pd.CategoricalDtype) or frame[name].dtype == "object":
                label_encode(name, initial=True)
            elif name not in ["TransactionAmt", "TransactionDT"]:
                minimum = np.float32(np.min((frame[name].iloc[:n_train].min(), frame[name].iloc[n_train:].min()))) if scope == "batch" else np.float32(fit_series(name).min())
                frame[name] = (frame[name] - minimum).fillna(-1)
        frame["cents"] = (frame.TransactionAmt - np.floor(frame.TransactionAmt)).astype("float32")
        frequency(["addr1", "card1", "card2", "card3", "P_emaildomain"])
        combine("card1", "addr1")
        combine("card1_addr1", "P_emaildomain")
        frequency(["card1_addr1", "card1_addr1_P_emaildomain"])
        aggregate(["TransactionAmt", "D9", "D11"], ["card1", "card1_addr1", "card1_addr1_P_emaildomain"], ["mean", "std"])
        dates = pd.Timestamp("2017-11-30") + pd.to_timedelta(frame.TransactionDT, unit="s")
        frame["DT_M"] = (dates.dt.year - 2017) * 12 + dates.dt.month
        frame["day"] = frame.TransactionDT / (24*60*60)
        frame["uid"] = frame.card1_addr1.astype(str) + "_" + np.floor(frame.day - frame.D1).astype(str)
        frequency(["uid"])
        aggregate(["TransactionAmt", "D4", "D9", "D10", "D15"], ["uid"], ["mean", "std"])
        aggregate([f"C{i}" for i in range(1, 15) if i != 3], ["uid"], ["mean"])
        aggregate([f"M{i}" for i in range(1, 10)], ["uid"], ["mean"])
        unique(["P_emaildomain", "dist1", "DT_M", "id_02", "cents"])
        aggregate(["C14"], ["uid"], ["std"])
        unique(["C13", "V314"])
        unique(["V127", "V136", "V309", "V307", "V320"])
        frame["outsider15"] = (np.abs(frame.D1 - frame.D15) > 3).astype("int8")
    diagnostics["validation_uid_seen_in_train_fraction"] = float(frame.uid.iloc[n_train:].isin(set(frame.uid.iloc[:n_train])).mean())
    diagnostics["uid_count_train"] = int(frame.uid.iloc[:n_train].nunique())
    frame = frame.drop(columns=DROP)
    diagnostics["output_columns"] = len(frame.columns)
    diagnostics["column_names"] = list(frame.columns)
    if frame.select_dtypes(include=["object", "category"]).shape[1]:
        raise ValueError("Unencoded fields remain")
    return frame.iloc[:n_train].copy(), frame.iloc[n_train:].copy(), diagnostics
