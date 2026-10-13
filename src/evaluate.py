"""Same held-out rows, AP (not trapezoidal PR-AUC), and fixed investigation budgets."""
from __future__ import annotations

import argparse
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)

from .common import (ROOT, load_partition, mode_paths, probability_check, read_json,
                     sha256, utc_now, write_json)

SCORE_COLUMNS = ["roc_auc", "average_precision", "recall_top_1pct", "precision_top_1pct",
                 "recall_top_5pct", "brier_score"]


def rank_indices(probability, transaction_id):
    # Fixed deterministic tie-break; TransactionID is metadata, never a feature.
    return np.lexsort((np.asarray(transaction_id), -np.asarray(probability)))


def top_budget(y, probability, transaction_id, fraction):
    if not (0 < fraction <= 1):
        raise ValueError("Investigation fraction must be in (0,1]")
    k = max(1, math.ceil(len(y) * fraction))
    indices = rank_indices(probability, transaction_id)[:k]
    captured = int(np.asarray(y)[indices].sum())
    positives = int(np.asarray(y).sum())
    if positives == 0:
        raise ValueError("No fraud cases; recall is undefined")
    return {"k": k, "actual_fraction": k / len(y), "captured": captured,
            "recall": captured / positives, "precision": captured / k}


def metrics(y, probability, transaction_id):
    y, probability, transaction_id = np.asarray(y), probability_check(probability), np.asarray(transaction_id)
    if not (len(y) == len(probability) == len(transaction_id)):
        raise ValueError("Evaluation lengths differ")
    if len(np.unique(transaction_id)) != len(y):
        raise ValueError("Duplicate evaluation IDs")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Evaluation requires both classes")
    one = top_budget(y, probability, transaction_id, 0.01)
    five = top_budget(y, probability, transaction_id, 0.05)
    return {"roc_auc": float(roc_auc_score(y, probability)),
            "average_precision": float(average_precision_score(y, probability)),
            "recall_top_1pct": one["recall"], "precision_top_1pct": one["precision"],
            "recall_top_5pct": five["recall"], "brier_score": float(brier_score_loss(y, probability)),
            "n_test": len(y), "n_fraud": int(y.sum()), "fraud_prevalence": float(y.mean()),
            "top_1pct_count": one["k"], "top_5pct_count": five["k"],
            "top_1pct_actual_fraction": one["actual_fraction"],
            "top_5pct_actual_fraction": five["actual_fraction"]}


def plots(y, probabilities, ids, table, destination):
    destination.mkdir(parents=True, exist_ok=True)
    colors = {"LightGBM": "#2563eb", "FT-Transformer": "#e77920"}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    for name, p in probabilities.items():
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, label=f"{name}: AP={average_precision_score(y, p):.4f}", color=colors[name])
    ax.axhline(np.mean(y), color="#64748b", linestyle="--", label=f"Prevalence {np.mean(y):.3%}")
    ax.set(xlabel="Fraud recall", ylabel="Precision", title="IEEE-CIS chronological held-out evaluation",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.savefig(destination / "precision_recall.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    for name, p in probabilities.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name}: AUC={roc_auc_score(y, p):.4f}", color=colors[name])
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.savefig(destination / "roc.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    budgets = np.linspace(0.001, 0.20, 200)
    for name, p in probabilities.items():
        captured = np.cumsum(np.asarray(y)[rank_indices(p, ids)]) / np.sum(y)
        positions = np.ceil(len(y) * budgets).astype(int) - 1
        ax.plot(budgets * 100, captured[positions], color=colors[name], label=name)
    for budget in [1, 5]:
        ax.axvline(budget, color="#64748b", linestyle="--", alpha=0.5)
    ax.set(xlabel="Transactions investigated (%)", ylabel="Fraction of fraud cases captured",
           title="Recall versus investigation budget", xlim=(0, 20), ylim=(0, 1))
    ax.legend()
    fig.savefig(destination / "recall_budget.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    for ax, key, title in zip(axes, ["average_precision", "recall_top_1pct", "precision_top_1pct"],
                              ["Average precision", "Fraud recall @ top 1%", "Precision @ top 1%"]):
        bars = ax.bar(table["model"], table[key], color=[colors[n] for n in table["model"]])
        ax.set(title=title, ylim=(0, 1))
        ax.tick_params(axis="x", labelsize=10)
        ax.bar_label(bars, fmt="%.4f", padding=3)
    fig.suptitle("Same chronological test transactions; seed 42; uncalibrated weighted probabilities")
    fig.savefig(destination / "model_comparison.png", dpi=180)
    plt.close(fig)


def write_summary(table, manifest, mode, total_seconds):
    a, b = table.iloc[0], table.iloc[1]
    ap_winner = ("FT-Transformer" if b.average_precision > a.average_precision else
                 "LightGBM" if a.average_precision > b.average_precision else "Neither (tie)")
    recall_winner = ("FT-Transformer" if b.recall_top_1pct > a.recall_top_1pct else
                     "LightGBM" if a.recall_top_1pct > b.recall_top_1pct else "Neither (tie)")
    columns = ["model", *SCORE_COLUMNS, "training_seconds", "peak_gpu_allocated_mib"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[c]) if c == "model" else
                     ("N/A" if pd.isna(row[c]) else f"{row[c]:.6f}") for c in columns) + " |")
    text = f"""# Experiment summary

Status: **COMPLETED ({mode})**. Generated from actual predictions, {utc_now()}.
{'This is a first-30,000-row smoke run, not the full benchmark.' if mode == 'smoke' else 'Full labeled training CSV; official competition test.csv was not used.'}

## How inputs produce outputs

The labeled transaction CSV is stably ordered by TransactionDT, then divided
70/15/15 by row count. Fifty-two domain-selected raw candidates are filtered only
for training-set constant/all-missing fields. Training medians, vocabularies,
numeric missing indicators and scaling are fitted once. The models see identical
raw fields and partitions. LightGBM builds trees; the official RTDL FT-Transformer
embeds columns and attends across them on CUDA. Validation average precision
chooses the boosting iteration / neural checkpoint. Only after both selections
are complete are test probabilities generated and test labels opened.

## Actual measurements

{chr(10).join(lines)}

Test rows: {int(a.n_test)}; fraud cases: {int(a.n_fraud)}; prevalence: {a.fraud_prevalence:.4%}.
Top 1% investigations: {int(a.top_1pct_count)}; top 5%: {int(a.top_5pct_count)}.
Selection uses ceil(N * budget), descending probability, then TransactionID to
break exact score ties. It does not use labels or optimize a test threshold.
Combined model training time: {a.training_seconds + b.training_seconds:.1f}s.
Complete runner wall time and failures are recorded in `reports/run_status.json`.
Data SHA256: `{manifest['source_sha256']}`.
Boundary equal-time ties: {manifest['boundary_time_ties']}.

## Research interpretation

1. **Does FT-Transformer outperform LightGBM?** On this fixed split/configuration,
   {ap_winner} has higher test AP. FT minus LightGBM AP is
   {b.average_precision - a.average_precision:+.6f}. This is a single seed, not an
   architecture-wide conclusion or an uncertainty-qualified superiority claim.
2. **Which captures more fraud at 1% investigation budget?** {recall_winner};
   LightGBM recall={a.recall_top_1pct:.4%}, FT-Transformer recall={b.recall_top_1pct:.4%}.
   If tied, the recorded recalls should be interpreted as a tie.
3. **Is ROC-AUC sufficient?** No. AP measures precision/recall tradeoffs and the
   top-budget metrics connect ranking to finite review capacity. ROC-AUC can look
   strong while precision at operational review volumes remains weak. AP here is
   sklearn average_precision_score, not trapezoidal area under the PR curve.
4. **Limitations:** one chronological split and one seed; anonymized fields with
   unverified point-in-time provenance; recurring entities across partitions;
   possible equal-time boundary overlap; no label-delay embargo; limited field
   subset and 10-epoch neural budget; no calibration; no monetary loss, investigator
   outcome, fairness, production latency, or drift validation. Positive weighting
   shifts both models' raw probability calibration, so Brier scores are descriptive
   rather than a deployment probability guarantee. The test has now been viewed;
   later tuning would need a new untouched time window.
5. **Most useful next experiment:** a rolling temporal backtest with realistic
   delayed-label availability, retaining the fixed 1% investigation-budget metric.
   Fit any probability calibration on a separate past calibration slice.

Better offline classification does not establish reduced financial losses.
This benchmark is not production ready. Raw data, checkpoints and row predictions
are local/gitignored. Hashes and aggregate results are suitable portfolio evidence.
"""
    path = ROOT / "reports" / ("experiment_summary.md" if mode == "full" else "smoke_summary.md")
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")


def evaluate(mode="full", total_seconds=float("nan")):
    import lightgbm as lgb
    import torch
    from .data import lightgbm_frame
    from .train_ft_transformer import build_model, gpu_features, predict, require_cuda

    prepared, results, models = mode_paths(mode)
    manifest = read_json(prepared / "manifest.json")
    state = read_json(prepared / "preprocessor.json")
    names = [("LightGBM", "lightgbm"), ("FT-Transformer", "ft_transformer")]
    records = {}
    expected_files = {"lightgbm": "lightgbm.txt", "ft_transformer": "ft_transformer.pt"}
    for name, key in names:
        record = read_json(results / f"{key}_training.json")
        if record["status"] != "COMPLETED":
            raise ValueError(f"{name} selection not complete; test evaluation is blocked")
        if record["prepared_manifest_sha256"] != sha256(prepared / "manifest.json"):
            raise ValueError("Training/prepared manifest mismatch")
        if record["model_sha256"] != sha256(models / expected_files[key]):
            raise ValueError(f"{name} model fingerprint mismatch")
        records[key] = record
    for filename, expected in manifest["artifact_sha256"].items():
        if sha256(prepared / filename) != expected:
            raise ValueError(f"Prepared data fingerprint mismatch: {filename}")
    if sha256(prepared / "preprocessor.json") != manifest["preprocessor_sha256"]:
        raise ValueError("Preprocessing fingerprint mismatch")
    arrays = load_partition(prepared, "test_features")
    started = time.perf_counter()
    booster = lgb.Booster(model_file=str(models / "lightgbm.txt"))
    probability_a = probability_check(booster.predict(lightgbm_frame(arrays, state)))
    lgb_inference = time.perf_counter() - started
    device = require_cuda(42)
    checkpoint = torch.load(models / "ft_transformer.pt", map_location=device, weights_only=True)
    if checkpoint["preprocessor_sha256"] != manifest["preprocessor_sha256"]:
        raise ValueError("FT checkpoint preprocessing mismatch")
    model = build_model(state, checkpoint["settings"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    x_num, x_cat = gpu_features(arrays, device)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.cuda.synchronize()
    started = time.perf_counter()
    probability_b = predict(model, x_num, x_cat, checkpoint["settings"]["batch_size"],
                            amp_dtype, checkpoint["settings"]["mixed_precision"])
    torch.cuda.synchronize()
    ft_inference = time.perf_counter() - started
    write_json(results / "test_access.json", {"opened_utc": utc_now(), "purpose": "final paired evaluation",
                                              "model_sha256": {k: r["model_sha256"] for k, r in records.items()}})
    labels = load_partition(prepared, "test_labels")
    if not np.array_equal(arrays["transaction_id"], labels["transaction_id"]):
        raise ValueError("Test ID alignment mismatch")
    y, ids = labels["y"], labels["transaction_id"]
    probabilities = {"LightGBM": probability_a, "FT-Transformer": probability_b}
    rows = []
    for (name, key), p, inference_seconds in zip(names, probabilities.values(), [lgb_inference, ft_inference]):
        pd.DataFrame({"TransactionID": ids, "isFraud": y, "probability": p}).to_csv(
            results / f"predictions_{key}_test.csv", index=False, float_format="%.17g")
        row = {"model": name, "status": "COMPLETED", **metrics(y, p, ids),
               "training_seconds": records[key]["training_seconds"], "inference_seconds": inference_seconds,
               "peak_gpu_allocated_mib": records[key]["peak_gpu_allocated_mib"],
               "peak_gpu_reserved_mib": records[key]["peak_gpu_reserved_mib"]}
        rows.append(row)
    table = pd.DataFrame(rows)
    destination = ROOT / "results" if mode == "full" else results
    table.to_csv(destination / "metrics.csv", index=False, float_format="%.17g")
    plots(y, probabilities, ids, table, destination)
    write_summary(table, manifest, mode, total_seconds)
    verify_saved_metrics(mode)
    print(table[["model", *SCORE_COLUMNS]].to_string(index=False), flush=True)
    return table


def verify_saved_metrics(mode="full"):
    """Recompute from disk; this is a self-check, not independent validation."""
    _, results, _ = mode_paths(mode)
    destination = ROOT / "results" if mode == "full" else results
    recorded = pd.read_csv(destination / "metrics.csv", float_precision="round_trip")
    reference = None
    recomputed = {}
    for name, key in [("LightGBM", "lightgbm"), ("FT-Transformer", "ft_transformer")]:
        predictions = pd.read_csv(results / f"predictions_{key}_test.csv", float_precision="round_trip")
        ids = predictions["TransactionID"].to_numpy()
        y = predictions["isFraud"].to_numpy()
        if reference is not None:
            if not np.array_equal(reference[0], ids) or not np.array_equal(reference[1], y):
                raise ValueError("Models do not use identical ordered evaluation samples")
        reference = (ids, y)
        actual = metrics(y, predictions["probability"].to_numpy(), ids)
        original = recorded[recorded.model == name].iloc[0]
        for column in SCORE_COLUMNS:
            if not np.isclose(actual[column], original[column], atol=1e-12, rtol=0):
                raise ValueError(f"Metric recomputation mismatch: {name} {column}")
        recomputed[name] = actual
    write_json(results / "metric_recomputation.json", {"status": "PASS", "kind": "self-recomputation",
                                                       "checked_utc": utc_now(), "metrics": recomputed})
    return recomputed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    verify_saved_metrics(args.mode) if args.verify_only else evaluate(args.mode)
