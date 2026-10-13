"""Reproduce the complete figure from published aggregate records; no ML execution."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', default='rendered_training_results')
parser.add_argument('--font', default=None, help='Optional font file; defaults to bundled DejaVu Sans')
args = parser.parse_args()
OUT = Path(args.output).resolve()
PROJECT = HERE.parents[1]
if OUT == HERE:
    raise ValueError('Choose a separate output folder to preserve the published artifacts')
EXTENSION = PROJECT / "experiments/excel_xgb_convergence_20261012"
FIRST = PROJECT / "experiments/excel_xgb_distillation_20261012"
TRANSFER = PROJECT / "experiments/identity_uid_transformer_transfer_20261011"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mpl-cache"))

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2166ac"
ORANGE = "#d06b25"
BASELINE = "#333333"
REFERENCE = "#7a8188"
PARENT = "#8a9299"
TEXT = "#252c32"
TRIALS = ("bce_control", "xgb_kd_lambda1")
NAMES = {"bce_control": "BCE ExcelFormer", "xgb_kd_lambda1": "Distilled ExcelFormer"}
COLORS = {"bce_control": BLUE, "xgb_kd_lambda1": ORANGE}
STYLES = {"bce_control": "-", "xgb_kd_lambda1": "--"}
MARKERS = {"bce_control": "o", "xgb_kd_lambda1": "s"}
SOURCES = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_source(path):
    SOURCES[str(path.relative_to(PROJECT)).replace("\\", "/")] = sha(path)


def read(path):
    record_source(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def history(path):
    record_source(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [{k: (v if k.endswith("sha256") or k == "stage" else float(v))
             for k, v in row.items()} for row in rows]


def setup():
    family = 'DejaVu Sans'
    if args.font:
        font = Path(args.font)
        font_manager.fontManager.addfont(font)
        family = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.family": family, "font.size": 12,
        "axes.titlesize": 14, "axes.labelsize": 12, "xtick.labelsize": 11,
        "ytick.labelsize": 11, "legend.fontsize": 11, "axes.unicode_minus": False,
        "axes.edgecolor": "#b9bfc4", "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT, "svg.fonttype": "none"})


def clean(ax):
    ax.grid(axis="y", color="#e5e8eb", linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def phases(ax, *, full=False):
    if full:
        ax.axvspan(.6, 16.5, color="#eef0f2", alpha=.6, zorder=0)
        ax.axvline(16.5, color="#9aa1a8", lw=.8, linestyle=":")
        ax.axvline(21.5, color="#9aa1a8", lw=.8, linestyle=":")
        labels = [(8.5, "Shared parent (16 epochs)"), (19, "First 5"), (31.5, "Extension (20 epochs)")]
    else:
        ax.axvspan(.5, 5.5, color="#eef0f2", alpha=.65, zorder=0)
        ax.axvline(5.5, color="#9aa1a8", lw=.8, linestyle=":")
        labels = [(3, "First 5 epochs"), (15.5, "Added 20 epochs")]
    for x, label in labels:
        ax.text(x, 1.02, label, transform=ax.get_xaxis_transform(), fontsize=10,
                va="bottom", ha="center", color="#5d666f")


def main():
    setup()
    snapshot = read(HERE / "figure_data.json")
    decision = snapshot["decision"]
    verification = read(PROJECT / "evidence/final_training/verification.json")
    replay = read(PROJECT / "evidence/final_training/fresh_verification.json")
    assert verification["status"] == replay["status"] == "PASS"
    assert decision["stopping_reason"] == "ADDITIONAL_EPOCH_CAP"
    summary = read(PROJECT / "evidence/results_summary.json")
    assert summary["saved_prediction_rescore"] == "PASS_CPU_NUMPY_WITH_IDENTICAL_IDS_AND_LABELS"
    lgb_ap = snapshot["original_lightgbm"]["ap"]
    lgb_capture = snapshot["original_lightgbm"]["captured_top_1pct"]
    reference_ap = snapshot["same_source_fields_reference"]["ap"]
    reference_capture = snapshot["same_source_fields_reference"]["captured_top_1pct"]
    matching = read(PROJECT / "evidence/enriched_lightgbm_reference.json")
    assert matching["metrics"]["n_rows"] == summary["validation_transactions"] == 88581
    assert abs(reference_ap - matching["metrics"]["average_precision"]) < 1e-12
    assert reference_capture == matching["metrics"]["captured_top_1pct"]
    ablations = {name: read(PROJECT / f"evidence/feature_ablation/{name}.json")
                 for name in ("B00", "B10", "B1C")}
    parent = snapshot["parent_epochs"]
    assert [int(r["epoch"]) for r in parent] == list(range(1, 17))
    parent_ap = ablations["B1C"]["metrics"]["average_precision"]
    assert abs(parent[-1]["validation_ap"] - parent_ap) < 1e-12
    histories = {}
    for trial in TRIALS:
        rows = snapshot["paired_continuation"][trial]
        assert [int(r["epoch"]) for r in rows] == list(range(1, 26))
        assert [int(r["parent_epoch_offset"]) for r in rows] == list(range(17, 42))
        peak = max(rows, key=lambda r: r["average_precision"])
        assert int(peak["epoch"]) == decision["selected_epoch"] == 22
        final = decision["paired_metrics"][trial]
        for metric in ("average_precision", "captured_top_1pct", "brier_score"):
            assert abs(peak[metric] - final[metric]) < 1e-12
        histories[trial] = rows
    assert len(histories[TRIALS[0]]) == len(histories[TRIALS[1]]) == 25
    completed_local = datetime.fromisoformat(decision["finished_utc"]).astimezone(ZoneInfo("America/New_York"))
    seconds = round(decision["additional_active_gpu_seconds"])
    elapsed = f"{seconds // 3600}h {seconds // 60 % 60:02d}m {seconds % 60:02d}s"

    fig = plt.figure(figsize=(14.5, 15.0), facecolor="white")
    grid = fig.add_gridspec(3, 2, height_ratios=[1.35, .87, .92],
                           left=.08, right=.965, top=.84, bottom=.15,
                           hspace=.48, wspace=.24)
    ax_ap = fig.add_subplot(grid[0, :])
    ax_loss = fig.add_subplot(grid[1, 0])
    ax_capture = fig.add_subplot(grid[1, 1])
    ax_stages = fig.add_subplot(grid[2, :])
    fig.text(.08, .970, "Fraud detection: training results and baseline comparison", fontsize=21, weight="bold")
    fig.text(.08, .943, "Same development-validation set: 88,581 transactions | 3,042 labeled fraud cases | Raw values, no smoothing", fontsize=12)
    fig.text(.08, .917, f"Completed and verified | {completed_local:%Y-%m-%d %H:%M} ET | Added GPU time: {elapsed} / 5h budget | Stopped at the 20-epoch extension cap", fontsize=11)

    for ax in (ax_ap, ax_loss, ax_capture, ax_stages):
        clean(ax)
    # Full trajectory: common parent 1..16, then each continuation 17..41.
    ax_ap.plot([r["epoch"] for r in parent], [r["validation_ap"] for r in parent],
               color=PARENT, lw=1.7, marker=".", ms=5, label="Shared parent: enriched inputs (B1C)")
    for trial in TRIALS:
        rows = histories[trial]
        xs = [16] + [r["parent_epoch_offset"] for r in rows]
        ys = [parent_ap] + [r["average_precision"] for r in rows]
        ax_ap.plot(xs, ys, color=COLORS[trial], lw=2, linestyle=STYLES[trial],
                   marker=MARKERS[trial], ms=3.8,
                   label=f"{NAMES[trial]} | best AP {decision['paired_metrics'][trial]['average_precision']:.4f}")
    ax_ap.axhline(lgb_ap, color=BASELINE, lw=1.55, linestyle=(0, (6, 3)),
                 label=f"Initial LightGBM: {lgb_ap:.4f} (seed42)")
    ax_ap.axhline(reference_ap, color=REFERENCE, lw=1.35, linestyle=(0, (2, 3)),
                 label=f"Enriched-field LightGBM: {reference_ap:.4f} (seed43)")
    ax_ap.scatter([38], [decision["selected_ap"]], marker="*", s=170,
                  color=BLUE, edgecolors="white", linewidths=.7, zorder=6)
    ax_ap.annotate("Selected checkpoint:\ncontinuation 22 / total 38", xy=(38, decision["selected_ap"]),
                   xytext=(25, .656), color=BLUE, fontsize=11,
                   arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": .9})
    ax_ap.set_xlim(.5, 42)
    ax_ap.set_ylim(.46, .708)
    ax_ap.set_xticks([1, 8, 16, 21, 28, 35, 41])
    ax_ap.set_ylabel("Validation average precision (AP)")
    ax_ap.set_xlabel("Total training epochs (16 shared + 25 continuation = 41 per arm)")
    ax_ap.set_title("A  Risk-ranking quality over the full training run", loc="left", pad=34)
    ax_ap.legend(loc="lower right", frameon=False, ncol=2, fontsize=10,
                  columnspacing=1.2, labelspacing=.5)
    phases(ax_ap, full=True)

    # Common supervised component only; KD total includes a separate soft term.
    for trial in TRIALS:
        rows = histories[trial]
        ax_loss.plot([r["epoch"] for r in rows], [r["hard_loss"] for r in rows],
                     color=COLORS[trial], lw=1.9, linestyle=STYLES[trial],
                     marker=MARKERS[trial], ms=3.5, label=NAMES[trial])
    ax_loss.set_xlim(.5, 25.7)
    losses = [r["hard_loss"] for rows in histories.values() for r in rows]
    ax_loss.set_ylim(min(losses) - .0018, max(losses) + .0018)
    ax_loss.set_xticks([1, 5, 10, 15, 20, 25])
    ax_loss.set_title("B  Training loss (shared BCE)", loc="left", pad=30)
    ax_loss.set_ylabel("Supervised loss (lower is better)")
    ax_loss.set_xlabel("Continuation epoch (first 5 + added 20)")
    ax_loss.legend(frameon=False, fontsize=10.5, loc="upper right")
    phases(ax_loss)

    # Capture follows AP-selected checkpoints; it is not the selection criterion.
    for trial in TRIALS:
        rows = histories[trial]
        ax_capture.plot([0] + [r["epoch"] for r in rows],
                        [ablations["B1C"]["metrics"]["captured_top_1pct"]] +
                        [r["captured_top_1pct"] for r in rows],
                        color=COLORS[trial], lw=1.8, linestyle=STYLES[trial],
                        marker=MARKERS[trial], ms=3.4)
        selected = decision["paired_metrics"][trial]["captured_top_1pct"]
        ax_capture.scatter([22], [selected], color=COLORS[trial], marker="D", s=50, zorder=6)
    ax_capture.axhline(lgb_capture, color=BASELINE, linestyle=(0, (6, 3)), lw=1.4)
    ax_capture.axhline(reference_capture, color=REFERENCE, linestyle=(0, (2, 3)), lw=1.1)
    ax_capture.text(.5, lgb_capture + 1.6, f"Initial LightGBM: {lgb_capture} cases", fontsize=10.5)
    ax_capture.text(.5, reference_capture - 4, f"Enriched-field tree: {reference_capture} cases", color="#666e75", fontsize=10)
    ax_capture.annotate("Selected BCE: 830 cases (+41)", xy=(22, 830), xytext=(6, 800),
                        fontsize=10.5, color=BLUE,
                        arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": .8})
    ax_capture.set_title("C  Fraud found within 886 reviews", loc="left", pad=30)
    ax_capture.set_xlim(-.5, 25.7)
    ax_capture.set_ylim(lgb_capture - 6, reference_capture + 6)
    ax_capture.set_xticks([0, 5, 10, 15, 20, 25])
    ax_capture.set_ylabel("Labeled fraud cases found")
    ax_capture.set_xlabel("Continuation epoch (0 = shared parent)")
    phases(ax_capture)

    # Show feature gains separately from later training, with both tree references.
    stage_values = [ablations[n]["metrics"]["average_precision"] for n in ("B00", "B10", "B1C")] + [
        decision["paired_metrics"][t]["average_precision"] for t in TRIALS]
    stage_labels = ["Basic ExcelFormer\nB00", "+ Device / identity\nB10", "+ Entity statistics\nB1C", "Enriched inputs +\nBCE continuation\nAP-selected checkpoint", "Enriched inputs +\ndistilled continuation\nAP-selected checkpoint"]
    ax_stages.plot([0, 1, 2], stage_values[:3], color=PARENT, marker="o", lw=1.7)
    ax_stages.plot([2, 3], [stage_values[2], stage_values[3]], color=BLUE, lw=1.7, linestyle="--")
    ax_stages.plot([2, 4], [stage_values[2], stage_values[4]], color=ORANGE, lw=1.4, linestyle="--")
    for index, value in enumerate(stage_values):
        color = BLUE if index == 3 else (ORANGE if index == 4 else PARENT)
        ax_stages.scatter([index], [value], color=color, s=70, marker="D" if index >= 3 else "o", zorder=5)
        offset = -.0105 if abs(value - lgb_ap) < .008 else .006
        ax_stages.text(index, value + offset, f"{value:.4f}", ha="center", va="center", fontsize=13, color=color)
    ax_stages.axhline(lgb_ap, color=BASELINE, lw=1.5, linestyle=(0, (6, 3)))
    ax_stages.axhline(reference_ap, color=REFERENCE, lw=1.1, linestyle=(0, (2, 3)))
    ax_stages.text(3.1, lgb_ap - .002, f"Initial LightGBM {lgb_ap:.4f}",
                   color=BASELINE, fontsize=10.5, va="top")
    ax_stages.text(3.1, reference_ap + .005, f"Enriched-field tree reference {reference_ap:.4f}", color=REFERENCE, fontsize=10.5)
    ax_stages.axvline(2.5, color="#b3bac0", lw=.9, linestyle=":")
    ax_stages.text(1, .669, "Matched feature comparisons: +0.0285 AP", fontsize=11, ha="center")
    ax_stages.text(3.55, .669, "Further BCE training: +0.0347 AP", fontsize=11, ha="center", color=BLUE)
    ax_stages.set_xticks(range(5), stage_labels)
    ax_stages.set_xlim(-.5, 4.6)
    ax_stages.set_ylim(.553, .708)
    ax_stages.set_ylabel("Validation average precision (AP)")
    ax_stages.set_title("D  What improved: richer inputs, then further training", loc="left", pad=13)
    fig.text(.08, .081, "Selection uses validation AP; the best-AP checkpoint may not catch the most fraud at a fixed review budget.", fontsize=10)
    fig.text(.08, .061, "Panel B compares the shared supervised loss. Distillation adds a separate teacher term to its total loss.", fontsize=10)
    fig.text(.08, .041, "Tree lines are fixed scores: initial inputs (52 fields, seed42), enriched inputs (seed43); a pipeline comparison.", fontsize=10)
    fig.text(.08, .021, "One development split and seed; epoch cap reached, convergence unconfirmed. Frozen logs and reload checks; no new training or inference.", fontsize=9.5, color="#656e76")
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"training_complete.{extension}", dpi=170, facecolor="white")
    plt.close(fig)
    payload = {"completed_local": completed_local.isoformat(), "decision": decision,
        "original_lightgbm": {"ap": lgb_ap, "captured_top_1pct": lgb_capture, "seed": 42},
        "same_source_fields_reference": {"ap": reference_ap, "captured_top_1pct": reference_capture, "seed": 43},
        "parent_epochs": parent, "paired_continuation": histories, "stage_ap": stage_values,
        "source_sha256": SOURCES}
    (OUT / "figure_data.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    assert all(sha(PROJECT / path) == expected for path, expected in SOURCES.items())
    receipt = {"status": "PASS_LOG_AND_SELECTED_CHECKPOINT_CONSISTENCY",
               "parent_epochs": 16, "continuation_epochs_each": 25,
               "additional_epochs_each": 20, "total_lineage_epochs_each": 41,
               "selected_continuation_epoch": 22, "selected_lineage_epoch": 38,
               "sources_unchanged": True, "smoothing": False,
               "new_training_or_inference": False, "test_accessed": False,
               "visual_review": "PENDING",
               "scope": "PUBLIC_AGGREGATE_FIGURE_REPRODUCTION_NOT_MODEL_REPLICATION",
               "files": {f"training_complete.{e}": sha(OUT / f"training_complete.{e}") for e in ("png", "svg")}}
    (OUT / "visualization_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "png": str(OUT / "training_complete.png"),
                      "history_points": 66, "original_lightgbm_ap": lgb_ap,
                      "bce_best_ap": decision["selected_ap"], "additional_gpu_seconds": seconds}))


if __name__ == "__main__":
    main()
