"""Train once; validation AP selects boosting iteration. Never reads test labels."""
from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score

from .common import (ROOT, load_partition, mode_paths, positive_weight, probability_check,
                     read_json, seed_everything, sha256, source_fingerprints, utc_now, write_json)
from .data import lightgbm_frame


def train(mode="full"):
    config = read_json(ROOT / "configs" / "mvp.json")
    seed_everything(config["seed"])
    prepared, results, models = mode_paths(mode)
    model_file = models / "lightgbm.txt"
    if model_file.exists():
        raise FileExistsError(f"Refusing to overwrite existing model: {model_file}")
    manifest = read_json(prepared / "manifest.json")
    state = read_json(prepared / "preprocessor.json")
    training = load_partition(prepared, "train")
    validation = load_partition(prepared, "validation")
    x_train = lightgbm_frame(training, state)
    x_val = lightgbm_frame(validation, state)
    settings = dict(config["lightgbm"])
    patience = settings.pop("early_stopping_rounds")
    limit = settings.pop("max_training_seconds")
    if mode == "smoke":
        settings["n_estimators"] = 80
        patience = 15
    weight = positive_weight(training["y"])
    model = lgb.LGBMClassifier(**settings, objective="binary", metric="None",
                               random_state=config["seed"], scale_pos_weight=weight,
                               deterministic=True, force_col_wise=True, verbosity=-1)
    started = time.perf_counter()

    def ap_metric(y_true, probability):
        return "average_precision", float(average_precision_score(y_true, probability)), True

    def time_guard(_):
        if time.perf_counter() - started > limit:
            raise TimeoutError(f"LightGBM exceeded {limit}s training budget; full experiment stopped")

    history = {}
    model.fit(x_train, training["y"], eval_set=[(x_val, validation["y"])],
              eval_metric=ap_metric, categorical_feature=state["categorical"],
              callbacks=[time_guard, lgb.early_stopping(patience, first_metric_only=True, verbose=True),
                         lgb.record_evaluation(history), lgb.log_evaluation(50)])
    elapsed = time.perf_counter() - started
    prediction = probability_check(model.predict_proba(x_val)[:, 1])
    model.booster_.save_model(str(model_file))
    pd.DataFrame({"TransactionID": validation["transaction_id"],
                  "isFraud": validation["y"], "probability": prediction}).to_csv(
                      results / "predictions_lightgbm_validation.csv", index=False)
    pd.DataFrame({"feature": model.booster_.feature_name(),
                  "gain": model.booster_.feature_importance(importance_type="gain"),
                  "split_count": model.booster_.feature_importance(importance_type="split")}).sort_values(
                      "gain", ascending=False).to_csv(results / "lightgbm_feature_importance.csv", index=False)
    pd.DataFrame({"iteration": range(1, len(history["valid_0"]["average_precision"]) + 1),
                  "validation_ap": history["valid_0"]["average_precision"]}).to_csv(
                      results / "lightgbm_training_curve.csv", index=False)
    record = {"status": "COMPLETED", "model": "LightGBM", "mode": mode,
              "finished_utc": utc_now(), "settings": settings, "seed": config["seed"],
              "scale_pos_weight": weight, "calibration": "weighted raw probabilities; not recalibrated",
              "best_iteration": int(model.best_iteration_),
              "best_validation_ap": float(average_precision_score(validation["y"], prediction)),
              "training_seconds": elapsed, "device": "CPU", "peak_gpu_allocated_mib": None,
              "peak_gpu_reserved_mib": None, "model_sha256": sha256(model_file),
              "prepared_manifest_sha256": sha256(prepared / "manifest.json"),
              "source_sha256": source_fingerprints(), "data_sha256": manifest["source_sha256"]}
    write_json(results / "lightgbm_training.json", record)
    write_json(models / "lightgbm_config.json", record)
    print(f"LightGBM {mode}: best validation AP={record['best_validation_ap']:.6f}; {elapsed:.1f}s", flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    train(parser.parse_args().mode)
