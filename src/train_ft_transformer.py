"""Official RTDL FT-Transformer; mandatory CUDA, AMP, validation AP selection."""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from rtdl_revisiting_models import FTTransformer
from sklearn.metrics import average_precision_score

from .common import (ROOT, load_partition, mode_paths, positive_weight, probability_check,
                     read_json, seed_everything, sha256, source_fingerprints, utc_now, write_json)


def require_cuda(seed=42):
    if not torch.cuda.is_available():
        raise RuntimeError("FT-Transformer requires CUDA; CPU fallback is forbidden")
    # Memory-stat APIs with an explicit device need CUDA's lazy initialization
    # completed before any GPU tensor has been created.
    torch.cuda.init()
    seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device("cuda:0")


def build_model(state, settings):
    return FTTransformer(n_cont_features=len(state["numeric_output_names"]),
                         cat_cardinalities=state["cat_cardinalities"], d_out=1,
                         n_blocks=settings["n_blocks"], d_block=settings["d_block"],
                         attention_n_heads=settings["attention_n_heads"],
                         attention_dropout=settings["attention_dropout"], ffn_d_hidden=None,
                         ffn_d_hidden_multiplier=settings["ffn_d_hidden_multiplier"],
                         ffn_dropout=settings["ffn_dropout"], residual_dropout=settings["residual_dropout"])


def gpu_features(arrays, device):
    return (torch.as_tensor(arrays["x_num"], device=device, dtype=torch.float32),
            torch.as_tensor(arrays["x_cat"], device=device, dtype=torch.int64))


@torch.inference_mode()
def predict(model, x_num, x_cat, batch_size, amp_dtype, amp_enabled):
    model.eval()
    outputs = []
    for start in range(0, len(x_num), batch_size):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            logits = model(x_num[start:start + batch_size], x_cat[start:start + batch_size]).squeeze(-1)
        outputs.append(logits.float().sigmoid().cpu().numpy())
    return probability_check(np.concatenate(outputs))


def train(mode="full"):
    config = read_json(ROOT / "configs" / "mvp.json")
    settings = dict(config["ft_transformer"])
    device = require_cuda(config["seed"])
    torch.set_num_threads(4)
    prepared, results, models = mode_paths(mode)
    model_file = models / "ft_transformer.pt"
    if model_file.exists():
        raise FileExistsError(f"Refusing to overwrite existing model: {model_file}")
    state = read_json(prepared / "preprocessor.json")
    manifest = read_json(prepared / "manifest.json")
    training = load_partition(prepared, "train")
    validation = load_partition(prepared, "validation")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    x_num, x_cat = gpu_features(training, device)
    v_num, v_cat = gpu_features(validation, device)
    y = torch.as_tensor(training["y"], device=device, dtype=torch.float32)
    model = build_model(state, settings).to(device)
    assert next(model.parameters()).is_cuda and x_num.is_cuda
    optimizer = torch.optim.AdamW(model.make_parameter_groups(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    weight = positive_weight(training["y"])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    amp_enabled = settings["mixed_precision"]
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    epochs = 1 if mode == "smoke" else settings["max_epochs"]
    batch_size = settings["batch_size"]
    best_ap, best_epoch, bad_epochs = -1.0, 0, 0
    history = []
    started = time.perf_counter()
    try:
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            order = torch.randperm(len(y), device=device)
            loss_sum = torch.zeros((), device=device)
            for batch_start in range(0, len(y), batch_size):
                if time.perf_counter() - started > settings["max_training_seconds"]:
                    raise TimeoutError("FT-Transformer exceeded fixed training time budget")
                indices = order[batch_start:batch_start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                    logits = model(x_num[indices], x_cat[indices]).squeeze(-1)
                    loss = criterion(logits.float(), y[indices])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at epoch {epoch}, batch {batch_start}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.detach() * len(indices)
            validation_probability = predict(model, v_num, v_cat, batch_size, amp_dtype, amp_enabled)
            validation_ap = float(average_precision_score(validation["y"], validation_probability))
            torch.cuda.synchronize()
            epoch_seconds = time.perf_counter() - epoch_started
            row = {"epoch": epoch, "training_loss": float(loss_sum.cpu()) / len(y),
                   "validation_ap": validation_ap, "epoch_seconds": epoch_seconds,
                   "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                   "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20}
            history.append(row)
            pd.DataFrame(history).to_csv(results / "ft_training_curve.csv", index=False)
            print(f"FT {mode} epoch {epoch}/{epochs}: AP={validation_ap:.6f}, loss={row['training_loss']:.5f}, "
                  f"{epoch_seconds:.1f}s, GPU peak={row['peak_gpu_allocated_mib']:.0f} MiB", flush=True)
            if validation_ap > best_ap + 1e-8:
                best_ap, best_epoch, bad_epochs = validation_ap, epoch, 0
                temporary = model_file.with_suffix(".pt.tmp")
                torch.save({"model_state": model.state_dict(), "settings": settings,
                            "best_epoch": epoch, "best_validation_ap": best_ap,
                            "preprocessor_sha256": manifest["preprocessor_sha256"],
                            "seed": config["seed"]}, temporary)
                temporary.replace(model_file)
            else:
                bad_epochs += 1
            # Cost rule fixed before observing any held-out test outcome.
            if mode == "full" and epoch == 1 and epoch_seconds * epochs > settings["max_estimated_seconds"]:
                raise TimeoutError(f"First epoch projects {epoch_seconds * epochs:.1f}s for {epochs} epochs, "
                                   f"above fixed {settings['max_estimated_seconds']}s cost limit")
            if bad_epochs >= settings["patience"]:
                break
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("CUDA OOM; run stopped without CPU fallback. Reduce batch size explicitly and record it.") from exc
    elapsed = time.perf_counter() - started
    checkpoint = torch.load(model_file, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    probability = predict(model, v_num, v_cat, batch_size, amp_dtype, amp_enabled)
    pd.DataFrame({"TransactionID": validation["transaction_id"], "isFraud": validation["y"],
                  "probability": probability}).to_csv(results / "predictions_ft_transformer_validation.csv", index=False)
    record = {"status": "COMPLETED", "model": "FT-Transformer", "mode": mode, "finished_utc": utc_now(),
              "settings": settings, "seed": config["seed"], "pos_weight": weight,
              "calibration": "weighted raw probabilities; not recalibrated", "best_epoch": best_epoch,
              "best_validation_ap": best_ap, "epochs_executed": len(history), "training_seconds": elapsed,
              "device": str(device), "gpu": torch.cuda.get_device_name(device),
              "cuda_build": torch.version.cuda, "amp_dtype": str(amp_dtype), "amp_enabled": amp_enabled,
              "deterministic_algorithms": True,
              "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
              "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "model_sha256": sha256(model_file), "prepared_manifest_sha256": sha256(prepared / "manifest.json"),
              "source_sha256": source_fingerprints(), "data_sha256": manifest["source_sha256"]}
    write_json(results / "ft_transformer_training.json", record)
    write_json(models / "ft_transformer_config.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    train(parser.parse_args().mode)
