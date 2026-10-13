"""Official FT backbone with official numeric embeddings and bounded CUDA training."""
from __future__ import annotations

import math
import os
import random
import time
import types

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, PeriodicEmbeddings, compute_bins
from rtdl_revisiting_models import FTTransformer, MultiheadAttention
from torch.nn import functional as F

from .utils import HERE, digest, fingerprints, now, read, scores, write


def cuda(seed):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback allowed")
    torch.cuda.init()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    return torch.device("cuda:0")


def sdpa_forward(self, x_q, x_kv):
    """Same projections/compression/head ordering as official attention.

    Use PyTorch's SDPA dispatcher to avoid explicitly materializing the full
    attention matrix where a fused CUDA kernel is available. No model weights or
    mathematical architecture are added. Eval equivalence is contract-tested.
    """
    q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
    if self.key_compression is not None:
        k = self.key_compression(k.transpose(1, 2)).transpose(1, 2)
        compression = self.key_compression if self.value_compression is None else self.value_compression
        v = compression(v.transpose(1, 2)).transpose(1, 2)
    b, nq, _ = q.shape
    h = self._n_heads
    q = q.reshape(b, nq, h, -1).transpose(1, 2)
    k = k.reshape(b, k.shape[1], h, -1).transpose(1, 2)
    v = v.reshape(b, v.shape[1], h, -1).transpose(1, 2)
    probability = self.dropout.p if self.dropout is not None and self.training else 0.0
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=probability)
    x = x.transpose(1, 2).reshape(b, nq, -1)
    return x if self.W_out is None else self.W_out(x)


def build(meta, cfg, bins=None):
    extra = {}
    if cfg.get("compression"):
        extra = {"linformer_kv_compression_ratio": cfg["compression"],
                 "linformer_kv_compression_sharing": "key-value"}
    model = FTTransformer(n_cont_features=len(meta["numeric_output_names"]),
                          cat_cardinalities=meta["cat_cardinalities"], d_out=1,
                          n_blocks=cfg["depth"], d_block=cfg["width"],
                          attention_n_heads=cfg["heads"], attention_dropout=cfg["dropout"],
                          ffn_d_hidden=None, ffn_d_hidden_multiplier=cfg["ffn_multiplier"],
                          ffn_dropout=cfg["dropout"], residual_dropout=cfg.get("residual_dropout", 0.0),
                          **extra)
    if cfg["embedding"] == "ple":
        if bins is None:
            raise ValueError("PLE requires saved training-only bin edges")
        model.cont_embeddings = PiecewiseLinearEmbeddings(bins, cfg["width"], activation=False, version="B")
    elif cfg["embedding"] == "periodic":
        model.cont_embeddings = PeriodicEmbeddings(len(meta["numeric_output_names"]), cfg["width"],
                                                  n_frequencies=cfg.get("frequencies", 32),
                                                  frequency_init_scale=cfg.get("frequency_scale", .05),
                                                  activation=True, lite=False)
    elif cfg["embedding"] != "linear":
        raise ValueError(cfg["embedding"])
    if cfg.get("sdpa", True):
        for module in model.modules():
            if isinstance(module, MultiheadAttention):
                module.forward = types.MethodType(sdpa_forward, module)
    return model


def training_bins(values, n_bins):
    """Official quantile bins, with a declared train-only degenerate-column rule.

    Sparse columns can be constant in a chronological smoke subset or in a
    100k-row bin-fitting sample. Use the full training column min/max then; if
    even that is constant, use [value-.5, value+.5]. No future rows or labels.
    """
    rows = np.random.RandomState(42).choice(len(values), min(100000, len(values)), replace=False)
    sample = torch.from_numpy(values[rows])
    varying = sample.amin(0) < sample.amax(0)
    generated = compute_bins(sample[:, varying], n_bins=n_bins) if varying.any() else []
    iterator, output = iter(generated), []
    for j in range(values.shape[1]):
        if varying[j]:
            output.append(next(iterator))
        else:
            low, high = float(values[:, j].min()), float(values[:, j].max())
            if low == high:
                low, high = low-.5, high+.5
            output.append(torch.tensor([low, high], dtype=torch.float32))
    return output


def loss_value(logits, y, cfg):
    logits = logits.float()
    pos_weight = torch.as_tensor(cfg["positive_weight"], device=logits.device)
    base = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
    if cfg["loss"] == "focal":
        p = logits.sigmoid()
        correct = p * y + (1-p) * (1-y)
        return (base * (1-correct).pow(cfg.get("focal_gamma", 2.0))).mean()
    if cfg["loss"] == "pairwise":
        positive, negative = logits[y > .5], logits[y < .5]
        if len(positive) and len(negative):
            # Focus ranking pressure on a bounded number of high-scoring train
            # negatives; no validation/test examples enter this objective.
            hard = negative.topk(min(128, len(negative))).values
            ranking = F.softplus(hard[None, :] - positive[:, None]).mean()
            return base.mean() + cfg.get("ranking_weight", .15) * ranking
    elif cfg["loss"] != "bce":
        raise ValueError(cfg["loss"])
    return base.mean()


@torch.inference_mode()
def predict(model, x_num, x_cat, batch_size, dtype):
    model.eval()
    output = []
    for start in range(0, len(x_num), batch_size):
        with torch.autocast("cuda", dtype=dtype):
            logit = model(x_num[start:start+batch_size], x_cat[start:start+batch_size]).squeeze(-1)
        output.append(logit.float().sigmoid().cpu().numpy())
    p = np.concatenate(output)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise RuntimeError("Invalid CUDA model probabilities")
    return p


def run_trial(trial_id, cfg, arrays, meta, stop_at_wall=None):
    path = HERE / "runs" / trial_id
    path.mkdir(parents=True, exist_ok=True)
    result_file = path / "result.json"
    if result_file.exists():
        record = read(result_file)
        if record["config"] != cfg or record["representation_sha256"] != digest(HERE / f"cache/{cfg['features']}/manifest.json"):
            raise RuntimeError("Cannot reuse a trial with different inputs")
        if digest(path / "best.pt") != record["checkpoint_sha256"]:
            raise RuntimeError("Saved checkpoint fingerprint mismatch")
        return record
    write(path / "config.json", cfg)
    device = cuda(cfg["seed"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    train, valid = arrays["train"], arrays["validation"]
    x_num = torch.as_tensor(train["x_num"], device=device)
    x_cat = torch.as_tensor(train["x_cat"], device=device)
    v_num = torch.as_tensor(valid["x_num"], device=device)
    v_cat = torch.as_tensor(valid["x_cat"], device=device)
    y = torch.as_tensor(train["y"], device=device, dtype=torch.float32)
    bins = None
    if cfg["embedding"] == "ple":
        # Covariate-only quantile edges, fixed subsample seed, never y.
        bins = training_bins(train["x_num"], cfg.get("n_bins", 32))
    model = build(meta, cfg, bins).to(device)
    assert x_num.is_cuda and next(model.parameters()).is_cuda
    optimizer = torch.optim.AdamW(model.make_parameter_groups(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5,
                                                         patience=4, min_lr=1e-6)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    start_epoch, best_ap, bad, previous_seconds = 1, -1., 0, 0.
    history = []
    last = path / "last.pt"
    if last.exists():
        restored = torch.load(last, map_location="cpu", weights_only=False)
        if restored["config"] != cfg or restored["code_fingerprints"] != fingerprints():
            raise RuntimeError("Resume code/config mismatch: preserve trial and choose a fresh ID")
        model.load_state_dict(restored["model"])
        optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["scheduler"])
        scaler.load_state_dict(restored["scaler"])
        start_epoch, best_ap, bad = restored["epoch"]+1, restored["best_ap"], restored["bad"]
        previous_seconds, history = restored["active_seconds"], restored["history"]
        torch.set_rng_state(restored["torch_rng"])
        torch.cuda.set_rng_state_all(restored["cuda_rng"])
        np.random.set_state(restored["numpy_rng"])
        random.setstate(restored["python_rng"])
        print("RESUMED", trial_id, "epoch", start_epoch, flush=True)
    positive = torch.nonzero(y > .5).flatten()
    negative = torch.nonzero(y < .5).flatten()
    # Generate only sampling indices on CPU: CUDA multinomial is not
    # deterministic on this PyTorch version. Model/arrays remain on CUDA.
    recency_weights = torch.linspace(1., cfg.get("recency_ratio", 1.), len(y), device="cpu")
    start = time.perf_counter()
    stop_reason = "maximum epochs"
    best_epoch = history[np.argmax([r["validation_ap"] for r in history])]["epoch"] if history else 0
    for epoch in range(start_epoch, cfg["max_epochs"]+1):
        if (stop_at_wall and time.time() >= stop_at_wall) or previous_seconds + time.perf_counter()-start >= cfg["max_seconds"]:
            stop_reason = "fixed time budget"
            break
        epoch_start = time.perf_counter()
        model.train()
        if cfg.get("positive_fraction"):
            count_pos = int(len(y)*cfg["positive_fraction"])
            order = torch.cat([positive[torch.randint(len(positive), (count_pos,), device=device)],
                               negative[torch.randint(len(negative), (len(y)-count_pos,), device=device)]])
            order = order[torch.randperm(len(order), device=device)]
        elif cfg.get("recency_ratio", 1.) != 1.:
            order = torch.multinomial(recency_weights, len(y), replacement=True).to(device)
        else:
            order = torch.randperm(len(y), device=device)
        total_loss = torch.zeros((), device=device)
        processed = 0
        for start_batch in range(0, len(y), cfg["batch_size"]):
            idx = order[start_batch:start_batch+cfg["batch_size"]]
            batch_num, batch_cat = x_num[idx], x_cat[idx]
            if cfg.get("feature_dropout", 0):
                batch_num = batch_num.masked_fill(torch.rand_like(batch_num) < cfg["feature_dropout"], 0)
                batch_cat = batch_cat.masked_fill(torch.rand(batch_cat.shape, device=device) < cfg["feature_dropout"], 0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                logits = model(batch_num, batch_cat).squeeze(-1)
                loss = loss_value(logits, y[idx], cfg)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach() * len(idx)
            processed += len(idx)
            # A whole validation pass is reserved, so the budget is checked each
            # batch but an interrupted epoch is not treated as a finished epoch.
            if stop_at_wall and time.time() >= stop_at_wall:
                break
        probability = predict(model, v_num, v_cat, cfg["batch_size"], dtype)
        validation_scores = scores(valid["y"], probability, valid["ids"])
        ap = validation_scores["average_precision"]
        scheduler.step(ap)
        torch.cuda.synchronize()
        row = {"epoch": epoch, "training_loss": float(total_loss.cpu())/processed,
               "validation_ap": ap, "validation_recall_1pct": validation_scores["recall_top_1pct"],
               "epoch_seconds": time.perf_counter()-epoch_start,
               "learning_rate": optimizer.param_groups[0]["lr"],
               "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
               "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved()/2**20}
        history.append(row)
        if ap > best_ap + 1e-6:
            best_ap, best_epoch, bad = ap, epoch, 0
            temporary = path / "best.pt.tmp"
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch,
                        "bins": bins, "validation_scores": validation_scores,
                        "representation_sha256": digest(HERE / f"cache/{cfg['features']}/manifest.json")}, temporary)
            temporary.replace(path / "best.pt")
            np.savez_compressed(path / "predictions_validation.npz", p=probability,
                                ids=valid["ids"], y=valid["y"])
        else:
            bad += 1
        pd.DataFrame(history).to_csv(path / "history.csv", index=False)
        active = previous_seconds + time.perf_counter()-start
        temporary = path / "last.pt.tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_ap": best_ap, "bad": bad, "history": history,
                    "active_seconds": active, "config": cfg, "code_fingerprints": fingerprints(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}, temporary)
        temporary.replace(last)
        write(path / "progress.json", {"status": "RUNNING", "updated_utc": now(),
                                       "epoch": epoch, "best_epoch": best_epoch, "best_ap": best_ap,
                                       "active_neural_seconds": active, "last_epoch": row})
        print(f"{trial_id} epoch={epoch} AP={ap:.6f} best={best_ap:.6f} "
              f"R1={row['validation_recall_1pct']:.4f} loss={row['training_loss']:.5f} "
              f"seconds={row['epoch_seconds']:.1f} MiB={row['peak_gpu_allocated_mib']:.0f}", flush=True)
        if epoch >= cfg.get("min_epochs", 20) and bad >= cfg["patience"]:
            stop_reason = "validation early stopping"
            break
    if not (path / "best.pt").exists():
        raise TimeoutError("No completed epoch before fixed budget")
    torch.cuda.synchronize()
    elapsed = previous_seconds + time.perf_counter()-start
    checkpoint = torch.load(path / "best.pt", map_location=device, weights_only=False)
    with np.load(path / "predictions_validation.npz", allow_pickle=False) as file:
        validation_scores = scores(file["y"], file["p"], file["ids"])
    record = {"status": "COMPLETED", "trial_id": trial_id, "finished_utc": now(), "config": cfg,
              "validation_scores": validation_scores, "best_epoch": best_epoch,
              "epochs_executed": len(history), "stop_reason": stop_reason,
              "active_neural_seconds": elapsed, "gpu": torch.cuda.get_device_name(), "device": "cuda:0",
              "amp_dtype": str(dtype), "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
              "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "checkpoint_sha256": digest(path / "best.pt"),
              "representation_sha256": digest(HERE / f"cache/{cfg['features']}/manifest.json"),
              "code_fingerprints": fingerprints()}
    write(result_file, record)
    return record
