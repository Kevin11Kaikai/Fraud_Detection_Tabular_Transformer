"""Infrastructure probe only. Dummy tensors are NEVER benchmark measurements."""
from __future__ import annotations

import time

import numpy as np
import torch

from .common import ROOT, environment, read_json, sha256, utc_now, write_json
from .data import CATEGORICAL, NUMERIC
from .train_ft_transformer import build_model, require_cuda


def check():
    record = environment()
    record.update(checked_utc=utc_now(), kind="INFRASTRUCTURE_PROBE_ONLY",
                  real_dataset_smoke_test=False, benchmark_metrics_computed=False)
    started = time.perf_counter()
    device = require_cuda(42)
    settings = read_json(ROOT / "configs" / "mvp.json")["ft_transformer"]
    # Worst-case numeric indicators, 20 capped categorical embeddings.
    state = {"numeric_output_names": NUMERIC + [n + "__missing" for n in NUMERIC],
             "cat_cardinalities": [2050] * len(CATEGORICAL)}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(state, settings).to(device)
    optimizer = torch.optim.AdamW(model.make_parameter_groups(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    x_num = torch.randn(settings["batch_size"], len(state["numeric_output_names"]), device=device)
    x_cat = torch.randint(0, 2050, (settings["batch_size"], len(CATEGORICAL)), device=device)
    labels = (torch.arange(settings["batch_size"], device=device) % 2).float()
    loss_value = None
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            output = model(x_num, x_cat).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output.float(), labels)
        if not torch.isfinite(loss):
            raise RuntimeError("CUDA architecture probe produced nonfinite loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_value = float(loss.detach())
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        probability = model(x_num, x_cat).float().sigmoid()
    torch.cuda.synchronize()
    assert probability.shape == (512, 1)
    assert torch.isfinite(probability).all() and (probability >= 0).all() and (probability <= 1).all()
    assert next(model.parameters()).device.type == "cuda"
    import rtdl_revisiting_models
    record.update(status="PASS", cuda_available=True, gpu=torch.cuda.get_device_name(0),
                  cuda_build=torch.version.cuda, amp_dtype=str(dtype), actual_model_device="cuda:0",
                  batch_size=settings["batch_size"], dummy_optimizer_steps=3,
                  numerical_probe_loss=loss_value, seconds=time.perf_counter() - started,
                  peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  peak_gpu_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                  official_package_source_sha256=sha256(rtdl_revisiting_models.__file__))
    write_json(ROOT / "reports" / "environment.json", record)
    print(f"Infrastructure PASS: {record['gpu']}; AMP={dtype}; actual model CUDA; "
          f"batch512 peak allocated={record['peak_gpu_allocated_mib']:.1f} MiB. "
          "This is NOT a real-data smoke test.", flush=True)
    return record


if __name__ == "__main__":
    check()
