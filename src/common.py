"""Shared provenance, timing, and configuration helpers."""
from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprints():
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
            for folder in [ROOT / "src", ROOT / "configs"]
            for p in sorted(folder.glob("*")) if p.is_file()}


def environment():
    packages = {}
    for name in ["torch", "numpy", "pandas", "scipy", "scikit-learn",
                 "lightgbm", "rtdl-revisiting-models", "kaggle", "matplotlib", "psutil"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
                              "--format=csv"], capture_output=True, text=True, timeout=10)
        gpu = smi.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        gpu = str(exc)
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": packages, "nvidia_smi": gpu}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def positive_weight(y):
    positives = int(np.asarray(y).sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        raise ValueError("Training requires both fraud and nonfraud examples")
    return float(np.clip(np.sqrt(negatives / positives), 1.0, 10.0))


def mode_paths(mode):
    if mode not in {"smoke", "full"}:
        raise ValueError(mode)
    prepared = ROOT / "results" / "prepared" / mode
    results = ROOT / "results" / mode
    models = ROOT / "models" / mode
    for path in [prepared, results, models]:
        path.mkdir(parents=True, exist_ok=True)
    return prepared, results, models


def load_partition(prepared, name):
    with np.load(Path(prepared) / f"{name}.npz", allow_pickle=False) as file:
        return {key: file[key] for key in file.files}


def probability_check(probability):
    probability = np.asarray(probability)
    if probability.ndim != 1 or not np.isfinite(probability).all():
        raise ValueError("Probabilities must be finite and one dimensional")
    if (probability < 0).any() or (probability > 1).any():
        raise ValueError("Probabilities outside [0,1]")
    return probability
