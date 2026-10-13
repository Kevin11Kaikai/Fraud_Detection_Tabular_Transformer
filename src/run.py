"""Bounded two-stage runner. Resume valid artifacts, never invent missing results."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .common import ROOT, read_json, sha256, source_fingerprints, utc_now, write_json


def execute(module, arguments, log_name):
    log = ROOT / "reports" / "logs" / log_name
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "-m", module, *arguments]
    print("RUN " + subprocess.list2cmdline(command), flush=True)
    # Child output is immediately visible and also retained verbatim.
    with log.open("a", encoding="utf-8") as file:
        file.write(f"\n[{utc_now()}] {subprocess.list2cmdline(command)}\n")
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace") as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                file.write(line)
                file.flush()
            returncode = process.wait()
    if returncode:
        raise RuntimeError(f"{module} exited {returncode}; exact output: {log}")


def resume_prepared(csv_path, mode):
    from .data import prepare
    prepared = ROOT / "results" / "prepared" / mode
    if not (prepared / "manifest.json").exists():
        return prepare(csv_path, mode, read_json(ROOT / "configs" / "mvp.json"))
    manifest = read_json(prepared / "manifest.json")
    if manifest["source_sha256"] != sha256(csv_path):
        raise ValueError("Raw data fingerprint changed; refusing to reuse prepared data")
    if manifest["config_sha256"] != sha256(ROOT / "configs" / "mvp.json"):
        raise ValueError("Configuration fingerprint changed; use a fresh project copy")
    if manifest["preprocessor_sha256"] != sha256(prepared / "preprocessor.json"):
        raise ValueError("Preprocessing fingerprint changed")
    for filename, expected in manifest["artifact_sha256"].items():
        if sha256(prepared / filename) != expected:
            raise ValueError(f"Prepared artifact fingerprint changed: {filename}")
    print(f"REUSE verified {mode} prepared data", flush=True)
    return manifest


def main(args):
    started = time.perf_counter()
    state_file = ROOT / "reports" / "run_status.json"
    state = {"started_utc": utc_now(), "status": "STARTED", "seed": 42,
             "smoke": "NOT_EXECUTED", "full": "NOT_EXECUTED", "source_sha256": source_fingerprints()}
    write_json(state_file, state)
    try:
        if not args.skip_environment:
            execute("src.check_environment", [], "environment.log")
        csv_path = Path(args.data).expanduser().resolve()
        if not csv_path.is_file():
            from .download_data import acquire
            csv_path = acquire()
        if csv_path is None or not csv_path.is_file():
            state.update(status="BLOCKED_DATA_ACCESS", problem="Real training CSV unavailable; Kaggle authentication missing",
                         elapsed_seconds=time.perf_counter() - started, finished_utc=utc_now())
            write_json(state_file, state)
            print("BLOCKED_DATA_ACCESS: smoke and full benchmark NOT EXECUTED; no fabricated metrics", flush=True)
            return 2
        state["data_path"] = str(csv_path)
        for mode in (["smoke", "full"] if args.stage == "all" else [args.stage]):
            if mode == "full":
                smoke_record = ROOT / "results" / "smoke" / "metric_recomputation.json"
                if not smoke_record.exists() or read_json(smoke_record)["status"] != "PASS":
                    raise ValueError("Full training requires a passing real-data smoke run")
                smoke_data = read_json(ROOT / "results" / "prepared" / "smoke" / "manifest.json")
                if smoke_data["source_sha256"] != sha256(csv_path):
                    raise ValueError("Smoke was run on a different dataset")
                if smoke_data["config_sha256"] != sha256(ROOT / "configs" / "mvp.json"):
                    raise ValueError("Configuration changed since smoke test; run a fresh smoke test")
            resume_prepared(csv_path, mode)
            state[mode] = "RUNNING"
            write_json(state_file, state)
            for key, module, artifact in [("lightgbm", "src.train_lightgbm", "lightgbm.txt"),
                                           ("ft_transformer", "src.train_ft_transformer", "ft_transformer.pt")]:
                record_path = ROOT / "results" / mode / f"{key}_training.json"
                if record_path.exists() and read_json(record_path).get("status") == "COMPLETED":
                    record = read_json(record_path)
                    if record["model_sha256"] != sha256(ROOT / "models" / mode / artifact):
                        raise ValueError("Checkpoint fingerprint mismatch")
                    if record["prepared_manifest_sha256"] != sha256(ROOT / "results" / "prepared" / mode / "manifest.json"):
                        raise ValueError("Model/prepared data mismatch")
                    if record["source_sha256"] != source_fingerprints():
                        raise ValueError("Code/config changed after training; refusing unqualified resume")
                    print(f"REUSE verified {mode} {key} checkpoint", flush=True)
                else:
                    execute(module, ["--mode", mode], f"{mode}_{key}.log")
            recomputed = ROOT / "results" / mode / "metric_recomputation.json"
            if recomputed.exists():
                execute("src.evaluate", ["--mode", mode, "--verify-only"], f"{mode}_recompute.log")
            else:
                execute("src.evaluate", ["--mode", mode], f"{mode}_evaluation.log")
            state[mode] = "PASS" if mode == "smoke" else "COMPLETED"
            state["elapsed_seconds"] = time.perf_counter() - started
            write_json(state_file, state)
        state.update(status="COMPLETED", finished_utc=utc_now(), elapsed_seconds=time.perf_counter() - started)
        write_json(state_file, state)
        return 0
    except Exception as exc:
        state.update(status="FAILED", error_type=type(exc).__name__, problem=str(exc),
                     finished_utc=utc_now(), elapsed_seconds=time.perf_counter() - started)
        write_json(state_file, state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "raw" / "train_transaction.csv"))
    parser.add_argument("--stage", choices=["smoke", "full", "all"], default="all")
    parser.add_argument("--skip-environment", action="store_true", help="Reuse a previously verified CUDA environment")
    raise SystemExit(main(parser.parse_args()))
