"""Official competition files only; never print credentials or use a mirror."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from .common import ROOT, utc_now, write_json


def acquire():
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    wanted = ["train_transaction.csv", "train_identity.csv"]
    found = {name: (raw / name).is_file() for name in wanted}
    home_config = Path.home() / ".kaggle"
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", str(home_config)))
    environment = os.environ.copy()
    if not config_dir.is_dir():
        config_dir = ROOT / ".kaggle"
        config_dir.mkdir(exist_ok=True)
        environment["KAGGLE_CONFIG_DIR"] = str(config_dir)
    credential_file = config_dir / "kaggle.json"
    env_pair = bool(environment.get("KAGGLE_USERNAME") and environment.get("KAGGLE_KEY"))
    cli = Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")
    record = {"checked_utc": utc_now(), "competition": "ieee-fraud-detection",
              "files": found, "config_dir": str(config_dir),
              "credential_file_present": credential_file.is_file(), "env_pair_present": env_pair,
              "api_token_present": bool(environment.get("KAGGLE_API_TOKEN")),
              "cli": str(cli), "status": "CHECKING", "download_attempted": False}
    if found["train_transaction.csv"]:
        record["status"] = "LOCAL_TRAINING_DATA_AVAILABLE"
        write_json(ROOT / "reports" / "dataset_access.json", record)
        print(record["status"], flush=True)
        return raw / "train_transaction.csv"
    if not cli.is_file():
        record.update(status="CLI_MISSING", problem="Install the pinned kaggle package in this environment")
        write_json(ROOT / "reports" / "dataset_access.json", record)
        raise RuntimeError(record["problem"])
    # Obtain exact CLI auth diagnosis even when no credentials exist.
    command = [str(cli), "competitions", "files", "-c", "ieee-fraud-detection"]
    record["authentication_command"] = ["kaggle", *command[1:]]
    if not credential_file.is_file() and not env_pair:
        completed = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=30,
                                   encoding="utf-8", errors="replace")
        record.update(status="AUTHENTICATION_MISSING", returncode=completed.returncode,
                      problem=(completed.stderr or completed.stdout).strip(),
                      note="Pinned Kaggle 1.7.4.5 requires legacy kaggle.json or username/key environment pair.")
        write_json(ROOT / "reports" / "dataset_access.json", record)
        print(record["status"] + "\n" + record["problem"], flush=True)
        return None
    completed = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=60,
                               encoding="utf-8", errors="replace")
    if completed.returncode:
        record.update(status="AUTH_OR_ACCESS_FAILED", returncode=completed.returncode,
                      problem=(completed.stderr or completed.stdout).strip())
        write_json(ROOT / "reports" / "dataset_access.json", record)
        raise RuntimeError(record["problem"])
    record["download_attempted"] = True
    write_json(ROOT / "reports" / "dataset_access.json", record)
    for filename in wanted:
        if (raw / filename).exists():
            continue
        command = [str(cli), "competitions", "download", "-c", "ieee-fraud-detection",
                   "-f", filename, "-p", str(raw)]
        completed = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=600,
                                   encoding="utf-8", errors="replace")
        if completed.returncode:
            record.update(status="DOWNLOAD_FAILED", returncode=completed.returncode,
                          problem=(completed.stderr or completed.stdout).strip(), failed_file=filename)
            write_json(ROOT / "reports" / "dataset_access.json", record)
            raise RuntimeError(record["problem"])
        archive = raw / (filename + ".zip")
        if archive.exists() and not (raw / filename).exists():
            with zipfile.ZipFile(archive) as file:
                members = [entry for entry in file.infolist() if Path(entry.filename).name == filename]
                if len(members) != 1:
                    raise ValueError(f"Expected exactly one {filename} in {archive}")
                # Fixed destination prevents archive path traversal.
                with file.open(members[0]) as source, (raw / filename).open("xb") as destination:
                    import shutil
                    shutil.copyfileobj(source, destination)
        if not (raw / filename).is_file():
            raise FileNotFoundError(f"Download succeeded but expected file not found: {raw / filename}")
    record.update(status="DOWNLOADED", files={name: (raw / name).is_file() for name in wanted})
    write_json(ROOT / "reports" / "dataset_access.json", record)
    return raw / "train_transaction.csv"


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    raise SystemExit(0 if acquire() else 2)
