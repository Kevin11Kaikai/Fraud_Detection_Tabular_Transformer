"""Archive the authors' public Kaggle responses without executing notebook cells."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REFS = [
    "cdeotte/rapids-feature-engineering-fraud-0-96",
    "cdeotte/xgb-fraud-with-magic-0-9600",
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(ref):
    name = ref.replace("/", "__")
    path = HERE / "references" / (name + ".response.json")
    url = "https://www.kaggle.com/api/v1/kernels/pull/" + ref
    if path.exists():
        body = path.read_bytes()
    else:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        body = response.content
        json.loads(body)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    value = json.loads(body)
    meta = value.get("metadata", {})
    print(ref, "keys", list(value), "metadata_ref", meta.get("ref"), flush=True)
    blob = value.get("blob", value.get("source", value.get("code")))
    if isinstance(blob, dict):
        blob = blob.get("source", blob.get("code", blob.get("content", blob)))
    if isinstance(blob, str):
        notebook = json.loads(blob)
    elif isinstance(blob, dict) and "cells" in blob:
        notebook = blob
    else:
        notebook = next((v for v in value.values() if isinstance(v, dict) and "cells" in v), None)
    record = {"ref": ref, "url": url, "response_file": str(path.relative_to(HERE)),
              "response_sha256": sha(body), "metadata": meta,
              "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    if notebook is not None:
        target = path.with_name(name + ".ipynb")
        data = json.dumps(notebook, ensure_ascii=False, indent=2).encode("utf-8")
        if target.exists() and target.read_bytes() != data:
            raise RuntimeError("Refusing to overwrite a different notebook snapshot")
        target.write_bytes(data)
        code = "\n\n".join(
            "# CELL " + str(i) + "\n" + ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
            for i, c in enumerate(notebook["cells"]) if c["cell_type"] == "code"
        )
        source = target.with_suffix(".source.py")
        source.write_text(code, encoding="utf-8")
        record.update(notebook_file=str(target.relative_to(HERE)), notebook_sha256=sha(data),
                      source_file=str(source.relative_to(HERE)), source_sha256=sha(code.encode("utf-8")),
                      cells=len(notebook["cells"]))
    return record


if __name__ == "__main__":
    records = list(ThreadPoolExecutor(2).map(fetch, REFS))
    out = HERE / "references" / "source_receipt.json"
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{k: r[k] for k in ["ref", "response_sha256", "notebook_file", "cells"] if k in r} for r in records], indent=2))
