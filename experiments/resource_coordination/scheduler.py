"""Process-safe FIFO CUDA admission at complete trial/stage boundaries.

Call lease before CUDA allocation and before active-training budget accounting.
The inner stage must return and release its CUDA objects before lease exit.
No process is terminated here. Uncoordinated legacy workers remain blockers.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import psutil

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
OWNERS = ("advanced_ft_transformer_goal_20261010", "advanced_selective_ft_20261010")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return default


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    # Windows observers may temporarily open the destination without delete
    # sharing. Keep the same complete payload and retry the atomic replacement
    # while the caller retains its guard; persistent denial still propagates.
    deadline = time.monotonic() + 8.0
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError as error:
            if (os.name != "nt" or getattr(error, "winerror", None) not in (5, 32)
                    or time.monotonic() >= deadline):
                raise
            time.sleep(.05)


@contextmanager
def guard(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "scheduler.guard").open("a+b") as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b"0")
            handle.flush()
        while True:
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (OSError, BlockingIOError):
                time.sleep(.05)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def identity(owner):
    return {"owner": owner, "pid": os.getpid(),
            "process_created": psutil.Process().create_time()}


def alive(record):
    try:
        process = psutil.Process(record["pid"])
        return process.is_running() and abs(process.create_time() - record["process_created"]) < .01
    except psutil.AccessDenied:
        return True
    except (psutil.NoSuchProcess, KeyError):
        return False


def state_read(root):
    return read(Path(root) / "scheduler_state.json", {
        "schema_version": 1, "next_ticket": 1, "workers": {}, "queue": [], "lease": None})


def event(root, kind, record=None, **extra):
    with (Path(root) / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": now(), "event": kind, **(record or {}), **extra}, ensure_ascii=False) + "\n")
        handle.flush()


def clean(root, state):
    state["workers"] = {k: r for k, r in state["workers"].items() if alive(r)}
    state["queue"] = [r for r in state["queue"] if alive(r)]
    if state["lease"] and not alive(state["lease"]):
        event(root, "RECOVER_DEAD_OWNER", state["lease"])
        state["lease"] = None


def policy_read(root):
    policy = read(Path(root) / "policy.json")
    if not policy or policy.get("mode") != "exclusive_fifo_trial":
        raise RuntimeError("Missing or incompatible common GPU policy")
    return policy


def register_worker(owner, *, root=ROOT):
    root = Path(root)
    record = identity(owner)
    with guard(root):
        policy = policy_read(root)
        if owner not in policy["authorized_owners"]:
            raise RuntimeError("Owner is not in the common GPU policy")
        state = state_read(root)
        clean(root, state)
        others = [r for r in state["workers"].values() if r["owner"] == owner and r["pid"] != record["pid"]]
        if others:
            raise RuntimeError("This group already has a registered live worker")
        existing = state["workers"].get(str(record["pid"]))
        if existing and existing.get("state") in ("ACTIVE", "UNCLEAN_EXIT"):
            raise RuntimeError("Cannot register over an active or unclean lease")
        state["workers"][str(record["pid"])] = record | {"state": "IDLE", "registered_utc": now()}
        write(root / "scheduler_state.json", state)
        event(root, "REGISTER", record)
    return record


def gpu_blockers(state):
    query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                            "--format=csv,noheader"], capture_output=True, text=True,
                           check=True, timeout=15).stdout
    own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    known_idle = {int(k) for k, r in state["workers"].items()
                  if r.get("state") in ("IDLE", "WAITING") and alive(r)}
    blockers = []
    for line in query.splitlines():
        fields = line.split(",", 1)
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
            if pid in own or pid in known_idle:
                continue
            process = psutil.Process(pid)
            if "python" not in process.name().lower():
                continue
            command = process.cmdline()
            if "http.server" in command:
                continue
            module = command[command.index("-m") + 1] if "-m" in command else Path(command[1]).name if len(command) > 1 else "unknown"
            blockers.append({"pid": pid, "name": process.name(), "module": module,
                             "reason": "UNREGISTERED_CUDA_WORKER"})
        except psutil.AccessDenied:
            if "python" in fields[1].lower():
                blockers.append({"pid": pid, "reason": "PYTHON_WORKER_DETAILS_UNAVAILABLE"})
        except (ValueError, psutil.NoSuchProcess):
            pass
    return blockers


def legacy_blockers(root):
    return [r for r in read(Path(root) / "legacy_workers.json", []) if alive(r)]


def status_file(owner_directory, data):
    if owner_directory is not None:
        write(Path(owner_directory) / "resource_status.json", data | {"updated_utc": now()})


def torch_quiesce():
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return 0
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return int(torch.cuda.memory_allocated())


@contextmanager
def lease(owner, purpose, *, deadline=None, owner_directory=None, root=ROOT,
          poll_seconds=3., check_gpu=True):
    root = Path(root)
    policy = policy_read(root)
    if not check_gpu and not (root.resolve() != ROOT.resolve() and policy.get("testing_only")):
        raise RuntimeError("Disabling GPU checks is permitted only in isolated CPU test fixtures")
    register_worker(owner, root=root)
    record = identity(owner) | {"purpose": purpose, "lease_id": uuid.uuid4().hex}
    start = time.monotonic()
    acquired = False
    with guard(root):
        state = state_read(root)
        clean(root, state)
        record.update(ticket=state["next_ticket"], queued_utc=now())
        state["next_ticket"] += 1
        state["queue"].append(record)
        state["workers"][str(record["pid"])].update(state="WAITING", purpose=purpose)
        write(root / "scheduler_state.json", state)
        event(root, "QUEUED", record)
    try:
        while not acquired:
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError("Original experiment deadline reached while queued")
            with guard(root):
                state = state_read(root)
                clean(root, state)
                policy = policy_read(root)
                blockers = legacy_blockers(root)
                front = bool(state["queue"] and state["queue"][0]["lease_id"] == record["lease_id"])
                available = psutil.virtual_memory().available >= policy.get("minimum_free_ram_bytes", 0)
                if front and not state["lease"] and not blockers and available and policy.get("admission_enabled"):
                    blockers = gpu_blockers(state) if check_gpu else []
                    if not blockers:
                        acquired = True
                        record.update(acquired_utc=now(), waiting_seconds=time.monotonic()-start,
                                      memory_fraction=policy["memory_fraction"], cpu_threads=policy["cpu_threads"])
                        state["queue"].pop(0)
                        state["lease"] = record
                        state["workers"][str(record["pid"])].update(state="ACTIVE", purpose=purpose)
                        event(root, "ACQUIRED", record)
                write(root / "scheduler_state.json", state)
                snapshot = {"status": "GPU_LEASE_ACQUIRED" if acquired else "WAITING_GPU_FIFO",
                            "mode": "exclusive_fifo_trial", "owner": owner, "pid": record["pid"],
                            "purpose": purpose, "ticket": record["ticket"], "current_lease": state["lease"],
                            "admission_enabled": policy.get("admission_enabled", False),
                            "blockers": blockers, "ram_available": available,
                            "waiting_seconds": time.monotonic()-start,
                            "memory_fraction": policy["memory_fraction"], "cpu_threads": policy["cpu_threads"]}
                status_file(owner_directory, snapshot)
            if not acquired:
                time.sleep(poll_seconds)
        yield {"waiting_seconds": record["waiting_seconds"], "lease_id": record["lease_id"],
               "ticket": record["ticket"], "memory_fraction": record["memory_fraction"],
               "cpu_threads": record["cpu_threads"]}
    finally:
        allocated = torch_quiesce() if acquired else 0
        with guard(root):
            state = state_read(root)
            state["queue"] = [r for r in state["queue"] if r["lease_id"] != record["lease_id"]]
            worker = state["workers"].get(str(record["pid"]))
            if acquired and state["lease"] and state["lease"]["lease_id"] == record["lease_id"]:
                if allocated > policy_read(root).get("maximum_idle_allocated_bytes", 256*1024**2):
                    event(root, "RELEASE_PENDING_PROCESS_EXIT", record, cuda_allocated_bytes=allocated)
                    if worker:
                        worker["state"] = "UNCLEAN_EXIT"
                    status_file(owner_directory, {"status": "CUDA_OBJECTS_STILL_LIVE", "purpose": purpose,
                                                  "pid": record["pid"], "cuda_allocated_bytes": allocated})
                else:
                    state["lease"] = None
                    if worker:
                        worker["state"] = "IDLE"
                    event(root, "RELEASED", record, cuda_allocated_bytes=allocated)
                    status_file(owner_directory, {"status": "GPU_RELEASED", "owner": owner,
                                                  "purpose": purpose, "pid": record["pid"],
                                                  "cuda_allocated_bytes": allocated})
            elif worker:
                worker["state"] = "IDLE"
            write(root / "scheduler_state.json", state)


def configure_torch(resource, device=0):
    """Apply the lease's capacity settings; do not change model or batch settings."""
    with guard(ROOT):
        active = state_read(ROOT).get("lease")
        if not active or active["pid"] != os.getpid() or active["lease_id"] != resource["lease_id"]:
            raise RuntimeError("CUDA configuration requires the current common lease")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback is not permitted")
    torch.set_num_threads(resource["cpu_threads"])
    torch.cuda.set_per_process_memory_fraction(resource["memory_fraction"], device)
    actual_fraction = (torch.cuda.get_per_process_memory_fraction(device)
                       if hasattr(torch.cuda, "get_per_process_memory_fraction") else None)
    with guard(ROOT):
        event(ROOT, "CUDA_CONFIGURED", active,
              actual_cpu_threads=torch.get_num_threads(),
              requested_memory_fraction=resource["memory_fraction"],
              actual_memory_fraction=actual_fraction,
              gpu_name=torch.cuda.get_device_name(device))


def snapshot(*, root=ROOT):
    with guard(root):
        state = state_read(root)
        clean(root, state)
        write(Path(root) / "scheduler_state.json", state)
        return {"observed_utc": now(), "policy": policy_read(root), "state": state,
                "legacy_blockers": legacy_blockers(root)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("status", "enable", "pause"))
    args = parser.parse_args()
    if args.action != "status":
        with guard(ROOT):
            policy = policy_read(ROOT)
            policy.update(admission_enabled=args.action == "enable", updated_utc=now())
            write(ROOT / "policy.json", policy)
            event(ROOT, "ADMISSION_ENABLED" if args.action == "enable" else "ADMISSION_PAUSED")
    print(json.dumps(snapshot(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
