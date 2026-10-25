"""Stop an owned idle Controller under its journal/lifecycle locks for restart."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import sqlite3
import time

from .contracts import REPO, atomic_json, identifier, inside, read_json
from .memory import CampaignLocked, MemoryStore
from .meta import _process_identity


def stop_waiting_controller(site, campaign):
    root = inside(site["user_root"], Path(site["campaign_root"])/identifier(campaign))
    repo = inside(site["user_root"], site["repo"])
    if not (root/"campaign.json").is_file():
        raise ValueError("Unknown campaign; no process was stopped")
    with (root/"controller.lock").open("a+") as controller_lock:
        try:
            fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            return {"stopped": False, "reason": "Controller is already stopped"}
        controller_lock.seek(0)
        owner = json.load(controller_lock)
        if owner.get("host") != socket.gethostname():
            raise ValueError("Run restart-control in the original devspace; Controller host differs")
        pid = owner.get("pid")
        identity = _process_identity(pid) if type(pid) is int and pid > 1 else None
        if identity is None or pid == os.getpid():
            raise ValueError("Cannot establish Controller process identity")
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        # A reused PID or an unrelated service must never become a signal target.
        if (Path(f"/proc/{pid}/cwd").resolve() != Path(repo).resolve()
                or not any(arg in {"scripts/rsi.py", str(Path(repo)/"scripts/rsi.py")} for arg in argv)
                or "run" not in argv or campaign not in argv or "executor" in argv):
            raise ValueError("Process does not match this campaign's Controller command")
        with ExitStack() as stack:
            # No platform submission/poll or journal write may be in flight.
            for path in (root/"executor/lifecycle.lock", root/"events.lock"):
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context(path.open("a+"))
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise CampaignLocked("Controller/Executor is writing or polling; retry later without stopping it") from exc
            with sqlite3.connect((root/"memory.sqlite").resolve().as_uri()+"?mode=ro", uri=True) as db:
                row = db.execute("SELECT body FROM events WHERE kind='state' ORDER BY rowid DESC LIMIT 1").fetchone()
            if row is None:
                raise ValueError("Missing authoritative campaign state")
            state = json.loads(row[0])["body"]["state"]
            meta = read_json(root/"meta-session.json") if (root/"meta-session.json").exists() else {}
            active = state.get("active_job")
            if state.get("phase") != "WAITING" or not active or meta.get("process") is not None:
                raise ValueError("Restart-control requires WAITING with no active Meta process; keep ongoing research intact")
            job = state["jobs"][active]
            intent = read_json(Path(job["manifest"]).with_name("submission.json"))
            expected = f"llmgo-rsi:{campaign}:{active}:{job['sha256']}"
            if not intent.get("task_id") or intent.get("description") != expected:
                raise ValueError("Task submission identity is not confirmed; reconcile before restarting")
            if _process_identity(pid) != identity:
                raise ValueError("Controller identity changed; no signal sent")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = root/"recoveries"/f"controller-{stamp}"
            atomic_json(backup/"campaign.json", state)
            atomic_json(backup/"meta-session.json", meta)
            atomic_json(backup/"controller-process.json", identity)
            os.kill(pid, signal.SIGTERM)  # Controller only; no cluster job or Meta child.
            deadline = time.monotonic()+30
            while True:
                try:
                    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Controller lock not released; no replacement started")
                    time.sleep(.1)
        # The journal lock is released before append; no old writer remains.
        MemoryStore(root).append("operator_controller_stopped", {"process": identity,
            "active_job": active, "task_id": intent["task_id"], "backup": str(backup),
            "cluster_task_stopped": False})
        return {"stopped": True, "active_job": active, "task_id": intent["task_id"], "backup": str(backup)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stop only an idle owned Controller; do not stop its cluster task")
    parser.add_argument("--site", default=str(REPO/"configs/rsi/site.beijing.json"))
    parser.add_argument("--campaign", default="chess-autonomous-v2")
    args = parser.parse_args(argv)
    print(json.dumps(stop_waiting_controller(read_json(args.site), args.campaign), ensure_ascii=False))
