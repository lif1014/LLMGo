"""Append-only evidence with a rebuildable SQLite search index and process lock."""
from __future__ import annotations

import fcntl
import json
import os
import re
import socket
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from .contracts import atomic_json, canonical, utc
from llmgo.store.activity import project

_SECRET = re.compile(r"(?i)(authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)")


class CampaignLocked(RuntimeError):
    """An actual process holds the advisory lock; a stale file is not enough."""


def redact(value):
    if isinstance(value, dict):
        return {k: "[REDACTED]" if _SECRET.fullmatch(str(k)) else redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~-]+", "Bearer [REDACTED]", value)
        value = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}", "[REDACTED]", value)
    return value


class MemoryStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.events = self.root / "events.jsonl"
        self.db = self.root / "memory.sqlite"
        with self.connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, ts TEXT, kind TEXT, body TEXT)")
            connection.execute("CREATE TABLE IF NOT EXISTS activity (id TEXT PRIMARY KEY, offset INTEGER, length INTEGER, view TEXT)")
            connection.execute("CREATE INDEX IF NOT EXISTS events_kind ON events(kind)")
        self.rebuild()

    def connect(self):
        db = sqlite3.connect(self.db, timeout=30)
        # Avoid SQLite WAL on a shared network filesystem.
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("PRAGMA synchronous=FULL")
        return db

    @contextmanager
    def lock(self, name="controller"):
        path = self.root / (name + ".lock")
        with path.open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CampaignLocked("Another process already owns this campaign") from exc
            handle.seek(0)
            handle.truncate()
            handle.write(canonical({"pid": os.getpid(), "host": socket.gethostname(), "started": utc()}))
            handle.flush()
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def append(self, kind: str, body: dict, event_id=None) -> dict:
        event = {"id": event_id or uuid.uuid4().hex, "ts": utc(), "kind": kind, "body": redact(body)}
        encoded = canonical(event)
        with (self.root / "events.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with self.connect() as db:
                previous = db.execute("SELECT body FROM events WHERE id=?", (event["id"],)).fetchone()
                if previous:
                    old = json.loads(previous[0])
                    if old["kind"] != kind or old["body"] != event["body"]:
                        raise ValueError("Conflicting duplicate event ID")
                    return old
            with self.events.open("ab") as stream:
                offset = stream.tell()
                stream.write((encoded + "\n").encode())
                stream.flush()
                os.fsync(stream.fileno())
            with self.connect() as db:
                db.execute("INSERT INTO events VALUES(?,?,?,?)", (event["id"], event["ts"], kind, encoded))
                self._index_activity(db, event, offset, len((encoded+"\n").encode()))
        return event

    @staticmethod
    def _index_activity(db, event, offset, length):
        if event["kind"] == "state":
            return
        if db.execute("SELECT 1 FROM activity WHERE id=?", (event["id"],)).fetchone():
            return
        view = project(event)
        if view is not None:
            db.execute("INSERT INTO activity VALUES(?,?,?,?)", (event["id"], offset, length, canonical(view)))

    def rebuild(self):
        """Replay the authoritative log; never silently discard a torn record."""
        if not self.events.exists():
            return
        with (self.root / "events.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with self.connect() as db, self.events.open("rb") as stream:
                offset = 0
                for number, line in enumerate(stream, 1):
                    try:
                        event = json.loads(line)
                    except ValueError as exc:
                        raise RuntimeError(f"Damaged memory event at line {number}; retain and repair the original") from exc
                    prior = db.execute("SELECT body FROM events WHERE id=?", (event["id"],)).fetchone()
                    if prior and json.loads(prior[0]) != event:
                        raise ValueError("Conflicting event in journal")
                    db.execute("INSERT OR IGNORE INTO events VALUES(?,?,?,?)",
                               (event["id"], event["ts"], event["kind"], canonical(event)))
                    self._index_activity(db, event, offset, len(line))
                    offset += len(line)

    def recent(self, limit=30, kind=None):
        limit = min(500, max(1, int(limit)))
        sql = "SELECT body FROM events"
        args = []
        if kind:
            sql += " WHERE kind=?"
            args.append(kind)
        sql += " ORDER BY rowid DESC LIMIT ?"
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute(sql, [*args, limit])]

    def search(self, query, limit=20):
        # Parameterized literal substring search works even without SQLite FTS.
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute(
                "SELECT body FROM events WHERE instr(lower(body),lower(?)) > 0 ORDER BY rowid DESC LIMIT ?",
                (query, min(100, max(1, int(limit)))))]

    def save_state(self, state):
        self.append("state", {"state": state})
        atomic_json(self.root / "campaign.json", state)

    def state(self):
        rows = self.recent(1, "state")
        if not rows:
            raise ValueError("Campaign has not been initialized")
        return rows[0]["body"]["state"]

    def handoff(self, body):
        self.append("handoff", body)
        atomic_json(self.root / "memory" / "handoff.json", redact(body))

    def context(self):
        state = self.state()
        summary = {k: state.get(k) for k in ("campaign_id", "phase", "round", "working", "recommended", "incumbent")}
        best = state.get("best")
        summary["best"] = {"candidate_id":best["candidate"]["id"], "score":best["score"], "panel":best["panel"]} if best else None
        for key in ("working", "incumbent"):
            item = state.get(key) or {}
            summary[key] = {k:item.get(k) for k in ("id", "model_path", "model_sha256", "hypothesis")}
        summary["protected_source"] = {k:state["source"][k] for k in ("root", "sha256")}
        summary["evaluation_objective"] = "L1-L8, 20 balanced games each; partial tests never update best"
        summary["active_job"] = state["jobs"].get(state.get("active_job"))
        summary["recent_attempts"] = state["attempts"][-3:]
        summary["attempt_count"] = len(state["attempts"])
        summary["evidence_root"] = str(self.root)
        with self.connect() as db:
            rows = db.execute(
                "SELECT body FROM events WHERE kind IN ('decision','evaluation','experiment_failed',"
                "'proposal_invalid','explicit_retry','opponent_level_promoted') ORDER BY rowid DESC LIMIT 12")
            recent = [json.loads(row[0]) for row in rows]
        context = {"state": summary, "recent": recent, "handoff": self.recent(1, "handoff")}
        # A bounded handoff, never the polling transcript or raw terminal logs.
        job_id = state.get("active_job") or next(reversed(state["jobs"]), None)
        if job_id:
            path = self.root / "jobs" / job_id / "executor/report.json"
            context["executor"] = {"mode": state["site"].get("executor_mode", "embedded"),
                                   "report_path": str(path), "raw_events_path": str(self.root / "executor/events.jsonl")}
            if path.is_file():
                report = json.loads(path.read_text(encoding="utf-8"))
                context["executor"]["latest"] = {k: report.get(k) for k in
                    ("job_id", "status", "task_id", "platform_status", "summary", "artifact", "updated")}
        return context
