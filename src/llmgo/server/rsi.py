"""Read-only RSI dashboard endpoints; controller operations stay in the CLI."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import Literal
from fastapi import APIRouter, HTTPException, Query

from llmgo.rsi.contracts import identifier, inside, read_json

router = APIRouter(prefix="/api/rsi")


@lru_cache(maxsize=8)
def _state_file(path, modified, size):
    return read_json(path)


def campaign_state(root):
    path = root/"campaign.json"
    stat = path.stat()
    return _state_file(str(path), stat.st_mtime_ns, stat.st_size)


def limited_json(path, maximum=1048576):
    if not path.is_file() or path.stat().st_size > maximum:
        return None
    return read_json(path)


def activity_page(root, source, cursor=None, before=None, limit=80):
    """Read a derived index only: HTTP never replays a large raw journal."""
    database = root / ("executor" if source == "executor" else "") / "memory.sqlite"
    path = "executor/events.jsonl" if source == "executor" else "events.jsonl"
    empty = {"available": False, "events": [], "cursor": cursor or 0, "has_older": False,
             "note": "活动索引尚未建立；运行 rsi.py index-activity CAMPAIGN，或用原始事件分页查看"}
    if not database.is_file():
        return empty
    try:
        with sqlite3.connect(database.as_uri()+"?mode=ro", uri=True, timeout=.2) as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='activity'").fetchone():
                return empty
            first = db.execute("SELECT rowid FROM activity ORDER BY rowid LIMIT 1").fetchone()
            last = db.execute("SELECT rowid FROM activity ORDER BY rowid DESC LIMIT 1").fetchone()
            low, high = first[0] if first else 0, last[0] if last else 0
            descending = before is not None or cursor is None
            clause, args = (" WHERE rowid < ?", [before]) if before is not None else ((" WHERE rowid > ?", [cursor]) if cursor is not None else ("", []))
            rows = db.execute("SELECT rowid, offset, length, view FROM activity"+clause+
                              (" ORDER BY rowid DESC LIMIT ?" if descending else " ORDER BY rowid ASC LIMIT ?"), [*args, limit])
            selected, total = [], 0
            for ordinal, offset, length, view in rows:
                cost = len(view.encode())+256
                if selected and total+cost > 196608:
                    break
                total += cost
                selected.append({"ordinal": ordinal, "event": json.loads(view), "offset": offset, "length": length, "path": path})
            if descending:
                selected.reverse()
            next_cursor = selected[-1]["ordinal"] if selected else cursor or high or 0
            older = selected[0]["ordinal"] if selected else before
            return {"available": True, "events": selected, "cursor": next_cursor, "older_cursor": older,
                    "has_older": bool(older and low and older > low), "head": high or 0,
                    "more_new": bool(high and next_cursor < high), "display_bytes": total, "source": source}
    except sqlite3.Error as exc:
        raise HTTPException(503, "活动索引正在写入或不可用；稍后重试") from exc


def campaign_root():
    return Path(os.environ.get("LLMGO_RSI_ROOT", "/user/lyh4170/rsi_campaigns")).resolve()


def campaign(name):
    try:
        path = inside(campaign_root(), campaign_root()/identifier(name))
        if not (path/"campaign.json").is_file():
            raise ValueError("Unknown campaign")
        return path
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


def game_display(event):
    """Bound replay snapshots while preserving positions and lazy raw references."""
    shown = {k: v for k, v in event.items() if k not in {
        "raw_response", "request_payload", "generated_token_ids", "token_ids", "llm_input"}}
    for key in ("prompt", "reasoning", "output", "text"):
        value = shown.get(key)
        if isinstance(value, str) and len(value) > 16000:
            shown[key] = value[:8000]+"\n[…显示已截断；可读取原始事件…]\n"+value[-8000:]
    if isinstance(shown.get("output_excerpt"), dict):
        shown["output_excerpt"] = {k: v for k, v in shown["output_excerpt"].items()
                                   if k not in {"generated_tokens", "tokenizer"}}
        for key in ("head", "tail"):
            value = shown["output_excerpt"].get(key)
            if isinstance(value, str) and len(value) > 4000:
                shown["output_excerpt"][key] = value[:4000]+"\n[…显示已截断；可读取原始事件…]"
    if len(json.dumps(shown, ensure_ascii=False).encode()) > 24000:
        keep = ("type", "status", "seat", "ply", "sequence", "timestamp", "request_id", "raw_ref",
                "fen", "side", "last_move", "in_check", "tool", "reason", "finish_reason", "uci", "ok",
                "source", "channel", "replace", "trace_format", "output_excerpt")
        compact = {k: shown[k] for k in keep if k in shown}
        for key in ("prompt", "reasoning", "output", "text"):
            if key in shown:
                compact[key] = str(shown[key])[:4000]
        # Full structured tools/skills remain available through the original record.
        compact["display_truncated"] = True
        compact["preview"] = json.dumps(shown, ensure_ascii=False)[:4000]
        shown = compact
    return shown


def request_fields(path, ref):
    """Seek a referee request directly; never replay its large raw SSE response.

    Older compact writers omit prompt/reasoning for the newer `request` type.
    Decode complete fields from a bounded JSON prefix, stopping before raw SSE.
    """
    if type(ref.get("offset")) is not int or type(ref.get("length")) is not int or min(ref["offset"], ref["length"]) < 0:
        return {}
    with path.open("rb") as stream:
        stream.seek(ref["offset"])
        prefix = stream.read(min(ref["length"], 524288)).decode("utf-8", "replace")
    decoder, index, result = json.JSONDecoder(), 1, {}
    while index < len(prefix):
        try:
            index += len(prefix[index:])-len(prefix[index:].lstrip())
            key, index = decoder.raw_decode(prefix, index)
            index += len(prefix[index:])-len(prefix[index:].lstrip())
            if prefix[index:index+1] != ":" or key in {"raw_response", "request_payload", "token_ids"}:
                break
            index += 1
            index += len(prefix[index:])-len(prefix[index:].lstrip())
            value, index = decoder.raw_decode(prefix, index)
            if key in {"prompt", "reasoning", "output", "text"}:
                result[key] = value
            index += len(prefix[index:])-len(prefix[index:].lstrip())
            if prefix[index:index+1] != ",":
                break
            index += 1
        except (ValueError, IndexError):
            break  # A long final field is still available through the raw link.
    return result


def page(path, cursor, limit, *, replay=False, original=None):
    rows, next_cursor, displayed = [], cursor, 0
    if not path.is_file():
        return {"events": [], "cursor": cursor}
    with path.open("rb") as stream:
        if cursor > path.stat().st_size:
            raise HTTPException(409, "Log was replaced or truncated; reload from cursor 0")
        stream.seek(cursor)
        if cursor:
            stream.seek(cursor-1)
            if stream.read(1) != b"\n":
                raise HTTPException(400, "Cursor must point to a JSONL record boundary")
        for _ in range(limit):
            start = stream.tell()
            maximum = 1048576 if replay else 65536
            line = stream.readline(maximum)
            complete = line.endswith(b"\n")
            if not line or not complete:
                if len(line) < maximum:
                    break
                # Skip long records in bounded chunks; don't parse megabytes of raw stdout.
                tail = line
                while tail and not tail.endswith(b"\n"):
                    tail = stream.readline(65536)
                if not tail:
                    break  # a writer has not yet committed the full record
            length = stream.tell()-start
            cost = min(length, 24000)+256
            if rows and displayed+cost > 196608:
                break
            # Preserve structured small events; large records retain an exact raw reference.
            if length <= (maximum if replay else 24000) and complete:
                try:
                    shown = json.loads(line)
                    if replay:
                        if original and shown.get("type") == "request" and isinstance(shown.get("raw_ref"), dict):
                            shown.update(request_fields(original, shown["raw_ref"]))
                        shown = game_display(shown)
                except ValueError as exc:
                    raise HTTPException(409, f"Malformed log at byte {start}") from exc
            else:
                shown = {"display_truncated": True, "preview": line[:16000].decode("utf-8", "replace")}
            if replay:
                cost = len(json.dumps(shown, ensure_ascii=False).encode())+256
                if rows and displayed+cost > 196608:
                    break
            displayed += cost
            next_cursor = stream.tell()
            rows.append({"event": shown, "offset": start, "length": length})
    return {"events": rows, "cursor": next_cursor}


@router.get("/campaigns")
def campaigns():
    root = campaign_root()
    if not root.is_dir():
        return []
    result = []
    for path in sorted(root.glob("*/campaign.json")):
        try:
            inside(root, path)
            state = read_json(path)
            result.append({k: state.get(k) for k in ("campaign_id", "phase", "updated")})
        except (ValueError, OSError):
            continue
    return result


@router.get("/campaigns/{name}")
def status(name: str):
    state = campaign_state(campaign(name))
    # This is a display snapshot; the controller replays its journal on recovery.
    return {k: v for k, v in state.items() if k not in ("source", "site")}


@router.get("/campaigns/{name}/overview")
def overview(name: str):
    root = campaign(name)
    from llmgo.rsi.research_control import read_control
    state = campaign_state(root)
    jobs = state.get("jobs", {})
    active = state.get("active_job")
    latest = jobs.get(active) or next(reversed(jobs.values()), None)
    session = limited_json(root/"meta-session.json", 262144) or {}
    result = {k: state.get(k) for k in ("campaign_id", "phase", "round", "created", "updated", "incumbent", "working", "recommended", "failure", "needs_input", "controller_recovery")}
    for key in ("working", "incumbent"):
        item = state.get(key) or {}
        result[key] = {k:item.get(k) for k in ("id", "model_path", "model_sha256", "hypothesis")}
    best = state.get("best")
    result["best"] = {"candidate_id":best["candidate"]["id"],"score":best["score"],"panel":best["panel"]} if best else None
    result["candidate"] = {k: state.get("candidate", {}).get(k) for k in ("id","hypothesis","evaluation")} if state.get("candidate") else None
    result.update(controller_status_version=1, active_job=jobs.get(active), latest_job=latest, active_job_id=active,
                  job_count=len(jobs), attempt_count=len(state.get("attempts", [])),
                  accepted_count=sum(bool(a.get("new_best")) for a in state.get("attempts", [])),
                  worker_gpu_hours=sum((j.get("worker_wall_seconds") or 0)*(j.get("allocated_gpu_count") or 0)/3600 for j in jobs.values()),
                  recent_jobs=[{k: j.get(k) for k in ("id", "role", "candidate_id", "level", "track", "status", "task_id", "executor_status")}
                               for j in list(jobs.values())[-30:]],
                  meta={k: session.get(k) for k in ("thread_id", "turn_id", "status", "cwd", "updated", "process", "token_usage")},
                  research_control=read_control(root), server_time=time.time())
    return result


@router.get("/campaigns/{name}/activity")
def activity(name: str, source: Literal["research", "executor"] = "research", cursor: int | None = Query(None, ge=0),
             before: int | None = Query(None, ge=1), limit: int = Query(80, ge=1, le=120)):
    if cursor is not None and before is not None:
        raise HTTPException(400, "cursor 和 before 只能选择一个")
    return activity_page(campaign(name), source, cursor, before, limit)


@router.get("/campaigns/{name}/events")
def events(name: str, cursor: int = Query(0, ge=0), limit: int = Query(80, ge=1, le=256)):
    return page(campaign(name)/"events.jsonl", cursor, limit)


@router.get("/campaigns/{name}/executor")
def executor_status(name: str):
    root = campaign(name)
    state = campaign_state(root)
    job_id = state.get("active_job") or next(reversed(state.get("jobs", {})), None)
    path = inside(root, root/"jobs"/identifier(job_id)/"executor/report.json") if job_id else None
    heartbeat = root/"executor/heartbeat.json"
    return {"mode": state.get("site", {}).get("executor_mode", "embedded"),
            "report": read_json(path) if path and path.is_file() else None,
            "heartbeat": read_json(heartbeat) if heartbeat.is_file() else None,
            "raw_events": "executor/events.jsonl"}


@router.get("/campaigns/{name}/job-files")
def job_files(name: str, job_id: str):
    root = campaign(name)
    try:
        job = inside(root, root/"jobs"/identifier(job_id))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    if not (job/"manifest.json").is_file():
        raise HTTPException(404, "Unknown job")
    names = ("manifest.json", "submission.json", "executor/report.json", "executor/platform.log", "output/progress.json",
             "output/environment.json", "output/failure.json", "output/result.json", "output/vllm.log", "output/metrics.json",
             "output/research.log", "output/submission.json", "output/research-result.json", "output/checkpoint-ref.json", "output/training/latest.json", "output/training/training.jsonl",
             "output/training/training-result.json",
             "output/training/installed.freeze.txt")
    # Bounded pointer, no polling or scanning all historical rollout files.
    latest = limited_json(job/"output/training/latest.json", 65536) or {}
    for key in ("raw_path", "targets_path", "rollout_path"):
        if latest.get(key):
            try:
                path = inside(job/"output/training/rollouts", latest[key])
                names += (path.relative_to(job).as_posix(),)
            except ValueError:
                pass
    # Candidate import/runtime failures get a lazy file link too; inspect at most20.
    for candidate_log in islice((job/"output/harness_runs").glob("*/candidate.log"),20):
        names += (candidate_log.relative_to(job).as_posix(),)
    # Bounded listing: raw rollout/teacher/shard logs are fetched only on click.
    for pattern in ("output/gpu-*/*.log", "output/attempts/*.json", "output/training/lc0/*.json", "output/training/rollouts/*.jsonl"):
        for artifact in islice((job).glob(pattern), 8):
            names += (artifact.relative_to(job).as_posix(),)
    names += ("output/training/model-preflight.json", "output/training/resolved-config.json", "output/training/identity.json", "output/training/teacher.json", "output/training/training-result.json", "output/training/batch-latest.json", "output/training/metrics.jsonl")
    files = []
    for name in names:
        path = inside(root, job/name)
        if path.is_file():
            stat = path.stat()
            files.append({"path": path.relative_to(root).as_posix(), "bytes": stat.st_size, "modified": stat.st_mtime})
    results = list(islice((job/"output/game_results").glob("*.json"), 21))
    progress = limited_json(job/"output/progress.json", 16384)
    manifest = limited_json(job/"manifest.json", 262144) or {}
    payload = manifest.get("payload", {})
    expected = payload.get("protocol", {}).get("games", 20) if isinstance(payload, dict) else 20
    return {"files": files, "game_results": len(results), "formal_games": expected, "progress": progress}


@router.get("/campaigns/{name}/evaluations")
def evaluations(name: str, limit: int = Query(200, ge=1, le=200)):
    root = campaign(name)
    database = root/"memory.sqlite"
    if not database.is_file():
        return []
    state, total, costs = campaign_state(root), 0, {}
    for job in state.get("jobs", {}).values():
        total += (job.get("worker_wall_seconds") or 0)*(job.get("allocated_gpu_count") or 0)/3600
        costs[job["id"]] = total
    try:
        with sqlite3.connect(database.as_uri()+"?mode=ro", uri=True, timeout=.2) as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='activity'").fetchone():
                return []
            rows = db.execute("SELECT a.view FROM events e JOIN activity a ON a.id=e.id WHERE e.kind='evaluation' ORDER BY e.rowid DESC LIMIT ?", (limit,))
            result = []
            for row in rows:
                item = json.loads(row[0]).get("evaluation", {})
                item["worker_gpu_hours"] = costs.get(item.get("ref", {}).get("job_id"), 0)
                result.append(item)
            return result[::-1]
    except sqlite3.Error as exc:
        raise HTTPException(503, "评测索引暂不可用") from exc


@router.get("/campaigns/{name}/executor-events")
def executor_events(name: str, cursor: int = Query(0, ge=0), limit: int = Query(80, ge=1, le=256)):
    return page(campaign(name)/"executor/events.jsonl", cursor, limit)


@router.get("/campaigns/{name}/games")
def games(name: str, limit: int = Query(200, ge=1, le=1000), job_id: str | None = None):
    root = campaign(name)
    try:
        base = inside(root, root/"jobs"/identifier(job_id)) if job_id else root/"jobs"
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    result = []
    pattern = "output/harness_runs/*/config.json" if job_id else "*/output/harness_runs/*/config.json"
    for path in sorted(base.glob(pattern))[-limit:]:
        inside(root, path)
        config = read_json(path)
        summary = path.with_name("summary.json")
        result.append({"id": config["id"], "folder": path.parent.relative_to(root).as_posix(),
                       "candidate_id": config.get("candidate_id"), "model_seat": config.get("model_seat"),
                       "summary": read_json(summary) if summary.is_file() else None})
    return result


@router.get("/campaigns/{name}/game-events")
def game_events(name: str, folder: str, cursor: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=256),
                view: Literal["full", "compact"] = "full"):
    root = campaign(name)
    try:
        path = inside(root, root/folder)
        if not (path/"config.json").is_file() or "harness_runs" not in path.relative_to(root).parts:
            raise ValueError("Unknown game")
        source = "short_events.jsonl" if view == "compact" and (path/"short_events.jsonl").is_file() else "events.jsonl"
        log = inside(root, path/source)
        original = inside(root, path/"events.jsonl") if source == "short_events.jsonl" and (path/"events.jsonl").is_file() else None
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    result = page(log, cursor, limit, replay=True, original=original)
    result.update(source=source, more=log.is_file() and result["cursor"] < log.stat().st_size)
    # Jump from completed compact history to the raw tail for live token chunks.
    refs = [row["event"].get("raw_ref", {}) for row in result["events"]]
    result["raw_cursor"] = max((r["offset"]+r["length"] for r in refs
                                if type(r.get("offset")) is int and type(r.get("length")) is int), default=None)
    return result


@router.get("/campaigns/{name}/artifact")
def artifact(name: str, path: str, offset: int = Query(0, ge=0), length: int = Query(131072, ge=1, le=524288)):
    root = campaign(name)
    try:
        target = inside(root, root/path)
        if target.suffix not in (".json", ".jsonl", ".log", ".md", ".yaml") or not target.is_file():
            raise ValueError("Not a readable text artifact")
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    with target.open("rb") as stream:
        stream.seek(offset)
        content = stream.read(length)
        return {"text": content.decode("utf-8", "replace"), "offset": offset,
                "next_offset": stream.tell(), "eof": stream.tell() >= target.stat().st_size,
                "note": "Display byte slice; original artifact remains unchanged"}
