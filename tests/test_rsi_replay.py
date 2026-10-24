"""Read-only replay pages and the handoff from compact history to live raw events."""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from llmgo.rsi.contracts import atomic_json
from llmgo.server.rsi import router
from llmgo.store.jsonl import TrajectoryStore


def setup_game(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMGO_RSI_ROOT", str(tmp_path))
    root = tmp_path/"demo"
    atomic_json(root/"campaign.json", {"campaign_id": "demo"})
    folder = "jobs/j1/output/harness_runs/g"
    store = TrajectoryStore(root/folder)
    store.write_config({"id": "g", "model_seat": "white"})
    app = FastAPI()
    app.include_router(router)
    return TestClient(app), root, folder, store


def test_compact_history_is_readonly_and_raw_tail_starts_at_exact_byte(tmp_path, monkeypatch):
    client, root, folder, store = setup_game(tmp_path, monkeypatch)
    store.append({"type": "position", "fen": "8/8/8/8/8/8/8/K6k w - - 0 1", "side": "white"})
    store.append({"type": "model_request", "status": "started", "prompt": "提示"*30000})
    store.append({"type": "model_request", "status": "finished", "reasoning": "思考"*30000, "text": "e2e4"})
    store.append({"type": "position", "fen": "8/8/8/8/8/8/8/K6k b - - 1 1", "side": "black"})
    store.append({"type": "model_request", "status": "started", "prompt": "下一次请求"})
    raw_cursor = store.events_path.stat().st_size
    store.append({"type": "token", "channel": "reasoning", "text": "实时尾端"})
    before = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    url = "/api/rsi/campaigns/demo/game-events"
    original = Path.open

    def no_raw_read(path, *args, **kwargs):
        if path == store.events_path:
            raise AssertionError("Compact replay must not scan the raw generation journal")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_raw_read)
    rows, cursor = [], 0
    while True:
        response = client.get(url, params={"folder": folder, "view": "compact", "cursor": cursor, "limit": 2})
        assert response.status_code == 200
        result = response.json()
        assert result["source"] == "short_events.jsonl"
        rows.extend(result["events"])
        cursor = result["cursor"]
        if not result["more"]:
            break
    assert result["raw_cursor"] == raw_cursor
    assert len(rows) == 5 and all(row["event"]["type"] != "token" for row in rows)
    assert rows[1]["event"]["type"] == "model_request"
    assert rows[1]["event"]["prompt"] and rows[1]["event"]["display_truncated"]
    assert rows[2]["event"]["output_excerpt"]["omitted"] > 0
    monkeypatch.setattr(Path, "open", original)
    tail = client.get(url, params={"folder": folder, "cursor": raw_cursor}).json()
    assert tail["source"] == "events.jsonl" and not tail["more"]
    assert tail["events"][0]["event"]["text"] == "实时尾端"
    assert tail["events"][0]["offset"] == raw_cursor
    assert {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()} == before
    assert client.post(url, params={"folder": folder}).status_code == 405


def test_game_pages_bound_large_records_and_preserve_cursor_and_position(tmp_path, monkeypatch):
    client, _, folder, store = setup_game(tmp_path, monkeypatch)
    # This legacy game has no compact file. Projection preserves replay fields
    # even when a prompt exceeds the general journal's 24KB display threshold.
    store.short_events_path.unlink()
    records = [{"type": "request", "status": "started", "prompt": "长提示"*30000} for _ in range(20)]
    records += [{"type": "request", "status": "finished", "raw_response": "x"*1100000},
                {"type": "position", "fen": "8/8/8/8/8/8/8/K6k b - - 1 1", "side": "black"}]
    lines = [(json.dumps(e, ensure_ascii=False)+"\n").encode() for e in records]
    store.events_path.write_bytes(b"".join(lines)+b'{"unfinished":')
    cursor, rows = 0, []
    url = "/api/rsi/campaigns/demo/game-events"
    while True:
        response = client.get(url, params={"folder": folder, "view": "compact", "cursor": cursor, "limit": 256})
        assert response.status_code == 200 and len(response.content) < 210000
        result = response.json()
        assert result["source"] == "events.jsonl"
        rows.extend(result["events"])
        if result["cursor"] == cursor:
            break
        cursor = result["cursor"]
    assert cursor == sum(map(len, lines)) and len(rows) == len(records)
    assert rows[0]["event"]["type"] == "request" and rows[0]["event"]["prompt"]
    assert rows[-2]["event"]["display_truncated"]
    assert rows[-1]["event"]["type"] == "position"
    assert rows[-2]["length"] == len(lines[-2])
    assert client.get(url, params={"folder": folder, "cursor": 1}).status_code == 400
    assert client.get(url, params={"folder": folder, "view": "unknown"}).status_code == 422
    assert client.get(url, params={"folder": "../outside"}).status_code == 404


def test_replay_cannot_follow_a_log_symlink_outside_campaign(tmp_path, monkeypatch):
    client, _, folder, store = setup_game(tmp_path, monkeypatch)
    outside = tmp_path/"private.jsonl"
    outside.write_text('{"private":true}\n')
    store.short_events_path.unlink()
    store.short_events_path.symlink_to(outside)
    response = client.get("/api/rsi/campaigns/demo/game-events", params={"folder": folder, "view": "compact"})
    assert response.status_code == 404


def test_referee_requests_restore_missing_compact_prompt_and_thought_by_direct_seek(tmp_path, monkeypatch):
    client, _, folder, store = setup_game(tmp_path, monkeypatch)
    # The shared writer predates referee request events. It strips these fields
    # from the compact copy, while their exact raw references remain available.
    store.append({"type": "request", "status": "started", "prompt": "当前提示"})
    store.append({"type": "request", "status": "finished", "text": "e2e4", "reasoning": "该手的思考",
                  "raw_response": "SSE"*1000000})
    original = Path.open
    reads = []

    class BoundedReader:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def seek(self, offset):
            reads.append(("seek", offset))
            return self.stream.seek(offset)
        def read(self, length=-1):
            assert 0 <= length <= 524288
            reads.append(("read", length))
            return self.stream.read(length)

    def inspect(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        return BoundedReader(stream) if path == store.events_path else stream

    monkeypatch.setattr(Path, "open", inspect)
    response = client.get("/api/rsi/campaigns/demo/game-events", params={"folder": folder, "view": "compact"})
    assert response.status_code == 200
    rows = response.json()["events"]
    assert rows[0]["event"]["prompt"] == "当前提示"
    assert rows[1]["event"]["reasoning"] == "该手的思考" and rows[1]["event"]["text"] == "e2e4"
    assert "raw_response" not in rows[1]["event"]
    assert len(reads) == 4 and reads[2][1] > 0
