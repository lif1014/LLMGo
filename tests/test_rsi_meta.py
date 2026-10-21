"""Exercise the real JSONL subprocess transport with an isolated fake server."""
import json
import subprocess
import sys

import pytest

from llmgo.rsi.contracts import atomic_json, read_json
from llmgo.rsi.memory import MemoryStore
from llmgo.rsi.meta import CodexBackend


SERVER = r'''
import json, sys
def emit(obj):
    print(json.dumps(obj), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if "id" not in request:
        continue
    rid = request["id"]
    if method == "initialize":
        emit({"id":rid,"result":{}})
    elif method in ("thread/start", "thread/resume"):
        emit({"id":rid,"result":{"thread":{"id":"thread-one", "turns":[]}}})
    elif method == "turn/start":
        emit({"id":rid,"result":{"turn":{"id":"turn-one"}}})
        emit({"method":"item/agentMessage/delta","params":{"turnId":"turn-one","delta":"visible analysis"}})
        emit({"method":"thread/tokenUsage/updated","params":{"threadId":"thread-one","tokenUsage":{"total":{"inputTokens":100,"outputTokens":20},"last":{"totalTokens":120}}}})
        status = "failed" if request["params"]["input"][0]["text"] == "fail" else "completed"
        answer = json.dumps({"status":"ready","summary":"one hypothesis","proposal_path":"proposal.json"})
        emit({"method":"item/completed","params":{"turnId":"turn-one","item":{"id":"item-one","type":"agentMessage","text":answer}}})
        emit({"method":"turn/completed","params":{"turn":{"id":"turn-one","status":status,"items":[]}}})
'''


def test_codex_jsonl_resume_visible_events_and_crash_result_reuse(tmp_path):
    server = tmp_path/"fake-server.py"
    server.write_text(SERVER)
    workspace = tmp_path/"workspace"
    workspace.mkdir()
    atomic_json(workspace/"context.json", {"round": 1})
    memory = MemoryStore(tmp_path/"memory")
    site = {"user_root": str(tmp_path), "codex_home": str(tmp_path/"codex"), "codex_bin": "fake"}
    calls = []
    def popen(argv, **kwargs):
        calls.append(argv)
        assert kwargs["env"]["CODEX_HOME"] == site["codex_home"]
        return subprocess.Popen([sys.executable, str(server)], **kwargs)
    backend = CodexBackend(site, memory, popen=popen)
    first = backend.turn(workspace, "prepare")
    assert first["status"] == "ready"
    # Controller crash before prepare: consume the saved answer instead of another API turn.
    restored = CodexBackend(site, memory, popen=popen)
    assert restored.turn(workspace, "prepare") == first and len(calls) == 1
    # A changed context is a new turn in the same persisted thread.
    atomic_json(workspace/"context.json", {"round": 1, "invalid_proposal": "retry"})
    assert restored.turn(workspace, "prepare") == first and len(calls) == 2
    sends = [e["body"]["message"] for e in memory.recent(100, "meta_send")]
    assert any(e.get("method") == "thread/resume" for e in sends)
    assert memory.search("visible analysis")
    assert read_json(memory.root/"meta-session.json")["process"] is None
    assert read_json(memory.root/"meta-session.json")["token_usage"]["total"]["inputTokens"] == 100
    with pytest.raises(RuntimeError, match="turn failed"):
        restored.turn(workspace, "fail")
    assert read_json(memory.root/"meta-session.json")["status"] == "failed"
    assert memory.recent(1, "meta_failure")[0]["body"]["error_type"] == "RuntimeError"
    assert memory.recent(1, "meta_status")[0]["body"]["status"] == "failed"



@pytest.mark.parametrize("failed", [False, True], ids=["completed-turn", "failed-turn"])
def test_shutdown_preserves_trailing_protocol_usage_and_stderr(tmp_path, failed):
    # The child flushes final usage/diagnostics only after the parent's input closes.
    # These are real pipe bytes, deliberately arriving after turn/completed.
    tail = r'''
emit({"method":"thread/tokenUsage/updated","params":{"threadId":"thread-one","tokenUsage":{"total":{"inputTokens":123,"outputTokens":45},"last":{"totalTokens":168}}}})
emit({"id":999,"method":"late/serverRequest","params":{}})
print("shutdown diagnostic Bearer abcdef0123456789", file=sys.stderr, flush=True)
'''
    server = tmp_path/"trailing-server.py"
    server.write_text(SERVER+tail)
    workspace = tmp_path/"workspace"
    workspace.mkdir()
    memory = MemoryStore(tmp_path/"memory")
    site = {"user_root": str(tmp_path), "codex_home": str(tmp_path/"codex"), "codex_bin": "fake"}
    def popen(argv, **kwargs):
        return subprocess.Popen([sys.executable, str(server)], **kwargs)
    backend = CodexBackend(site, memory, popen=popen)
    if failed:
        with pytest.raises(RuntimeError, match="turn failed"):
            backend.turn(workspace, "fail")
    else:
        assert backend.turn(workspace, "prepare")["status"] == "ready"
    session = read_json(memory.root/"meta-session.json")
    assert session["token_usage"]["total"]["inputTokens"] == 123
    assert session["status"] == ("failed" if failed else "completed")
    assert session["process"] is None
    diagnostics = memory.recent(20, "meta_stderr")
    assert any("shutdown diagnostic" in e["body"]["message"]["text"] for e in diagnostics)
    assert "abcdef0123456789" not in memory.events.read_text()
    updates = [e for e in memory.recent(100, "meta_event")
               if e["body"]["message"].get("method") == "thread/tokenUsage/updated"]
    assert len(updates) == 2
    assert len({e["id"] for e in updates}) == 2
    assert len(memory.search("late/serverRequest")) == 1
