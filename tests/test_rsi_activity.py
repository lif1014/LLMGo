"""Web observability stays bounded while original evidence remains recoverable."""
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llmgo.rsi.contracts import atomic_json, canonical
from llmgo.rsi.memory import MemoryStore
from llmgo.server.rsi import activity_page, page, router
from llmgo.store.activity import project


def event(number, kind="research_note", body=None):
    return {"id": f"event-{number}", "ts": "2026-10-21T12:00:00Z", "kind": kind,
            "body": body if body is not None else {"text": f"记录 {number}"}}


def indexed(root, records):
    root.mkdir(parents=True, exist_ok=True)
    raw = b"".join((canonical(record)+"\n").encode() for record in records)
    (root/"events.jsonl").write_bytes(raw)
    return MemoryStore(root), raw


def client_for(root, monkeypatch):
    monkeypatch.setenv("LLMGO_RSI_ROOT", str(root.parent))
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_latest_history_forward_paging_and_exact_unicode_raw_offsets(tmp_path):
    records = [event(i) for i in range(250)]
    records.insert(90, event("state", "state", {"state": {"ignored": "large snapshot"}}))
    memory, raw = indexed(tmp_path, records)
    latest = activity_page(tmp_path, "research")
    assert len(latest["events"]) == 80 and latest["head"] == 250
    assert latest["events"][0]["event"]["id"] == "event-170"
    assert latest["events"][-1]["event"]["id"] == "event-249"
    assert latest["has_older"] and not latest["more_new"]
    older = activity_page(tmp_path, "research", before=latest["older_cursor"])
    assert len(older["events"]) == 80
    assert older["events"][-1]["ordinal"] < latest["events"][0]["ordinal"]
    for row in older["events"]+latest["events"]:
        exact = json.loads(raw[row["offset"]:row["offset"]+row["length"]])
        assert exact["id"] == row["event"]["id"]
    assert activity_page(tmp_path, "research", cursor=latest["cursor"])["events"] == []
    memory.append("research_note", {"text": "刚到的新事件"}, event_id="new")
    delta = activity_page(tmp_path, "research", cursor=latest["cursor"])
    assert [r["event"]["id"] for r in delta["events"]] == ["new"]
    assert memory.append("research_note", {"text": "刚到的新事件"}, event_id="new")["id"] == "new"
    assert len(activity_page(tmp_path, "research", cursor=250)["events"]) == 1


def test_large_output_is_bounded_and_index_rebuild_preserves_raw(tmp_path):
    huge = "完整命令输出\n"*100000
    memory = MemoryStore(tmp_path)
    saved = memory.append("command", {"argv": ["cctl", "job", "get"], "stdout": huge,
                                      "stderr": "", "returncode": 0, "seconds": 2.5})
    raw = memory.events.read_bytes()
    view = activity_page(tmp_path, "research")
    assert len(canonical(view).encode()) < 20000
    row = view["events"][0]
    assert json.loads(raw[row["offset"]:row["offset"]+row["length"]]) == saved
    assert json.loads(raw)["body"]["stdout"] == huge
    memory.db.unlink()
    MemoryStore(tmp_path)
    assert memory.events.read_bytes() == raw
    assert activity_page(tmp_path, "research")["events"] == view["events"]


def test_research_recommendation_has_readable_title_summary_and_complete_evidence(tmp_path, monkeypatch):
    root = tmp_path/"demo"
    memory = MemoryStore(root)
    body = {"title":"起点建议：r2＋原始Qwen＋修补PPO（待采用）",
            "summary":"建议已发布，模型/Harness/skills与训练代码分开登记。",
            "status":"recommended_not_adopted", "text":"完整取证记录\n"*1000}
    saved = memory.append("research_note",body,event_id="startpoint-recommendation")
    atomic_json(root/"campaign.json",{"campaign_id":"demo","jobs":{},"attempts":[]})
    client = client_for(root,monkeypatch)
    page = client.get("/api/rsi/campaigns/demo/activity?limit=10").json()
    row = page["events"][0]
    assert row["event"]["channel"] == "research"
    assert row["event"]["title"] == body["title"] and row["event"]["text"] == body["summary"]
    raw = client.get("/api/rsi/campaigns/demo/artifact",params={"path":row["path"],"offset":row["offset"],"length":row["length"]}).json()
    assert json.loads(raw["text"]) == saved
    huge = project(event("large-note","research_note",{"title":"t"*10000,"text":"全文"*10000}))
    assert len(huge["title"]) < 250 and len(huge["text"]) < 1900


def test_projection_keeps_tool_lifecycle_usage_errors_and_stream_identity():
    def notification(method, params):
        return project(event(method, "meta_event", {"message": {"method": method, "params": params}}))
    common = {"threadId": "thread", "turnId": "turn"}
    started = notification("item/started", {**common, "item": {"id": "cmd", "type": "commandExecution", "command": ["python", "train.py"]}})
    delta = notification("item/commandExecution/outputDelta", {**common, "itemId": "cmd", "delta": "loss=0.1"})
    done = notification("item/completed", {**common, "item": {"id": "cmd", "type": "commandExecution", "exitCode": 1, "aggregatedOutput": "failure"}})
    assert started["stream_key"] == delta["stream_key"] == done["stream_key"]
    assert "train.py" in started["title"] and delta["mode"] == "append"
    assert done["category"] == "error" and done["mode"] == "replace"
    usage = notification("thread/tokenUsage/updated", {**common, "tokenUsage": {"total": {"inputTokens": 100, "outputTokens": 30}, "last": {"totalTokens": 50}}})
    assert usage["usage"]["total"]["inputTokens"] == 100
    assert project(event(1, "meta_failure", {"error": "transport failed"}))["category"] == "error"
    assert notification("error", {"error": {"message": "turn interrupted"}})["category"] == "error"
    assert project(event(2, "command", {"returncode": 1, "seconds": 3}))["category"] == "error"
    poll = project(event(3, "poll_finished", {"job_id": "j1", "checks": 8, "status": "waiting", "platform_status": "Running", "interval_seconds": 120}))
    assert "Running / waiting" in poll["title"] and "120秒" in poll["title"]


def test_recovery_events_and_overview_distinguish_latest_task_from_active(tmp_path,monkeypatch):
    root=tmp_path/'demo';memory,_=indexed(root,[])
    for kind in ['controller_retry','controller_recovered','meta_continuation','meta_protocol_recovery','meta_completion_committed']:
        memory.append(kind,{'stage':'meta','attempt':1,'next_retry_at':1234})
        assert memory.recent(1,kind)
        assert project(event(kind,kind,{}))['category']=='lifecycle'
    assert project(event('blocked','controller_blocked',{}))['category']=='error'
    atomic_json(root/'campaign.json',{'campaign_id':'demo','phase':'PLANNING','active_job':None,
        'jobs':{'j1':{'id':'j1','status':'Killed'}},'attempts':[],
        'controller_recovery':{'status':'waiting','attempt':1,'next_retry_at':1234}})
    result=client_for(root,monkeypatch).get('/api/rsi/campaigns/demo/overview').json()
    assert result['active_job'] is result['active_job_id'] is None
    assert result['latest_job']['id']=='j1'
    assert result['controller_recovery']['next_retry_at']==1234


def test_api_is_readonly_and_never_replays_raw_on_refresh(tmp_path, monkeypatch):
    root = tmp_path/"demo"
    indexed(root, [event(i) for i in range(180)])
    atomic_json(root/"campaign.json", {"campaign_id": "demo", "jobs": {}, "attempts": [], "site": {"private": "not display data"}})
    client = client_for(root, monkeypatch)
    original = Path.open
    def reject_journal(path, *args, **kwargs):
        if path.name == "events.jsonl":
            raise AssertionError("HTTP refresh must not scan raw journals")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", reject_journal)
    url = "/api/rsi/campaigns/demo"
    result = client.get(url+"/activity").json()
    assert len(result["events"]) == 80 and result["events"][-1]["event"]["id"] == "event-179"
    assert client.get(url+"/activity?cursor=1&before=2").status_code == 400
    assert client.get(url+"/activity?source=unknown").status_code == 422
    assert "site" not in client.get(url+"/overview").json()
    # A legacy campaign isn't rebuilt as a side effect of opening a browser.
    (root/"memory.sqlite").unlink()
    missing = client.get(url+"/activity").json()
    assert missing["available"] is False and "index-activity" in missing["note"]
    assert not (root/"memory.sqlite").exists()


def test_large_event_pages_have_budget_and_complete_record_cursors(tmp_path):
    path = tmp_path/"events.jsonl"
    lines = [(json.dumps({"type": "model_request", "output": "x"*18000})+"\n").encode() for _ in range(80)]
    huge = (json.dumps({"output": "y"*2000000})+"\n").encode()
    path.write_bytes(huge+b"".join(lines)+b'{"not_yet_complete":')
    first = page(path, 0, 256)
    assert first["events"][0]["event"]["display_truncated"]
    assert first["events"][0]["length"] == len(huge)
    assert len(canonical(first).encode()) < 210000
    rows, cursor = first["events"], first["cursor"]
    while True:
        current = page(path, cursor, 256)
        if current["cursor"] == cursor:
            break
        rows += current["events"]
        cursor = current["cursor"]
    assert len(rows) == 81
    assert cursor == len(huge)+sum(map(len, lines))
    assert path.read_bytes()[cursor:] == b'{"not_yet_complete":'
    with pytest.raises(Exception) as failure:
        page(path, 1, 80)
    assert failure.value.status_code == 400


@pytest.mark.parametrize('game_count',[10,20])
def test_overview_job_files_and_evaluations_are_compact_and_isolated(tmp_path, monkeypatch,game_count):
    root = tmp_path/"demo"
    memory, _ = indexed(root, [])
    jobs = {"j1": {"id": "j1", "role": "candidate", "worker_wall_seconds": 3600, "allocated_gpu_count": 1}}
    atomic_json(root/"campaign.json", {"campaign_id": "demo", "phase": "WAITING", "jobs": jobs, "attempts": [], "active_job": "j1"})
    atomic_json(root/"meta-session.json", {"status": "running", "thread_id": "t", "answer": "private large answer"})
    job = root/"jobs/j1"
    atomic_json(job/"manifest.json", {"payload": {"protocol":{"games":game_count}}})
    atomic_json(job/"output/progress.json", {"stage": "training", "curriculum_records": 64})
    (job/"output/training").mkdir()
    (job/"output/research.log").write_text("loss\n"*300000)
    for job_id in ("j1", "j2"):
        atomic_json(root/f"jobs/{job_id}/output/harness_runs/g/config.json", {"id": "g", "model_seat": "white"})
    memory.append("evaluation", {"candidate_id": "candidate", "level": 1, "statistics": {"score": .75}, "ref": {"job_id": "j1"}})
    client = client_for(root, monkeypatch)
    url = "/api/rsi/campaigns/demo"
    overview = client.get(url+"/overview").json()
    assert overview["worker_gpu_hours"] == 1 and "answer" not in overview["meta"]
    files = client.get(url+"/job-files?job_id=j1").json()
    assert files['formal_games']==game_count
    assert files["progress"]["stage"] == "training"
    assert any(f["bytes"] > 1000000 for f in files["files"])
    assert len(canonical(files)) < 2000
    assert len(client.get(url+"/games?job_id=j1").json()) == 1
    assert "j1" in client.get(url+"/games?job_id=j1").json()[0]["folder"]
    assert client.get(url+"/job-files?job_id=../j2").status_code == 404
    evidence = client.get(url+"/evaluations").json()[0]
    assert evidence["worker_gpu_hours"] == 1 and evidence["statistics"]["score"] == .75
    # File bodies are still available only through explicit byte-range reads.
    body = client.get(url+"/artifact", params={"path": "jobs/j1/output/research.log", "offset": 5, "length": 6}).json()
    assert body["text"] == "loss\nl" and body["next_offset"] == 11
