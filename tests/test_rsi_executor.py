"""Lifecycle, process boundary and context isolation; no cluster/model calls."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llmgo.remote.cctl import CctlCommandError, CctlExecutor, SubmissionUncertain
from llmgo.remote.lifecycle import JobExecutor
from llmgo.rsi.contracts import REPO, atomic_json, digest, file_hash, read_json
from llmgo.rsi.controller import Controller
from llmgo.rsi.memory import CampaignLocked, MemoryStore
from llmgo.server.rsi import router

RAW = "RAW_EXECUTOR_ONLY_" + "x"*4000


class Clock:
    def __init__(self):
        self.value = 1000.

    def __call__(self):
        return self.value

    def advance(self, seconds=121):
        self.value += seconds


class Platform:
    def __init__(self):
        self.creates = self.gets = self.log_reads = 0
        self.status = "Running"
        self.timeout = False

    def submit(self, manifest):
        self.creates += 1
        wrapper = read_json(manifest)
        p = wrapper["payload"]
        intent = {"task_id": 4170, "description": f"llmgo-rsi:{p['campaign_id']}:{p['job_id']}:{wrapper['sha256']}"}
        atomic_json(Path(manifest).with_name("submission.json"), intent)
        return intent

    def get(self, intent):
        self.gets += 1
        if self.timeout:
            self.timeout = False
            raise subprocess.TimeoutExpired(["cctl", "job", "get", str(intent["task_id"])], 90, output=RAW)
        return {"id": intent["task_id"], "status": self.status, "diagnostic": RAW}

    def logs(self, intent, pod):
        self.log_reads += 1
        assert pod == "job-rsi-4170-main-0"
        return RAW


@pytest.fixture
def prepared(tmp_path, request):
    repo = tmp_path/"repo"
    cards = repo/"src/llmgo/harness/skills/cards.json"
    cards.parent.mkdir(parents=True)
    shutil.copyfile(REPO/"src/llmgo/harness/skills/cards.json", cards)
    (repo/"scripts").mkdir()
    (repo/"scripts/rsi_worker.py").write_text("# isolated fixture\n")
    site = {"user_root": str(tmp_path), "repo": str(repo), "campaign_root": str(tmp_path/"campaigns"),
            "worktree_root": str(tmp_path/"workspaces"), "base_model": str(tmp_path/"model"),
            "engine": "fixture", "gpu_count": 1, "poll_seconds": 30, "executor_max_poll_seconds": 120,
            "executor_mode": getattr(request, "param", "embedded")}
    c = Controller.create(site, "isolation")
    c.engine_probe = lambda path, spec: {"engine": "fixture", "spec": spec}
    c.step()
    transport = Platform()
    clock = Clock()
    c.lifecycle = JobExecutor(c.root, c.site, transport=transport, clock=clock)
    return c, c.state["jobs"][c.state["active_job"]], transport, clock


def result_for(c, job):
    payload = read_json(job["manifest"])["payload"]
    games = [{"game_id": f"{pair}-{side}", "model_seat": side, "pair_id": pair, "result": "draw"}
             for pair in range(10) for side in ("white", "black")]
    from llmgo.rsi.ladder import batch_stats
    metrics = {"track":payload["track"],"candidate_source_identity":payload["candidate_source"]["sha256"],"games": games, "candidate_id": payload["candidate_id"], "skills_sha256": payload["skills_sha256"],
               "source_identity": payload["source"]["sha256"], "comparison_id": payload["comparison_id"],
               "protocol_identity": digest(payload["protocol"]), "opponent_identity": digest(payload["engine_identity"]),
               "seed_schedule": payload["seed_schedule"], "model_identity": {"sha256": "fixture-model"}}
    metrics["statistics"] = batch_stats(metrics)
    path = Path(payload["output"])/"metrics.json"
    atomic_json(path, metrics)
    atomic_json(path.with_name("result.json"), {"manifest_sha256": job["sha256"], "artifact": str(path),
                                               "artifact_sha256": file_hash(path), "worker_wall_seconds": 99,
                                               "allocated_gpu_count": 1})
    return path


def test_371_status_checks_do_not_fill_research_context_or_trigger_meta(prepared):
    c, job, transport, clock = prepared
    for _ in range(371):
        clock.advance()
        assert c.step()["status"] == "waiting"
    assert transport.creates == 1 and transport.gets == 371 and transport.log_reads == 0
    raw = c.lifecycle.memory.events.read_text()
    research = c.memory.events.read_text()
    context = json.dumps(c.memory.context())
    assert raw.count(RAW) == 371
    assert RAW not in research and RAW not in context
    assert len(c.memory.recent(500, "executor_transition")) == 1
    assert len(context) < 6000
    assert c.memory.recent(10, "meta_send") == []
    assert c.memory.recent(10, "meta_event") == []
    assert c.memory.context()["executor"]["latest"]["status"] == "waiting"


@pytest.mark.parametrize("prepared", ["external"], indirect=True)
def test_external_executor_progresses_while_controller_lock_is_held_and_replays_report(prepared):
    c, job, transport, clock = prepared
    class NoMeta:
        def turn(self, *args):
            raise AssertionError("Meta-Agent must not be invoked for pending jobs")
    # No external Executor yet: Controller only reads local report files.
    c.run(backend=NoMeta(), max_steps=1, poll_seconds=.001)
    assert transport.creates == transport.gets == 0
    result_for(c, job)
    transport.status = "Succeeded"
    script = '''
import json, os, sys
from test_rsi_executor import Platform
from llmgo.rsi.memory import MemoryStore
from llmgo.remote.lifecycle import JobExecutor
root = sys.argv[1]
site = MemoryStore(root).state()["site"]
transport = Platform()
transport.status = "Succeeded"
report = JobExecutor(root, site, transport=transport).run(once=True)
print(json.dumps({"status":report["status"], "pid":os.getpid(), "creates":transport.creates,
                  "gets":transport.gets, "logs":transport.log_reads}))
'''
    with c.memory.lock():
        child = subprocess.run([sys.executable, "-c", script, str(c.root)], capture_output=True, text=True,
            check=True, timeout=15, env={**os.environ, "PYTHONPATH": f"{REPO/'src'}:{Path(__file__).parent}"})
    result = json.loads(child.stdout)
    assert result["status"] == "collected" and result["pid"] != os.getpid()
    assert result["creates"] == result["gets"] == result["logs"] == 1
    # Simulate Controller dying before collection and restarting after Executor finished.
    restored = Controller(c.root, executor=transport, engine_probe=c.engine_probe)
    assert restored.step()["status"] == "collected"
    assert restored.state["phase"] == "BASELINE"
    assert transport.creates == transport.gets == 0  # no platform call from read-only Controller process
    assert restored.state["jobs"][job["id"]]["worker_wall_seconds"] == 99
    assert restored.memory.context()["executor"]["latest"]["artifact"]["path"].endswith("metrics.json")


def test_transient_status_timeout_keeps_live_task_across_executor_restart(prepared):
    c, job, transport, clock = prepared
    c.step()
    transport.timeout = True
    clock.advance()
    assert c.step()["status"] == "waiting"
    assert c.state["phase"] == "WAITING"
    assert c.lifecycle.report(job)["status"] == "retrying"
    restarted = JobExecutor(c.root, c.site, transport=transport, clock=clock)
    clock.advance()
    assert restarted.advance(job)["status"] == "waiting"
    assert transport.creates == 1 and transport.gets == 3
    assert restarted.report(job)["task_id"] == 4170


def test_poll_backoff_persists_without_requerying_or_doubling_submissions(prepared):
    c, job, transport, clock = prepared
    first = c.lifecycle.advance(job)
    for _ in range(10):
        c.lifecycle.advance(job)
    assert transport.gets == 1
    clock.advance(30)
    second = c.lifecycle.advance(job)
    assert second["interval_seconds"] > first["interval_seconds"]
    restarted = JobExecutor(c.root, c.site, transport=transport, clock=clock)
    restarted.advance(job)
    assert transport.gets == 2 and transport.creates == 1


def test_failed_job_logs_are_collected_once_and_not_forwarded_to_meta(prepared):
    c, job, transport, clock = prepared
    transport.status = "OOMKilled"
    assert c.step()["status"] == "needs_input"
    report = c.lifecycle.report(job)
    assert Path(report["evidence"]["platform_log"]).read_text() == RAW
    for _ in range(3):
        c.lifecycle.advance(job, force=True)
    assert transport.gets == transport.log_reads == 1
    assert RAW not in json.dumps(c.memory.context())
    assert RAW not in c.memory.events.read_text()
    assert c.state["phase"] == "NEEDS_INPUT"


def test_succeeded_platform_waits_for_artifact_and_rejects_mismatched_receipt(prepared):
    c, job, transport, clock = prepared
    transport.status = "Succeeded"
    assert c.lifecycle.advance(job)["status"] == "waiting_for_artifact"
    path = result_for(c, job)
    result = read_json(path.with_name("result.json"))
    result["manifest_sha256"] = "another-experiment"
    atomic_json(path.with_name("result.json"), result)
    clock.advance()
    assert c.lifecycle.advance(job)["status"] == "artifact_invalid"
    with pytest.raises(ValueError, match="identity/hash"):
        c.step()
    assert c.state["incumbent"]["id"] == "initial"
    assert transport.creates == 1


def test_executor_process_lock_and_report_identity(prepared):
    c, job, transport, clock = prepared
    with c.lifecycle.memory.lock("lifecycle"):
        assert c.step()["status"] == "waiting"
    assert transport.creates == 0
    c.lifecycle.advance(job)
    report = c.lifecycle.report(job)
    report["manifest_sha256"] = "wrong"
    atomic_json(c.lifecycle.report_path(job), report)
    with pytest.raises(ValueError, match="different job"):
        c.lifecycle.report(job)
    with c.lifecycle.memory.lock("service"):
        with pytest.raises(CampaignLocked):
            c.lifecycle.run(once=True)


def test_real_cctl_transport_raw_failure_is_isolated_and_secrets_redacted(tmp_path):
    memory = MemoryStore(tmp_path/"executor")
    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, RAW, "Bearer hidden-secret-token")
    transport = CctlExecutor({"cctl_bin": "cctl"}, memory, runner, record_commands=True)
    with pytest.raises(CctlCommandError) as failure:
        transport.command(["job", "get", "4170"])
    assert RAW not in str(failure.value)
    journal = memory.events.read_text()
    assert RAW in journal and "hidden-secret-token" not in journal


def test_uncertain_submission_keeps_intent_and_next_attempt_only_reconciles(prepared):
    c, job, transport, clock = prepared
    def uncertain_submit(path):
        transport.creates += 1
        intent_path = Path(path).with_name("submission.json")
        if not intent_path.exists():
            atomic_json(intent_path, {"status": "UNKNOWN", "manifest": str(path)})
        raise SubmissionUncertain(RAW)
    transport.submit = uncertain_submit
    assert c.step()["status"] == "submission_uncertain"
    assert c.state["phase"] == "WAITING"
    assert RAW not in c.memory.events.read_text() and RAW not in json.dumps(c.memory.context())
    # Actual CctlExecutor is responsible for reconciliation; this fixture models
    # its next call without creating another task. Existing transport tests cover it.
    def reconciled(path):
        payload = read_json(path)["payload"]
        intent = {"task_id": 4170, "description": f"llmgo-rsi:{payload['campaign_id']}:{job['id']}:{job['sha256']}"}
        atomic_json(Path(path).with_name("submission.json"), intent)
        return intent
    transport.submit = reconciled
    clock.advance()
    assert c.step()["status"] == "waiting"
    assert transport.creates == 1 and transport.gets == 1


def test_explicit_retry_requires_a_confirmed_terminal_handle(prepared):
    c, job, transport, clock = prepared
    c.lifecycle.advance(job)
    with pytest.raises(ValueError, match="not confirmed terminal"):
        c.lifecycle.confirm_failed(job)
    transport.status = "Failed"
    c.lifecycle.confirm_failed(job)
    assert transport.creates == 1


def test_executor_raw_api_is_separate_from_research_events(prepared, monkeypatch):
    c, job, transport, clock = prepared
    c.step()
    monkeypatch.setenv("LLMGO_RSI_ROOT", str(c.root.parent))
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    url = f"/api/rsi/campaigns/{c.root.name}"
    assert RAW not in client.get(url+"/events").text
    assert RAW in client.get(url+"/executor-events").text
    status = client.get(url+"/executor").json()
    assert status["report"]["task_id"] == 4170 and RAW not in json.dumps(status)


def test_reconciliation_accepts_actual_v2_name_only_listings(tmp_path):
    intent_path = tmp_path/'submission.json'
    atomic_json(intent_path, {'description': 'fixture-experiment', 'status': 'UNKNOWN'})
    memory = MemoryStore(tmp_path/'memory')
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv[1:])
        if argv[1:3] == ['job', 'list']:
            # Real platform list responses omit id and description.
            result = {'tasks': [{'name': 'tasks/12', 'status': 'Running'},
                                {'name': 'tasks/13', 'status': 'Succeeded'}], 'total': 2}
        else:
            assert argv[1:3] == ['job', 'get']
            task_id = int(argv[3])
            result = {'id': task_id, 'name': f'tasks/{task_id}', 'status': 'Running',
                      'description': 'fixture-experiment' if task_id == 12 else 'unrelated',
                      'project': 'rsi', 'cluster': 'beijing', 'resourcePoolId': 525, 'priority': 'NORMAL'}
        return subprocess.CompletedProcess(argv, 0, json.dumps(result), '')
    transport = CctlExecutor({'cctl_bin': 'cctl', 'project': 'rsi'}, memory, runner)
    recovered = transport.reconcile(intent_path)
    assert recovered['task_id'] == 12
    assert transport.get(recovered)['id'] == 12
    assert not any(c[:2] == ['job', 'create'] for c in calls)


def test_reconciliation_deduplicates_id_and_resource_name_aliases(tmp_path):
    path = tmp_path/'submission.json'
    atomic_json(path, {'description': 'same', 'status': 'UNKNOWN'})
    def runner(argv, **kwargs):
        assert argv[1:3] == ['job', 'list']
        return subprocess.CompletedProcess(argv, 0, json.dumps({'tasks': [
            {'id': 12, 'description': 'same', 'status': 'Running'},
            {'name': 'tasks/12', 'description': 'same', 'status': 'Running'}]}), '')
    transport = CctlExecutor({'cctl_bin': 'cctl', 'project': 'rsi'}, MemoryStore(tmp_path/'memory'), runner)
    assert transport.reconcile(path)['task_id'] == 12


def test_reconciliation_retains_intent_for_distinct_matching_tasks(tmp_path):
    path = tmp_path/'submission.json'
    atomic_json(path, {'description': 'same', 'status': 'UNKNOWN'})
    before = path.read_bytes()
    def runner(argv, **kwargs):
        assert argv[1:3] == ['job', 'list']
        return subprocess.CompletedProcess(argv, 0, json.dumps({'tasks': [
            {'name': f'tasks/{i}', 'description': 'same', 'status': 'Running'} for i in (12,13)]}), '')
    transport = CctlExecutor({'cctl_bin': 'cctl', 'project': 'rsi'}, MemoryStore(tmp_path/'memory'), runner)
    with pytest.raises(SubmissionUncertain, match='2 matching'):
        transport.reconcile(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize('handle', [{'name':'projects/12'}, {'id':12,'name':'tasks/13'}, {'id':True}, {}])
def test_invalid_reconciliation_handles_preserve_uncertainty(tmp_path, handle):
    path = tmp_path/'submission.json'
    atomic_json(path, {'description':'same','status':'UNKNOWN'})
    before = path.read_bytes()
    def runner(argv, **kwargs):
        assert argv[1:3] == ['job','list']
        result={'tasks':[{**handle,'description':'same','status':'Running'}]}
        return subprocess.CompletedProcess(argv,0,json.dumps(result),'')
    transport=CctlExecutor({'cctl_bin':'cctl','project':'rsi'},MemoryStore(tmp_path/'memory'),runner)
    with pytest.raises(SubmissionUncertain,match='Invalid task handle'):
        transport.reconcile(path)
    assert path.read_bytes()==before


def test_incomplete_listing_does_not_adopt_a_partially_unique_match(tmp_path):
    path=tmp_path/'submission.json'
    atomic_json(path,{'description':'same','status':'UNKNOWN'})
    before=path.read_bytes()
    def runner(argv,**kwargs):
        assert argv[1:3]==['job','list']
        page=int(argv[argv.index('--page')+1])
        rows=[{'id':i,'description':'same' if i==1 else 'unrelated','status':'Running'}
              for i in range((page-1)*100+1,page*100+1)]
        return subprocess.CompletedProcess(argv,0,json.dumps({'tasks':rows}),'')
    transport=CctlExecutor({'cctl_bin':'cctl','project':'rsi'},MemoryStore(tmp_path/'memory'),runner)
    with pytest.raises(SubmissionUncertain,match='listing limit reached'):
        transport.reconcile(path)
    assert path.read_bytes()==before


def test_task_get_rejects_wrong_id_even_with_matching_description_and_resources(tmp_path):
    def runner(argv,**kwargs):
        result={'id':13,'description':'same','status':'Running','project':'rsi',
                'cluster':'beijing','resourcePoolId':525,'priority':'NORMAL'}
        return subprocess.CompletedProcess(argv,0,json.dumps(result),'')
    transport=CctlExecutor({'cctl_bin':'cctl','project':'rsi'},MemoryStore(tmp_path/'memory'),runner)
    with pytest.raises(ValueError,match='identity'):
        transport.get({'task_id':12,'description':'same'})


def test_submission_accepts_resource_name_and_remains_idempotent(tmp_path,monkeypatch):
    calls=[]
    def runner(argv,**kwargs):
        calls.append(argv[1:3])
        if argv[1:3]==['pool','list']:
            result=[{'name':'model-autosft'}]
        else:
            assert argv[1:3]==['job','create']
            result={'name':'tasks/44','status':'Queued'}
        return subprocess.CompletedProcess(argv,0,json.dumps(result),'')
    transport=CctlExecutor({'cctl_bin':'cctl','project':'rsi'},MemoryStore(tmp_path/'memory'),runner)
    monkeypatch.setattr(transport,'request',lambda path: (['job','create'],'same'))
    path=tmp_path/'manifest.json'
    assert transport.submit(path)['task_id']==44
    assert transport.submit(path)['task_id']==44
    assert calls.count(['job','create'])==1
