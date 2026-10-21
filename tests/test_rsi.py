"""RSI contracts and recovery. Platform/model fixtures are not chess-strength evidence."""
import copy
import json
import shutil
import subprocess
from pathlib import Path

import chess
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llmgo.remote.cctl import CctlExecutor, SubmissionUncertain
from llmgo.rsi.artifacts import source_snapshot, verify_source, write_manifest
from llmgo.rsi.contracts import REPO, atomic_json, digest, file_hash, inside, protocol, read_json, validate_protocol
from llmgo.rsi.controller import Controller
from llmgo.rsi.ladder import batch_stats, opponent, panel_summary, evaluation_plan
from llmgo.rsi.memory import MemoryStore
from llmgo.server.rsi import router
from llmgo.training.ppo import training_config
from llmgo.training.rewards import evaluate_action, terminal_value
from llmgo.training.rollout import board_from_record, load_curriculum, prompt_for_record, record_identity, write_curriculum


def batch(points=.5, **extra):
    games = []
    for pair in range(10):
        for side in ("white", "black"):
            point = points[pair] if isinstance(points, list) else points
            winner = side if point == 1 else ("black" if side == "white" else "white")
            games.append({"game_id": f"g{pair}-{side}", "pair_id": pair, "model_seat": side,
                          "result": "draw" if point == .5 else f"{winner}_win", "result_reason": "checkmate"})
    result = {"games": games, "comparison_id": "same-parent:L1:b0", "opponent_identity": "engine",
              "protocol_identity": "protocol", "seed_schedule": list(range(10)), **extra}
    result["statistics"] = batch_stats(result)
    return result








def test_memory_rebuild_dedup_redaction_and_process_lock(tmp_path):
    memory = MemoryStore(tmp_path)
    first = memory.append("research_note", {"text": "Lost queen", "api_key": "private"}, "one")
    assert first["body"]["api_key"] == "[REDACTED]"
    assert memory.append("research_note", {"text": "Lost queen", "api_key": "other"}, "one") == first
    with pytest.raises(ValueError, match="Conflicting"):
        memory.append("research_note", {"text": "different"}, "one")
    with memory.lock():
        with pytest.raises(RuntimeError, match="already owns"):
            with MemoryStore(tmp_path).lock():
                pass
    memory.db.unlink()
    restored = MemoryStore(tmp_path)
    assert restored.search("QUEEN")[0] == first
    with restored.events.open("a") as stream:
        stream.write('{"torn":')
    with pytest.raises(RuntimeError, match="Damaged"):
        MemoryStore(tmp_path)


def test_source_freezing_detects_edits_and_path_escape(tmp_path):
    repo = tmp_path/"repo"
    (repo/"src").mkdir(parents=True)
    (repo/"src/code.py").write_text("a = 1\n")
    source = source_snapshot(repo, tmp_path/"frozen")
    verify_source(source)
    (tmp_path/"frozen/src/code.py").write_text("a = 2\n")
    with pytest.raises(ValueError, match="Frozen source"):
        verify_source(source)
    with pytest.raises(ValueError, match="outside"):
        inside(tmp_path, tmp_path/"../elsewhere")


@pytest.fixture
def controller(tmp_path):
    repo = tmp_path/"repo"
    cards = repo/"src/llmgo/harness/skills/cards.json"
    cards.parent.mkdir(parents=True)
    shutil.copyfile(REPO/"src/llmgo/harness/skills/cards.json", cards)
    (repo/"scripts").mkdir()
    (repo/"scripts/rsi_worker.py").write_text("# fixture worker\n")
    site = {"user_root": str(tmp_path), "repo": str(repo), "campaign_root": str(tmp_path/"campaigns"),
            "worktree_root": str(tmp_path/"workspaces"), "base_model": str(tmp_path/"model"),
            "engine": "engine", "gpu_count": 1}
    created = Controller.create(site, "test", repo=repo)
    return Controller(created.root, engine_probe=lambda path, spec: {"engine": "fixture", "spec": spec})


def consume_evaluation(c, points=.5):
    job = c.state["jobs"][c.state["active_job"]]
    payload = read_json(job["manifest"])["payload"]
    evidence = batch(points, track=payload["track"], candidate_source_identity=payload["candidate_source"]["sha256"], candidate_id=payload["candidate_id"], skills_sha256=payload["skills_sha256"],
                     source_identity=payload["source"]["sha256"], comparison_id=payload["comparison_id"],
                     protocol_identity=digest(payload["protocol"]), opponent_identity=digest(payload["engine_identity"]),
                     seed_schedule=payload["seed_schedule"], model_identity={"sha256": "model-a"})
    path = Path(payload["output"])/"metrics.json"
    atomic_json(path, evidence)
    atomic_json(path.with_name("result.json"), {"manifest_sha256": job["sha256"], "artifact": str(path),
                                               "artifact_sha256": file_hash(path)})
    return evidence


class Platform:
    def __init__(self):
        self.submissions = set()
        self.calls = 0

    def submit(self, path):
        self.submissions.add(path)
        self.calls += 1
        return {"task_id": len(self.submissions)}

    def get(self, intent):
        return {"status": "Succeeded"}


def skill_proposal(c):
    workspace = c.workspace()
    cards = read_json(workspace/"skills.json")
    cards[0]["rule"] += " Test one concrete hypothesis."
    atomic_json(workspace/"skills.json", cards)
    proposal = workspace/"proposal.json"
    atomic_json(proposal, {"hypothesis": "Improve tactical capture checks", "jobs": [], "evaluation": {"levels":[1],"tracks":["player"]}, "evidence_refs": []})
    c.prepare(proposal)
    return proposal








def test_submission_timeout_reconciles_without_second_create(tmp_path, monkeypatch):
    # Isolate transport tests; real resource/site validation is exercised separately.
    site = {"cctl_bin": "cctl", "project": "rsi"}
    memory = MemoryStore(tmp_path/"memory")
    created = []
    def runner(argv, **kwargs):
        assert kwargs["env"]["CCTL_NO_UPDATE_CHECK"] == "1"
        if argv[1:3] == ["pool", "list"]:
            return subprocess.CompletedProcess(argv, 0, '[{"name":"model-autosft"}]', '')
        if argv[1:3] == ["job", "create"]:
            created.append(argv)
            raise subprocess.TimeoutExpired(argv, 90)
        assert argv[1:3] == ["job", "list"] and "--own" in argv
        return subprocess.CompletedProcess(argv, 0, '[{"id":12,"description":"fixture","status":"Running"}]', '')
    executor = CctlExecutor(site, memory, runner=runner)
    monkeypatch.setattr(executor, "request", lambda p: (["job", "create"], "fixture"))
    manifest = tmp_path/"job/manifest.json"
    with pytest.raises(SubmissionUncertain):
        executor.submit(manifest)
    assert read_json(manifest.with_name("submission.json"))["status"] == "UNKNOWN"
    assert executor.submit(manifest)["task_id"] == 12
    assert executor.submit(manifest)["task_id"] == 12
    assert len(created) == 1


def test_reward_actor_pov_and_no_teacher_answers_in_prompt(tmp_path):
    board = chess.Board()
    for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push_uci(move)
    assert terminal_value(board, chess.BLACK) == 1
    assert terminal_value(board, chess.WHITE) == 0
    record = {"initial_fen": chess.STARTING_FEN, "moves": ["f2f3", "e7e5", "g2g4"], "split": "train"}
    class Critic:
        def value(self, board, actor):
            return .6, {"mock": True}
    judgement = evaluate_action(record, "</think>\nd8h4", Critic())
    assert judgement["environment_reward"] == 1
    assert judgement["bootstrap"] == 0
    assert judgement["advantage"] == pytest.approx(.8)
    assert evaluate_action(record, "</think>\nd8h4", Critic(), truncated=True)["failure_reason"] == "length"
    assert evaluate_action(record, "</think>\na1a8", Critic())["uci"] is None
    prompt = prompt_for_record(record, [])
    assert "d8h4" in prompt["legal_moves"] and "reward" not in prompt["prompt"].lower()
    invalid = {**record, "teacher_answer": "d8h4"}
    with pytest.raises(ValueError, match="teacher answers"):
        write_curriculum(tmp_path/"bad.jsonl", [invalid])
    with pytest.raises(ValueError, match="training-split"):
        write_curriculum(tmp_path/"bad.jsonl", [{**record, "split": "eval"}])


def test_history_is_preserved_and_unknown_training_commands_rejected(tmp_path):
    first = {"initial_fen": chess.STARTING_FEN, "moves": []}
    second = {**first, "moves": ["g1f3", "g8f6", "f3g1", "f6g8"]}
    assert board_from_record(first).board_fen() == board_from_record(second).board_fen()
    assert record_identity(first) != record_identity(second)
    path = tmp_path/"positions.jsonl"
    assert write_curriculum(path, [first, first, second]) == 2
    assert len(load_curriculum(path)) == 2
    with pytest.raises(ValueError, match="Unknown RL"):
        training_config({"sft": True})
    assert training_config({})["backend"] == "hf_ppo"


def test_readonly_dashboard_cursor_raw_artifact_and_path_containment(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMGO_RSI_ROOT", str(tmp_path))
    root = tmp_path/"test"
    atomic_json(root/"campaign.json", {"campaign_id": "test", "phase": "PLANNING"})
    lines = [json.dumps({"kind": "meta_event", "body": {"text": "思考"}}).encode()+b"\n",
             b'{"kind":"evaluation"}\n']
    (root/"events.jsonl").write_bytes(b"".join(lines)+b'{"partial":')
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    assert client.get("/api/rsi/campaigns").json()[0]["campaign_id"] == "test"
    url = "/api/rsi/campaigns/test"
    page = client.get(url+"/events?limit=1").json()
    assert page["cursor"] == len(lines[0]) and len(page["events"]) == 1
    rest = client.get(url+f"/events?cursor={page['cursor']}").json()
    assert len(rest["events"]) == 1 and rest["cursor"] == sum(map(len, lines))
    assert client.get(url+"/events?cursor=1").status_code == 400
    assert client.get(url+"/artifact", params={"path": "../private.json"}).status_code == 404
    assert client.get(url+"/artifact", params={"path": "events.jsonl", "length": 5}).json()["eof"] is False
    assert client.post(url).status_code == 405










def test_panel_is_fixed_balanced_and_partial_scores_cannot_win():
    assert (opponent(1)["skill"],opponent(1)["depth"],opponent(1)["movetime_sec"]) == (-9,5,.05)
    assert opponent(8)["skill"]==20
    with pytest.raises(ValueError):opponent(9)
    p=protocol(opponent(1),"engine")
    validate_protocol(p)
    for change in ({"games":10},{"enable_thinking":False},{"max_plies":201}):
        with pytest.raises(ValueError):validate_protocol({**p,**change})
    good=batch(1)
    assert batch_stats(good)["score"]==1
    bad=copy.deepcopy(good);bad["games"][1]["pair_id"]=2
    with pytest.raises(ValueError,match="pairs"):batch_stats(bad)
    assert panel_summary([{"track":"player","level":1,"statistics":good["statistics"]}])["player"]["score"] is None
    rows=[{"track":"player","level":i,"statistics":batch(.5)["statistics"]} for i in range(1,9)]
    assert panel_summary(rows)["player"]["score"]==.5
    assert panel_summary(rows)["model"]["score"] is None


def complete_baseline(c):
    c.executor=Platform()
    for _ in range(8):
        c.step();consume_evaluation(c,.5);c.step()
    c.step()
    assert c.state["phase"]=="PLANNING"
    assert c.state["best"]["score"]==.5


def test_full_panel_baseline_resume_and_partial_candidate_does_not_replace_best(controller):
    c=controller
    complete_baseline(c)
    assert "current_level" not in c.state
    original=copy.deepcopy(c.state["best"])
    skill_proposal(c)
    c.step();c.step() # no research commands -> EVALUATING -> single L1 job
    consume_evaluation(c,1)
    restored=Controller(c.root,executor=c.executor,engine_probe=c.engine_probe)
    assert restored.step()["status"]=="collected"
    restored.step()
    assert restored.state["best"]==original
    assert restored.state["working"]["id"]!="initial"
    assert restored.state["attempts"][-1]["panel"]["player"]["score"] is None


def test_free_form_training_receipt_routes_new_model_to_fixed_panel(controller):
    c=controller;complete_baseline(c)
    workspace=c.workspace()
    # Arbitrary reward/trainer/helper code is captured, not restricted to preset knobs.
    (workspace/"custom").mkdir();(workspace/"custom/train.py").write_text("print('custom SFT/PPO')")
    path=workspace/"proposal.json"
    atomic_json(path,{"hypothesis":"Explore a different training objective","jobs":[{"argv":["{python}","custom/train.py"]}],
                     "reward":{"anything":"editable"},"evaluation":{"levels":[1,8],"tracks":["player","model"]}})
    c.prepare(path);c.step()
    job=c.state["jobs"][c.state["active_job"]];payload=read_json(job["manifest"])["payload"]
    assert "custom/train.py" in payload["candidate_source"]["files"]
    assert payload["kind"]=="research"
    output=Path(payload["output"]);model=Path(c.site["user_root"])/"exported"
    atomic_json(model/"config.json",{"model_type":"qwen3_5"})
    (model/"model.safetensors").write_bytes(b"fixture weights")
    artifact=output/"research-result.json";atomic_json(artifact,{"model_path":str(model)})
    atomic_json(output/"result.json",{"manifest_sha256":job["sha256"],"artifact":str(artifact),"artifact_sha256":file_hash(artifact)})
    c.step();c.step();c.step()
    evaluation=read_json(c.state["jobs"][c.state["active_job"]]["manifest"])["payload"]
    assert evaluation["model_path"]==str(model)
    assert evaluation["protocol"]["games"]==20 and evaluation["track"]=="player"
    assert "reward" not in evaluation and "command" not in evaluation


def test_evaluation_identity_tamper_does_not_update_best(controller):
    c=controller;complete_baseline(c);skill_proposal(c);c.step();c.step()
    job=c.state["jobs"][c.state["active_job"]];payload=read_json(job["manifest"])["payload"]
    evidence=consume_evaluation(c,1);evidence["candidate_source_identity"]="wrong"
    artifact=Path(payload["output"])/"metrics.json";atomic_json(artifact,evidence)
    ref={"path":str(artifact),"sha256":file_hash(artifact)}
    with pytest.raises(ValueError,match="registered"):c._consume(job,ref,payload)
    assert c.state["best"]["score"]==.5


def test_full_candidate_panel_updates_best_without_mastery_gate(controller):
    c=controller;complete_baseline(c);workspace=c.workspace()
    atomic_json(workspace/"proposal.json",{"hypothesis":"Small broad improvement","jobs":[],"recommend":True})
    c.prepare(workspace/"proposal.json");c.step()
    for _ in range(8):
        c.step();consume_evaluation(c,.5 if _ else 1);c.step()
    c.step()
    assert c.state["best"]["score"]==.5625
    assert c.state["recommended"]["candidate_id"]==c.state["working"]["id"]
    assert c.state["phase"]=="PLANNING" and "current_level" not in c.state


def test_failed_experiment_is_returned_to_research_instead_of_permanent_stop(controller):
    c=controller;complete_baseline(c);workspace=c.workspace()
    atomic_json(workspace/"proposal.json",{"hypothesis":"Failure also informs research","jobs":[{"argv":["false"]}],"evaluation":False})
    c.prepare(workspace/"proposal.json");c.step()
    def fail(job):
        return {"status":"failed","platform_status":"Failed","report_path":"fixture","task_id":123}
    c.lifecycle.advance=fail
    assert c.step()["phase"]=="PLANNING"
    assert c.state["attempts"][-1]["status"]=="failed"
    assert c.state["active_job"] is None


def test_failure_handoff_never_persists_a_finished_candidate_with_active_job(controller):
    c=controller;complete_baseline(c);workspace=c.workspace()
    atomic_json(workspace/"proposal.json",{"hypothesis":"Crash-consistent failure handoff","jobs":[{"argv":["false"]}],"evaluation":False})
    c.prepare(workspace/"proposal.json");c.step()
    c.lifecycle.advance=lambda job:{"status":"failed","platform_status":"Failed","report_path":"fixture","task_id":123}
    c.step()
    states=[x["body"]["state"] for x in c.memory.recent(20,"state")]
    assert not any(s["phase"]=="PLANNING" and s["active_job"] for s in states)
    restored=Controller(c.root,executor=c.executor,engine_probe=c.engine_probe)
    assert restored.step()["status"]=="needs_meta"


def test_engine_is_pinned_across_difficulties_and_candidates(controller):
    c=controller
    c.step()
    assert c.state["engine_pin"]=={"engine":"fixture"}
    c.state["active_job"]=None
    c.engine_probe=lambda path,spec:{"engine":"changed","spec":spec}
    with pytest.raises(ValueError,match="identity changed"):
        c._new_job("evaluate","baseline",c.state["initial"],level=8)
