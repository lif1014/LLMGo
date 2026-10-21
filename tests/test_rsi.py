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
from llmgo.rsi.ladder import batch_stats, compare, mastered, opponent
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


def test_twenty_games_balanced_pairs_and_no_health_veto():
    good = batch([1]*8 + [.5]*2)
    good["random_rate"] = 1  # diagnostic only; explicitly not a promotion gate
    assert mastered(good)
    assert not mastered(batch([1]*7 + [.5]*3))
    bad = copy.deepcopy(good)
    bad["games"][1]["pair_id"] = 2
    with pytest.raises(ValueError, match="pairs"):
        batch_stats(bad)
    for count in (19, 21):
        with pytest.raises(ValueError, match="20"):
            batch_stats({"games": (good["games"]*2)[:count]})
    bad = copy.deepcopy(good)
    bad["games"][0]["result_reason"] = "engine_error"
    with pytest.raises(ValueError, match="Infrastructure"):
        batch_stats(bad)


def test_evidence_improvement_rejection_uncertainty_and_identity():
    assert compare([batch(1)], [batch(.5)])["verdict"] == "accept"
    assert compare([batch(.5)], [batch(.5)])["verdict"] == "reject"
    assert compare([batch(0)], [batch(.5)])["verdict"] == "reject"
    assert compare([batch([1]+[.5]*9)], [batch(.5)])["verdict"] == "inconclusive"
    with pytest.raises(ValueError, match="protocol_identity"):
        compare([batch(1, protocol_identity="changed")], [batch(.5)])


def test_ladder_and_formal_protocol():
    assert (opponent(1)["skill"], opponent(1)["depth"], opponent(1)["movetime_sec"]) == (-9, 5, .05)
    assert opponent(8)["skill"] == 20
    with pytest.raises(ValueError):
        opponent(9)
    spec = opponent(10, nodes_base=1000)
    assert (spec["nodes"], spec["depth"], spec["movetime_sec"]) == (2000, None, None)
    p = protocol(opponent(1), "engine")
    validate_protocol(p)
    for key, value in (("games", 10), ("max_plies", 201), ("enable_thinking", False)):
        with pytest.raises(ValueError):
            validate_protocol({**p, key: value})


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
    evidence = batch(points, candidate_id=payload["candidate_id"], skills_sha256=payload["skills_sha256"],
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
    atomic_json(proposal, {"mode": "skills", "hypothesis": "Improve tactical capture checks",
                          "evidence_refs": []})
    c.prepare(proposal)
    return proposal


def test_controller_crash_replay_skill_candidate_acceptance_and_separate_promotion(controller):
    c = controller
    platform = Platform()
    c.executor = platform
    c.step()  # prepare baseline, no submission yet
    assert c.state["phase"] == "WAITING" and platform.calls == 0
    consume_evaluation(c, .5)
    # Restore after a simulated Controller crash before collecting the completed task.
    c = Controller(c.root, executor=platform, engine_probe=c.engine_probe)
    assert c.step()["status"] == "collected"
    c.step()
    assert c.state["phase"] == "PLANNING"
    original = c.state["incumbent"].copy()
    skill_proposal(c)
    assert c.state["incumbent"] == original  # proposal alone never replaces incumbent
    c.step()
    consume_evaluation(c, 1)
    c.step()
    c.step()
    assert c.state["incumbent"]["id"] != "initial"
    assert c.state["current_level"] == 1  # accepting an improvement != promoting opponent
    c.step()  # first mastering batch only
    assert c.state["phase"] == "BASELINE" and c.state["current_level"] == 1
    c.step()
    consume_evaluation(c, 1)
    c.step()
    c.step()
    assert c.state["current_level"] == 2 and len(c.state["level_certificates"]) == 1
    assert len(platform.submissions) == 3


def test_frozen_candidate_and_result_identity(controller):
    c = controller
    c.step()
    consume_evaluation(c)
    c.executor = Platform()
    c.step()
    c.step()
    skill_proposal(c)
    c.step()
    evidence = consume_evaluation(c, 1)
    job = c.state["jobs"][c.state["active_job"]]
    output = Path(read_json(job["manifest"])["payload"]["output"])
    evidence["skills_sha256"] = "wrong-candidate"
    atomic_json(output/"metrics.json", evidence)
    result = read_json(output/"result.json")
    result["artifact_sha256"] = file_hash(output/"metrics.json")
    atomic_json(output/"result.json", result)
    with pytest.raises(ValueError, match="frozen manifest"):
        c.step()
    assert c.state["incumbent"]["id"] == "initial"


def test_old_level_regression_prevents_replacement(controller):
    c = controller
    c.state.update(phase="PLANNING", highest_cleared_level=1)
    c.state["incumbent"]["model_sha256"] = "model-a"
    c.save()
    skill_proposal(c)
    folder = c.root/"regression"
    refs = []
    for i, score in enumerate((1, .5)):
        path = folder/f"{i}.json"
        atomic_json(path, batch(score))
        refs.append({"path": str(path), "sha256": file_hash(path)})
    c.state["candidate"].update(regression_baseline=refs[0], regression_candidate=refs[1],
                                 pending_decision={"verdict": "accept", "gain": .2})
    c.state["phase"] = "REGRESSION_REVIEW"
    c.save()
    c.step()
    assert c.state["incumbent"]["id"] == "initial"
    assert c.state["attempts"][-1]["decision"]["reason"] == "old_level_regression"


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


def test_unrated_difficulty_report_has_wdl_without_elo():
    from llmgo.eval.elo import summarize_games
    report = summarize_games([{"anchor_elo": None, "model_seat": "black", "result": "black_win"}])
    assert report["finished"] == 1 and report["by_anchor"] == []
    assert report["unrated"]["wins"] == 1


def test_controller_ppo_manifest_then_exported_student_enters_formal_twenty_games(controller):
    c = controller
    c.executor = Platform()
    c.step()
    consume_evaluation(c, .5)
    c.step()
    c.step()
    assert c.state["phase"] == "PLANNING"
    parent = copy.deepcopy(c.state["incumbent"])
    workspace = c.workspace()
    proposal = workspace/"proposal.json"
    atomic_json(proposal, {"mode": "rl", "hypothesis": "Reduce tactical loss with frozen Lc0 TD0",
                          "evidence_refs": [], "training": {"ppo_epochs": 2}})
    c.prepare(proposal)
    assert c.state["candidate"]["training"]["backend"] == "hf_ppo"
    assert c.state["candidate"]["reward"]["mode"] == "lc0_bootstrap"
    c.step()
    job = c.state["jobs"][c.state["active_job"]]
    payload = read_json(job["manifest"])["payload"]
    assert payload["kind"] == job["role"] == "train"
    assert payload["training"]["ppo_epochs"] == 2
    assert file_hash(payload["curriculum_path"]) == payload["curriculum_sha256"]
    assert payload["expected_model_sha256"] == parent["model_sha256"]
    assert "labels" not in payload and "group_size" not in payload["training"]
    # A fixture export only proves lifecycle/identity routing, never Qwen strength.
    exported = Path(c.site["user_root"])/"exported-student"
    artifact = Path(payload["output"])/"checkpoint-ref.json"
    atomic_json(artifact, {"model_path": str(exported), "identity": {"sha256": "ppo-updated"},
                          "training": payload["training"], "parent_identity": {"sha256": payload["expected_model_sha256"]}, "sft_enabled": False})
    atomic_json(artifact.with_name("result.json"), {"manifest_sha256": job["sha256"],
                 "artifact": str(artifact), "artifact_sha256": file_hash(artifact)})
    c.step()
    assert c.state["phase"] == "CANDIDATE_EVAL" and c.state["incumbent"] == parent
    c.step()
    evaluation = read_json(c.state["jobs"][c.state["active_job"]]["manifest"])["payload"]
    assert evaluation["kind"] == "evaluate"
    assert evaluation["model_path"] == str(exported)
    assert evaluation["expected_model_sha256"] == "ppo-updated"
    assert evaluation["protocol"]["games"] == 20
    assert evaluation["protocol"]["enable_thinking"] is True
    assert evaluation["protocol"]["max_plies"] == 200
    assert "reward" not in evaluation and "training" not in evaluation


def test_repeated_failed_training_retry_preserves_earlier_resume_root(controller):
    c = controller
    c.state.update(phase="PLANNING")
    c.state["incumbent"]["model_sha256"] = "parent"
    c.save()
    workspace = c.workspace()
    proposal = workspace/"proposal.json"
    atomic_json(proposal, {"mode": "rl", "hypothesis": "Test real checkpoint continuity", "evidence_refs": []})
    c.prepare(proposal)
    c.step()
    def active_payload():
        return read_json(c.state["jobs"][c.state["active_job"]]["manifest"])["payload"]
    first = active_payload()
    original = Path(first["output"])/"training"
    atomic_json(original/"checkpoint-000004/state.json", {"fixture": True})
    c.lifecycle.confirm_failed = lambda job: None  # no cluster calls; test resume selection only
    def retry():
        c.state["phase"] = "FAILED"
        c.save()
        c.resume_failed()
        return active_payload()
    second = retry()
    assert second["resume_from"] == str(original)
    third = retry()  # second never got far enough to write any checkpoint
    assert third["resume_from"] == str(original)
    newer = Path(third["output"])/"training"
    atomic_json(newer/"checkpoint-000008/state.json", {"fixture": True})
    fourth = retry()
    assert fourth["resume_from"] == str(newer)


def test_training_receipt_must_match_parent_and_ppo_before_candidate_evaluation(controller):
    c = controller
    c.state.update(phase="PLANNING")
    c.state["incumbent"]["model_sha256"] = "parent"
    c.save()
    workspace = c.workspace()
    proposal = workspace/"proposal.json"
    atomic_json(proposal, {"mode": "rl", "hypothesis": "Validate frozen PPO training receipt", "evidence_refs": []})
    c.prepare(proposal)
    c.step()
    job = c.state["jobs"][c.state["active_job"]]
    payload = read_json(job["manifest"])["payload"]
    valid = {"model_path": str(Path(payload["output"])/"training/model"), "identity": {"sha256": "new"},
             "parent_identity": {"sha256": "parent"}, "training": payload["training"], "sft_enabled": False}
    path = Path(payload["output"])/"checkpoint-ref.json"
    for bad in ({"sft_enabled": True}, {"training": {"backend": "hf_grpo"}},
                {"parent_identity": {"sha256": "wrong"}}, {"identity": {"sha256": "parent"}}):
        atomic_json(path, {**valid, **bad})
        with pytest.raises(ValueError, match="Training result"):
            c._consume(job, {"path": str(path), "sha256": file_hash(path)}, payload)
        assert c.state["phase"] == "WAITING" and c.state["incumbent"]["id"] == "initial"
