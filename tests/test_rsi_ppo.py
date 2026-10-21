"""Frozen external critic contracts; no engine/GPU required."""
from types import SimpleNamespace
import chess
import chess.engine
import pytest

from llmgo.training.ppo import training_config
from llmgo.training.rewards import evaluate_action, validate_reward, Lc0Reward


class Critic:
    def __init__(self, before=.4, after=.7):
        self.values = iter((before, after))
        self.calls = []

    def value(self, board, actor):
        self.calls.append((board.copy(stack=True), actor))
        return next(self.values), {"wdl": [600, 200, 200]}


def record(moves=(), initial=chess.STARTING_FEN):
    return {"initial_fen": initial, "moves": list(moves), "split": "train"}


@pytest.mark.parametrize("moves,uci,actor", [((), "e2e4", chess.WHITE), (("e2e4",), "e7e5", chess.BLACK)])
def test_q_delta_uses_same_actor_even_after_turn_changes(moves, uci, actor):
    critic = Critic()
    row = evaluate_action(record(moves), f"thoughts</think>\n{uci}", critic)
    assert row["value_before"] == pytest.approx(-.2)
    assert row["value_after"] == pytest.approx(.4)
    assert row["advantage"] == pytest.approx(.6)  # Q delta, not E delta .3
    assert row["environment_reward"] == 0 and row["bootstrap_mask"] == 1
    assert row["bootstrap"] == pytest.approx(.4)
    assert not row["terminated"] and row["segment_truncated"]
    assert [c[1] for c in critic.calls] == [actor, actor]
    assert critic.calls[1][0].turn != actor
    assert critic.calls[1][0].move_stack[-1].uci() == uci


def test_native_uci_wdl_flips_to_student_pov():
    provider = object.__new__(Lc0Reward)
    provider.config = {"nodes": 1}
    provider.identity = {"uci_options": {}}
    provider.engine = SimpleNamespace(options={}, id={}, analyse=lambda *a, **k: {
        "wdl": chess.engine.PovWdl(chess.engine.Wdl(700, 200, 100), chess.BLACK)})
    board = chess.Board()
    board.push_uci("e2e4")
    e, detail = provider.value(board, chess.WHITE)
    assert e == pytest.approx(.2)
    assert detail["wdl"] == [100, 200, 700]


@pytest.mark.parametrize("moves,uci", [(("f2f3", "e7e5", "g2g4"), "d8h4"),
    (("e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6"), "h5f7")])
def test_checkmate_does_not_double_count_bootstrap(moves, uci):
    critic = Critic(.6)
    row = evaluate_action(record(moves), f"</think>\n{uci}", critic)
    assert row["environment_reward"] == 1
    assert row["value_after"] == 1
    assert row["bootstrap"] == row["bootstrap_mask"] == 0
    assert row["advantage"] == pytest.approx(.8)
    assert len(critic.calls) == 1 and row["terminated"]


def test_stalemate_uses_real_draw():
    row = evaluate_action(record(initial="7k/5K2/6Q1/8/8/8/8/8 w - - 0 1"), "</think>\ng6f5", Critic(.9))
    assert row["environment_reward"] == row["bootstrap"] == 0
    assert row["advantage"] == pytest.approx(-.8)
    assert row["terminated"] and not row["protocol_cap_draw"]


def test_200_ply_cap_is_draw_without_engine_bootstrap():
    import random
    rng = random.Random(7)
    # Reproducible valid history with no terminal repetition/50-move draw before cap.
    for attempt in range(100):
        board = chess.Board()
        while len(board.move_stack) < 199 and not board.outcome(claim_draw=True):
            candidates = []
            for move in board.legal_moves:
                child = board.copy(stack=True)
                child.push(move)
                if not child.outcome(claim_draw=True):
                    candidates.append(move)
            if not candidates:
                break
            captures = [m for m in candidates if board.is_capture(m) or board.piece_at(m.from_square).piece_type == chess.PAWN]
            board.push(rng.choice(captures if board.halfmove_clock > 70 and captures else candidates))
        if len(board.move_stack) == 199 and not board.outcome(claim_draw=True):
            break
    assert len(board.move_stack) == 199
    move = next(iter(board.legal_moves)).uci()
    critic = Critic(.6)
    row = evaluate_action(record([m.uci() for m in board.move_stack]), f"</think>\n{move}", critic)
    assert row["protocol_cap_draw"] and row["terminated"]
    assert row["environment_reward"] == row["bootstrap"] == 0
    assert len(critic.calls) == 1


@pytest.mark.parametrize("fen,uci", [
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
    ("7k/P7/8/8/8/8/8/K7 w - - 0 1", "a7a8q"),
    ("7k/8/8/3pP3/8/8/8/K7 w - d6 0 1", "e5d6")])
def test_bootstrap_evaluates_real_special_move_transition(fen, uci):
    critic = Critic()
    row = evaluate_action(record(initial=fen), f"</think>\n{uci}", critic)
    expected = chess.Board(fen)
    expected.push_uci(uci)
    assert row["after_fen"] == expected.fen()
    assert row["uci"] == uci


@pytest.mark.parametrize("text,truncated,reason", [
    ("</think>\ne2e4", True, "length"), ("thinking e2e4", False, "incomplete_thinking"),
    ("</think>\na1a8", False, "invalid_final_answer")])
def test_protocol_failure_is_not_random_move_or_real_chess_loss(text, truncated, reason):
    critic = Critic(.6)
    row = evaluate_action(record(), text, critic, truncated=truncated)
    assert row["uci"] is row["after_fen"] is None
    assert row["failure_reason"] == reason
    assert row["environment_reward"] == -1 and row["bootstrap"] == 0
    assert row["advantage"] == pytest.approx(-1.2)
    assert len(critic.calls) == 1


def test_bad_critic_or_old_configs_fail_explicitly():
    with pytest.raises(ValueError):
        evaluate_action(record(), "</think>\ne2e4", Critic(float("nan")))
    for cfg in ({"backend": "hf_grpo"}, {"backend": "verl"}, {"group_size": 4}, {"ppo_epochs": 0}):
        with pytest.raises(ValueError):
            training_config(cfg)
    for cfg in ({"mode": "quality"}, {"mode": "delta"}, {"epsilon": .02}):
        with pytest.raises(ValueError):
            validate_reward(cfg)
    assert validate_reward({})["mode"] == "lc0_bootstrap"


def test_dashboard_only_lists_latest_ppo_files_with_safe_pointers(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from llmgo.server.rsi import router
    from llmgo.rsi.contracts import atomic_json
    monkeypatch.setenv("LLMGO_RSI_ROOT", str(tmp_path))
    root = tmp_path/"campaign"
    job = root/"jobs/j000001"
    atomic_json(root/"campaign.json", {"campaign_id": "campaign"})
    atomic_json(job/"manifest.json", {})
    rollout = job/"output/training/rollouts/latest.json"
    atomic_json(rollout, {"samples": ["large hidden raw data"]})
    atomic_json(job/"output/training/latest.json", {"rollout_path": str(rollout),
                "raw_path": str(tmp_path/"outside.json")})
    atomic_json(tmp_path/"outside.json", {"secret": "outside"})
    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).get("/api/rsi/campaigns/campaign/job-files", params={"job_id": "j000001"})
    assert response.status_code == 200
    paths = [f["path"] for f in response.json()["files"]]
    assert "jobs/j000001/output/training/rollouts/latest.json" in paths
    assert not any("outside" in path for path in paths)
    assert "large hidden raw data" not in response.text
