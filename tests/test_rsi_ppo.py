"""LC0 POV and dashboard contracts; terminal rewards tested in test_game_gae."""
from types import SimpleNamespace
import chess
import chess.engine
import pytest
from llmgo.training.lc0 import Lc0Value, teacher_config


def test_native_uci_wdl_flips_to_student_pov():
    provider = object.__new__(Lc0Value)
    provider.config = {"nodes": 1}
    provider.identity = {"uci_options": {}}
    provider.engine = SimpleNamespace(options={}, id={}, analyse=lambda *a, **k: {
        "wdl": chess.engine.PovWdl(chess.engine.Wdl(700, 200, 100), chess.BLACK)})
    board = chess.Board()
    board.push_uci("e2e4")
    e, detail = provider.value(board, chess.WHITE)
    assert e == pytest.approx(-.6)
    assert detail["wdl"] == [100, 200, 700]


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


def test_teacher_rejects_old_reward_recipes():
    assert teacher_config({})['mode']=='lc0_value'
    for config in ({'mode':'lc0_bootstrap'},{'mode':'delta'},{'epsilon':.1}):
        with pytest.raises(ValueError):teacher_config(config)
