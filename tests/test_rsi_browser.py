"""Opt-in real Chromium check; all data is synthetic, with no models or cluster calls.

LLMGO_BROWSER_TEST=1 PLAYWRIGHT_BROWSERS_PATH=/tmp/llmgo-playwright \
  PYTHONPATH=src python -m pytest -q tests/test_rsi_browser.py
"""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import pytest
import chess

from llmgo.rsi.contracts import atomic_json
from llmgo.rsi.memory import MemoryStore
from llmgo.store.jsonl import TrajectoryStore


@pytest.mark.skipif(os.environ.get("LLMGO_BROWSER_TEST") != "1", reason="opt-in Chromium integration")
def test_real_dashboard_streaming_history_artifacts_and_mobile(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    root = tmp_path/"campaigns/demo"
    memory = MemoryStore(root)
    job = root/"jobs/j1"
    atomic_json(root/"campaign.json", {"campaign_id": "demo", "phase": "PLANNING", "round": 1, "current_level": 1,
        "highest_cleared_level": 0, "incumbent": {"id": "initial"}, "attempts": [], "active_job": "j1",
        "jobs": {"j1": {"id": "j1", "role": "candidate", "status": "Running", "candidate_id": "candidate"}}})
    atomic_json(root/"meta-session.json", {"status": "running", "thread_id": "thread", "turn_id": "turn", "cwd": "/user/lyh4170/research"})
    for i in range(110):
        memory.append("meta_event", {"message": {"method": "item/agentMessage/delta", "params": {
            "threadId": "thread", "turnId": "turn", "itemId": f"m{i}", "delta": f"研究可见片段 {i}"}}})
    executor = MemoryStore(root/"executor")
    executor.append("poll_started", {"job_id": "j1", "checks": 3, "task_id": 4170})
    executor.append("command_started", {"argv": ["cctl", "job", "get", "4170"], "operation_id": "op", "timeout_sec": 90})
    executor.append("command", {"argv": ["cctl", "job", "get", "4170"], "operation_id": "op", "returncode": 0, "seconds": 1.2, "stdout": "platform full output\n"*50000})
    executor.append("poll_finished", {"job_id": "j1", "checks": 3, "task_id": 4170, "platform_status": "Running", "status": "waiting", "interval_seconds": 60})
    atomic_json(job/"manifest.json", {"payload": "synthetic"})
    atomic_json(job/"executor/report.json", {"job_id": "j1", "task_id": 4170, "status": "waiting", "checks": 3,
                                           "interval_seconds": 60, "next_check_at": time.time()+60, "platform_status": "Running"})
    atomic_json(job/"output/progress.json", {"stage": "training", "message": "训练阶段测试", "curriculum_records": 64})
    game = job/"output/harness_runs/g"
    store = TrajectoryStore(game)
    store.write_config({"id": "g", "model_seat": "white"})
    board = chess.Board()
    store.append({"type": "position", "fen": board.fen(), "side": "white"})
    store.append({"type": "turn_started", "seat": "white"})
    store.append({"type": "model_request", "status": "started", "seat": "white", "prompt": "第一手提示", "request_id": "first"})
    store.append({"type": "model_request", "status": "finished", "seat": "white", "reasoning": "第一手思考", "text": "e2e4", "request_id": "first"})
    for side, uci in (("white", "e2e4"), ("black", "e7e5")):
        board.push_uci(uci)
        store.append({"type": "position", "fen": board.fen(), "side": "black" if side == "white" else "white",
                      "last_move": {"seat": side, "uci": uci}})
    store.append({"type": "turn_started", "seat": "white"})
    store.append({"type": "request", "status": "started", "seat": "white", "prompt": "第二手提示", "request_id": "second"})
    store.append({"type": "token", "channel": "reasoning", "text": "学生可见思考", "seat": "white"})
    store.append({"type": "candidate_event", "seat": "white", "origin": "candidate",
                  "payload": {"type": "token", "channel": "reasoning", "text": "学生可见思考"}})
    history = TrajectoryStore(root/"jobs/j0/output/harness_runs/old")
    history.write_config({"id": "old", "model_seat": "black"})
    history.append({"type": "position", "fen": chess.STARTING_FEN, "side": "white"})
    history.write_summary({"result": "draw", "result_reason": "truncated", "model_decisions": 1})
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "LLMGO_RSI_ROOT": str(root.parent), "LLMGO_RUNS": str(tmp_path/"runs")}
    repo = Path(__file__).resolve().parents[1]
    with (tmp_path/"server.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "llmgo.server.app:app", "--host", "127.0.0.1", "--port", str(port)], cwd=repo, env=env, stdout=log, stderr=log)
        try:
            base = f"http://127.0.0.1:{port}"
            for _ in range(120):
                try:
                    with urlopen(base+"/api/rsi/campaigns", timeout=.3):
                        break
                except OSError:
                    if process.poll() is not None:
                        pytest.fail((tmp_path/"server.log").read_text())
                    time.sleep(.05)
            with playwright.sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1440, "height": 1100})
                errors, requests = [], []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.on("request", lambda r: requests.append(r.url))
                page.goto(base+"/rsi")
                page.wait_for_function("document.querySelector('#board').children.length===64")
                page.wait_for_function("document.querySelector('#coach').children.length===80")
                assert "training" in page.locator("#worker-state").inner_text()
                assert "4170" in page.locator("#executor-state").inner_text()
                assert "学生可见思考" in page.locator("#thinking").inner_text()
                page.wait_for_function("document.querySelector('#moves').children.length===3")
                assert page.locator("#student-prompt").inner_text() == "第二手提示"
                layout = page.locator(".game-layout").evaluate("e=>getComputedStyle(e).gridTemplateColumns.split(' ').length")
                assert layout == 3
                page.locator("#replay-prev").click()
                assert "1 / 2" in page.locator("#replay-label").inner_text()
                assert "第一手思考" in page.locator("#thinking").inner_text()
                assert page.locator("#student-prompt").inner_text() == "第一手提示"
                assert page.locator("#student-output").inner_text() == "e2e4"
                assert page.locator('#board [title="e4"]').inner_text() == "♙"
                # An actual new raw chunk must not replace the selected historical hand.
                store.append({"type": "token", "channel": "reasoning", "text": "新增实时思考", "seat": "white"})
                page.wait_for_timeout(3300)
                assert "第一手思考" in page.locator("#thinking").inner_text()
                assert "1 / 2" in page.locator("#replay-label").inner_text()
                page.locator("#follow-game").click()
                page.wait_for_function("document.querySelector('#thinking').textContent.includes('新增实时思考')")
                assert page.locator("#thinking").inner_text() == "学生可见思考新增实时思考"
                page.locator("#replay-restart").click()
                page.wait_for_function("document.querySelector('#replay-label').textContent.includes('复盘 · 2 / 2')")
                page.wait_for_function("document.querySelector('#replay-play').textContent==='播放'")
                page.locator("#moves button").first.click()
                assert page.locator("#student-prompt").inner_text() == ""
                page.locator("#replay-range").fill("1")
                assert "第一手思考" in page.locator("#thinking").inner_text()
                page.locator("#game").select_option("jobs/j0/output/harness_runs/old")
                page.wait_for_function("document.querySelector('#game-summary').textContent.includes('对局：old')")
                assert page.locator("#thinking").inner_text() == ""
                assert "黑方" in page.locator("#student-seat").inner_text()
                assert page.locator("#replay-next").is_disabled()
                page.locator("#game").select_option("jobs/j1/output/harness_runs/g")
                page.wait_for_function("document.querySelector('#thinking').textContent.includes('新增实时思考')")
                page.locator(".game-view").screenshot(path=str(tmp_path/"rsi-replay-desktop.png"))
                assert page.locator("#coach details pre").first.inner_text() == ""
                # Trigger a genuine incremental protocol event after page load.
                memory.append("meta_event", {"message": {"method": "item/agentMessage/delta", "params": {
                    "threadId": "thread", "turnId": "turn", "itemId": "new", "delta": "新增强Agent文字"}}})
                page.wait_for_function("document.querySelector('#coach-live').textContent.includes('新增强Agent文字')")
                page.locator("#coach-filter").select_option("tool")
                assert page.locator("#coach details:visible").count() == 0
                page.locator("#coach-filter").select_option("all")
                page.locator("#coach-older").click()
                page.wait_for_function("document.querySelector('#coach-window').textContent.includes('历史页')")
                page.locator("#coach-live-button").click()
                page.wait_for_function("document.querySelector('#coach-window').textContent.startsWith('实时')")
                # A huge command output remains lazy until explicitly requested.
                command = page.locator("#executor-events details").filter(has_text="命令返回").first
                command.locator("summary").click()
                command.get_by_role("button", name="读取原始事件").click()
                page.wait_for_function("document.querySelector('#raw-content').textContent.length>100")
                assert len(page.locator("#raw-content").inner_text().encode()) <= 65536
                assert page.locator("#raw-next").is_enabled()
                page.locator("#raw-next").click()
                page.wait_for_function("document.querySelector('#raw-position').textContent.includes('65536')")
                page.locator("#close-raw").click()
                page.locator("#pause").click()
                before = len(requests)
                page.wait_for_timeout(3300)
                assert len(requests) == before
                assert not any("/executor-events?" in r or "/events?" in r for r in requests)
                page.set_viewport_size({"width": 390, "height": 844})
                assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth+1")
                page.screenshot(path=str(tmp_path/"rsi-mobile.png"), full_page=True)
                assert not errors, errors
                browser.close()
        finally:
            process.terminate()  # only our local test server, never a cluster job
            process.wait(timeout=10)
