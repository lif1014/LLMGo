"""Serving diagnostic regression: separated reasoning is never a final move."""
import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from llmgo.harness.agent import AskResult
from llmgo.harness.player import VllmAsker


spec = importlib.util.spec_from_file_location(
    "graph_ab", Path(__file__).resolve().parents[1]/"scripts/graph_ab.py")
graph_ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_ab)


@pytest.mark.parametrize("message,expected,boundary", [
    ({"reasoning_content": "Consider d2d4 before selecting e2e4.",
      "content": '{"move":"e2e4"}'}, "e2e4", True),
    ({"reasoning": "Choose a legal move.", "content": "e2e4"}, "e2e4", True),
    ({"content": 'Consider d2d4. </think> {"move":"e2e4"}'}, "e2e4", True),
    ({"content": "e2e4"}, "e2e4", False),
    ({"reasoning_content": "e2e4", "content": ""}, None, False),
    ({"reasoning_content": "e2e4", "content": '{"move":"a1a8"}'}, None, True),
    ({"content": "<think>e2e4"}, None, False),
    ({"reasoning_content": "e2e4", "content": '{"move":"e2e4","move":"a1a8"}'}, None, True),
])
def test_check_uses_normalized_final_from_real_http_adapter(message, expected, boundary):
    async def run():
        asker = VllmAsker("http://fixture/v1", "fixture", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [
                {"message": message, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 5}})))
        return await asker("Choose a move", 100)
    answer = asyncio.run(run())
    checked = graph_ab.check_answer(answer, {"e2e4"})
    assert checked["uci"] == expected
    assert checked["protocol_ok"] is (expected is not None)
    assert checked["has_boundary"] is boundary
    if message.get("reasoning_content") or message.get("reasoning"):
        assert not checked["has_think_tag"]


@pytest.mark.parametrize("diagnostics,reason", [
    ({"error": "request_timeout"}, "request_error"),
    ({"normalization_error": "incomplete_thinking"}, "incomplete_thinking"),
    ({"finish_reason": "length"}, "length_without_choice"),
])
def test_valid_looking_move_does_not_override_failed_or_truncated_request(diagnostics, reason):
    checked = graph_ab.check_answer(AskResult("e2e4", 2, **diagnostics), {"e2e4"})
    assert not checked["protocol_ok"] and checked["uci"] is None
    assert checked["failure_reason"] == reason


def test_run_mode_retains_raw_reply_and_recheckable_inputs(tmp_path, monkeypatch):
    import llmgo.harness.player as player
    stopped = []

    class Process:
        def terminate(self): stopped.append("terminate")
        def wait(self, **kwargs): return 0

    def server(python, model, name, port, log, *args):
        return Process(), log.open("w"), ["fixture-server"]

    def responder(request):
        return httpx.Response(200, json={"choices": [{"message": {
            "reasoning_content": "Consider d2d4. Select e2e4.", "content": '{"move":"e2e4"}'},
            "finish_reason": "stop", "token_ids": [1, 2, 3]}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3}})

    monkeypatch.setattr(graph_ab, "build_server", server)
    monkeypatch.setattr(graph_ab, "wait_ready", lambda *args: True)
    monkeypatch.setattr(player, "VllmAsker", lambda *args, **kwargs:
        VllmAsker(*args, **kwargs, transport=httpx.MockTransport(responder)))
    protocol = {"temperature": .9, "top_p": .95, "top_k": 20, "presence_penalty": 0,
                "seed": 4170, "max_tokens": 100, "harness": {"request_timeout_sec": 5}}
    prompts = [{"id": "a", "prompt": "Choose e2e4", "legal_moves": ["e2e4"]},
               {"id": "b", "prompt": "Choose a legal move", "legal_moves": ["d2d4"]}]
    row = asyncio.run(graph_ab.run_mode("graph", "fixture-model", "fixture", "fixture-python",
                                       "0", tmp_path, prompts, protocol))
    assert row["status"] == "ok" and row["protocol_ok"] == 1 and row["boundary_ok"] == 2
    assert row["protocol_check"] == "normalized-final-v1" and stopped == ["terminate"]
    for item, request in zip(prompts, row["requests"]):
        saved = json.loads(Path(request["response_path"]).read_text())
        answer = saved["answer"]
        assert saved["prompt"] == item["prompt"] and saved["legal_moves"] == item["legal_moves"]
        assert saved["check"]["protocol_ok"] == request["protocol_ok"]
        assert json.loads(answer["raw_response"])["choices"][0]["message"]["content"] == answer["text"]
        assert answer["reasoning"] == "Consider d2d4. Select e2e4."
        assert answer["token_ids"] == [1, 2, 3] and answer["tokens"] == 3
        assert answer["finish_reason"] == "stop" and answer["http_status"] == 200
        assert answer["request_payload"]["chat_template_kwargs"]["enable_thinking"] is True
        assert answer["latency_ms"] >= 0
