import asyncio
import random
import tempfile
import unittest
from pathlib import Path

import chess

from llmgo.harness.agent import AskResult, choose_move
from llmgo.harness.choices import parse_choice
from llmgo.harness.tools import hanging_line, king_line, legal_lines, material_line, movable_pieces
from llmgo.harness.choices import option_labels
from llmgo.runner.harness_loop import HarnessRunner, HarnessSeat
from llmgo.store.jsonl import TrajectoryStore


def _line(lines: list[str], prefix: str) -> str:
    return next(line for line in lines if line.startswith(prefix))


class Script:
    def __init__(self, answers: list[str], tokens: int = 1, error: str | None = None):
        self.answers = list(answers)
        self.tokens = tokens
        self.error = error
        self.prompts: list[str] = []
        self.caps: list[int] = []

    async def __call__(self, prompt: str, max_tokens: int) -> AskResult:
        self.prompts.append(prompt)
        self.caps.append(max_tokens)
        if self.error:
            return AskResult("", 0, error=self.error)
        text = self.answers.pop(0)
        return AskResult(text, self.tokens, finish_reason="stop")


def _run(board, ask, budget=32768, history=None, rng=None):
    events = []

    async def emit(event):
        events.append(event)

    decision = asyncio.run(
        choose_move(
            board,
            history or [],
            ask,
            emit,
            seat="white",
            budget=budget,
            rng=rng or random.Random(0),
        )
    )
    return decision, events


class ToolTests(unittest.TestCase):
    def test_opening_groups_and_counts(self):
        board = chess.Board()
        lines = legal_lines(board)
        self.assertEqual(_line(lines, "legal: captures"), "legal: captures none")
        self.assertEqual(_line(lines, "legal: checks"), "legal: checks none")
        quiet = _line(lines, "legal: quiet")
        self.assertIn("e2e4", quiet)
        self.assertIn("g1f3", quiet)
        self.assertIn("in_check no", king_line(board)[0])
        self.assertIn("castling_legal_now none", king_line(board)[1])
        self.assertIn("no legal capture threats", hanging_line(board)[1])
        self.assertEqual(
            material_line(board)[0],
            "material: white P8 N2 B2 R2 Q1 K1, black p8 n2 b2 r2 q1 k1",
        )

    def test_capture_that_checks_is_in_both(self):
        board = chess.Board("k7/8/8/8/8/8/p7/R6K w - - 0 1")
        lines = legal_lines(board)
        self.assertIn("a1a2", _line(lines, "legal: captures"))
        self.assertIn("a1a2", _line(lines, "legal: checks"))
        self.assertNotIn("a1a2", _line(lines, "legal: quiet"))

    def test_en_passant_is_a_capture(self):
        board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
        self.assertIn("e5d6", _line(legal_lines(board), "legal: captures"))

    def test_hanging_queen(self):
        board = chess.Board("r7/8/8/8/Q7/8/8/7K w - - 0 1")
        self.assertIn("queen a4; enemy a8a4; legal recaptures none; local material loss +9.00", hanging_line(board)[1])


class PromptTests(unittest.TestCase):
    def test_first_card_explains_every_tool_without_parentheses(self):
        board = chess.Board()
        decision, events = _run(board, Script(["E", "A", "A"]))
        opening = next(e['prompt'] for e in events if e.get('stage') == 'tool' and e.get('status') == 'asking')
        self.assertIn("Side\nWhite", opening)
        self.assertIn("captures means a move that takes an enemy piece, including en passant", opening)
        self.assertIn("checks means a move that attacks the enemy king", opening)
        self.assertIn("quiet means any other legal move", opening)
        self.assertIn("Pinned geometric defenders do not prove safety", opening)
        self.assertIn("Not in check does not mean safe", opening)
        self.assertIn("F preview", opening)
        self.assertIn("you can look up 2 more tools on this turn", opening)
        self.assertIn("choosing E does not use a lookup", opening)
        self.assertIn("reply E", opening)
        self.assertNotIn("(", opening)
        self.assertNotIn(")", opening)
        self.assertNotIn("<move>", opening)
        self.assertNotIn("Result\n", opening)
        self.assertTrue(decision.uci)
        self.assertEqual(decision.source, "model")

    def test_blackboard_keeps_both_tools_and_drops_used_letters(self):
        board = chess.Board()
        pieces = movable_pieces(board)
        script = Script(["B", "C", option_labels(len(pieces))[0], "A"])
        decision, events = _run(board, script)
        self.assertEqual(decision.source, "model")
        self.assertEqual(decision.tools, ["hanging", "king"])
        second = script.prompts[1]
        self.assertIn("you can look up 1 more tools on this turn", second)
        self.assertIn("no legal capture threats", second)
        self.assertNotIn("B hanging", second)
        self.assertIn("A legal", second)
        self.assertIn("C king", second)
        third = script.prompts[2]
        self.assertIn("no legal capture threats", third)
        self.assertLess(third.index("hanging:"), third.index("king:"))
        self.assertNotIn("A legal", third)
        self.assertNotIn("D material", third)
        self.assertIn("Pick", third)
        self.assertTrue(any(e.get('tool') == 'piece' and e.get('source') == 'automatic' for e in events))
        self.assertNotIn(script.prompts[0], third)

    def test_direct_piece_plays_e2e4(self):
        board = chess.Board()
        pieces = movable_pieces(board)
        labels = option_labels(len(pieces))
        piece_index = next(index for index, item in enumerate(pieces) if chess.square_name(item[0]) == "e2")
        moves = pieces[piece_index][2]
        move_index = next(index for index, move in enumerate(moves) if move.uci() == "e2e4")
        script = Script(["E", labels[piece_index], option_labels(len(moves))[move_index]])
        decision, events = _run(board, script)
        self.assertEqual((decision.uci, decision.source), ("e2e4", "model"))
        move_event = next(event for event in events if event.get("stage") == "move" and event.get("status") == "asking")
        self.assertEqual(move_event["origins"], ["e2"])
        self.assertIn("e4", move_event["targets"])
        self.assertIn("e3", move_event["targets"])

    def test_bad_letter_twice_plays_a_random_legal_move(self):
        board = chess.Board()
        legal = {move.uci() for move in board.legal_moves}
        decision, _events = _run(board, Script(["Z", "no"]), rng=random.Random(1))
        self.assertEqual(decision.source, "random")
        self.assertIn(decision.uci, legal)

    def test_failed_move_letter_stays_on_that_piece(self):
        board = chess.Board()
        pieces = movable_pieces(board)
        labels = option_labels(len(pieces))
        piece_index = next(index for index, item in enumerate(pieces) if chess.square_name(item[0]) == "e2")
        allowed = {move.uci() for move in pieces[piece_index][2]}
        script = Script(["E", labels[piece_index], "nope", "still no"])
        decision, _events = _run(board, script, rng=random.Random(2))
        self.assertEqual(decision.source, "random")
        self.assertIn(decision.uci, allowed)

    def test_request_error_and_empty_budget_do_not_ask_forever(self):
        board = chess.Board()
        legal = {move.uci() for move in board.legal_moves}
        broken = Script([], error="down")
        decision, _events = _run(board, broken, rng=random.Random(3))
        self.assertEqual(decision.source, "random")
        self.assertIn(decision.uci, legal)
        self.assertEqual(len(broken.prompts), 2)
        silent = Script(["E"])
        decision, _events = _run(board, silent, budget=0, rng=random.Random(4))
        self.assertEqual(decision.source, "random")
        self.assertEqual(silent.prompts, [])

    def test_parse_one_letter(self):
        allowed = {"A", "B", "C", "D", "E"}
        self.assertEqual(parse_choice("E", allowed), "E")
        self.assertEqual(parse_choice("reply A", allowed), "A")
        self.assertIsNone(parse_choice("A or B", allowed))
        self.assertIsNone(parse_choice("A\nB", allowed))


class LoopTests(unittest.TestCase):
    def test_two_plies_record_choices(self):
        async def ask(prompt: str, max_tokens: int) -> AskResult:
            if "\nPick\n" in prompt or "\nMoves\n" in prompt:
                return AskResult("A", 1)
            return AskResult("E", 1)

        async def play():
            with tempfile.TemporaryDirectory() as folder:
                store = TrajectoryStore(Path(folder) / "game")
                runner = HarnessRunner(
                    HarnessSeat("white", ask),
                    HarnessSeat("black", ask),
                    store,
                    max_plies=2,
                )
                events = []

                async def emit(event):
                    events.append(event)

                summary = await runner.run(emit)
                return summary, events

        summary, events = asyncio.run(play())
        self.assertEqual(summary["result_reason"], "truncated")
        self.assertEqual(summary["first_try_legal"], 2)
        self.assertEqual(summary["random_moves"], 0)
        self.assertTrue(any(event["type"] == "choice" and event["stage"] == "piece" for event in events))
        self.assertEqual(sum(1 for event in events if event["type"] == "position" and event.get("last_move")), 2)


class ServerHarnessTests(unittest.TestCase):
    def test_xiangqi_is_rejected(self):
        import httpx
        from llmgo.server.app import app
        async def probe():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                return await client.post("/api/games", json={"backend": "harness", "game": "xiangqi", "max_plies": 1})
        self.assertEqual(asyncio.run(probe()).status_code, 400)

    def test_chess_game_continues_when_vllm_is_down(self):
        import httpx
        from unittest.mock import patch
        import llmgo.server.app as server
        async def down(*args, **kwargs):
            return AskResult("", 0, error="test connection failure")
        async def probe():
            with tempfile.TemporaryDirectory() as folder, patch.object(server, 'RUNS', Path(folder)), patch('llmgo.harness.player.VllmAsker.__call__', down):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
                    response = await client.post('/api/games', json={'backend':'harness','game':'chess','opponent':'model','max_plies':2,'max_tokens':64})
                    self.assertEqual(response.status_code, 200, response.text)
                    game = server.hub.games[response.json()['id']]
                    await game.task
                    data = (await client.get(f'/api/games/{game.id}/events')).json()
                    self.assertEqual(game.summary['random_moves'], 2)
                    self.assertGreater(game.summary['request_errors'], 0)
                    self.assertTrue(any(e['type'] == 'model_request' and e['status'] == 'finished' for e in data['events']))
                    self.assertTrue(any(e['type'] == 'fallback' for e in data['events']))
                    self.assertEqual((await client.get('/api/harness')).status_code, 200)
        asyncio.run(probe())


class EvalScriptTests(unittest.TestCase):
    def test_script_refuses_without_gpu_or_url(self):
        import importlib.util
        import os
        import subprocess
        import sys

        script = Path(__file__).resolve().parents[1] / "scripts" / "eval_harness.py"
        loaded = importlib.util.spec_from_file_location("eval_harness", script)
        module = importlib.util.module_from_spec(loaded)
        assert loaded.loader is not None
        loaded.loader.exec_module(module)
        if module.has_gpu():
            self.skipTest("this machine has a GPU")
        completed = subprocess.run(
            [sys.executable, str(script), "--protocol", str(script.parents[1].parent / "PostTrainBench" / "protocol.json"), "--output", str(Path(tempfile.gettempdir()) / "llmgo-test-probe" / "metrics.json")],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        )
        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertTrue("沒有 GPU" in completed.stderr or "引擎不能執行" in completed.stderr)

    def test_eight_gpus_cover_ten_games(self):
        import importlib.util

        script = Path(__file__).resolve().parents[1] / "scripts" / "eval_harness.py"
        loaded = importlib.util.spec_from_file_location("eval_harness_jobs", script)
        module = importlib.util.module_from_spec(loaded)
        assert loaded.loader is not None
        loaded.loader.exec_module(module)
        ids = module.choose_gpu_ids([str(i) for i in range(8)], games=10, requested=None)
        self.assertEqual(ids, [str(i) for i in range(8)])
        seats = ["white"] * 5 + ["black"] * 5
        urls = [f"http://127.0.0.1:{8000 + i}/v1" for i in range(len(ids))]
        jobs = module.assign_jobs(seats, urls)
        counts = {}
        for job in jobs:
            counts[job["url"]] = counts.get(job["url"], 0) + 1
        self.assertEqual(sorted(counts.values()), [1, 1, 1, 1, 1, 1, 2, 2])
        self.assertEqual(module.parse_gpu_list("GPU 0: A100\nGPU 3: A100\n"), ["0", "3"])
        self.assertEqual(module.choose_gpu_ids(ids, games=2, requested=None), ["0", "1"])
        with self.assertRaises(ValueError):
            module.choose_gpu_ids(["0", "1"], games=10, requested=8)


if __name__ == "__main__":
    unittest.main()
