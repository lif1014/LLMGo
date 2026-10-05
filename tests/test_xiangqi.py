import tempfile
import unittest
from pathlib import Path

from llmgo.env.notation import chinese_notation, parse_model_move, render_board
from llmgo.env.xiangqi import Move, Position
from llmgo.eval.report import aggregate, summarize_attempts
from llmgo.runner.loop import GameRunner
from llmgo.runtime.prompt import Observation, render_user_prompt
from llmgo.runtime.scripted import ScriptedPlayer
from llmgo.store.jsonl import TrajectoryStore


def move(text: str) -> Move:
    parsed = Move.parse(text)
    assert parsed is not None
    return parsed


class XiangqiTests(unittest.TestCase):
    def test_initial_moves_and_notation(self):
        pos = Position.initial()
        self.assertEqual(pos.fen().split()[0], "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR")
        legal = {item.uci for item in pos.legal_moves()}
        self.assertIn("h2e2", legal)
        self.assertIn("b0c2", legal)
        self.assertIn("a3a4", legal)
        self.assertNotIn("a3b3", legal)
        self.assertEqual(chinese_notation(pos, move("h2e2")), "炮二平五")
        self.assertEqual(chinese_notation(pos, move("b2e2")), "炮八平五")
        self.assertEqual(chinese_notation(pos, move("b0c2")), "马八进七")
        text = render_board(pos)
        self.assertIn("楚河", text)
        self.assertIn("汉界", text)
        self.assertIn("（帅）", text)
        self.assertIn("【将】", text)

    def test_geometry(self):
        pos = Position.initial()
        self.assertTrue(pos.classify(move("b0a2")).ok)
        self.assertTrue(pos.classify(move("d0e1")).ok)
        self.assertTrue(pos.classify(move("c0e2")).ok)
        blocked = Position.from_fen("4k4/9/9/9/9/9/9/9/1P7/1N2K4 w - - 0 1")
        self.assertEqual(blocked.classify(move("b0a2")).error_detail, "马腿被挡")
        edge = Position.from_fen("4k4/9/9/9/9/2B6/9/9/9/4K4 w - - 0 1")
        verdict = edge.classify(move("c4a6"))
        self.assertFalse(verdict.ok)
        self.assertIn("过河", verdict.error_detail or "")

    def test_cannon_screen_and_king_face(self):
        pos = Position.initial()
        self.assertTrue(pos.classify(move("b2b9")).ok)
        self.assertEqual(pos.classify(move("b2b7")).error_detail, "炮吃子必须隔恰好一个棋子")
        face = Position.from_fen("4k4/9/9/9/9/4R4/9/9/9/4K4 w - - 0 1")
        verdict = face.classify(move("e4d4"))
        self.assertEqual(verdict.error_type, "king_face")
        self.assertTrue(face.classify(move("e4e8")).ok)

    def test_checkmate_and_repetition(self):
        mate = Position.from_fen("3aka3/R8/3N5/9/9/4P4/9/9/9/4K4 b - - 0 1")
        self.assertTrue(mate.in_check("black"))
        self.assertEqual(mate.legal_moves(), [])
        pos = Position.initial()
        sequence = ["b0c2", "b9c7", "c2b0", "c7b9"] * 2
        result = reason = None
        for uci in sequence:
            self.assertTrue(pos.classify(move(uci)).ok, uci)
            pos, result, reason = pos.apply(move(uci))
        self.assertEqual((result, reason), ("draw", "repetition"))

    def test_parser_and_isolated_prompt(self):
        pos = Position.initial()
        parsed, how = parse_model_move("我想了想\n<move>炮二平五</move>", pos)
        self.assertEqual((parsed.uci if parsed else None, how), ("h2e2", "chinese"))
        parsed, how = parse_model_move("<move>h2e2</move>", pos)
        self.assertEqual(how, "iccs")
        parsed, how = parse_model_move("<move>h2e2</move> 后来决定 <move>b2e2</move>", pos)
        self.assertEqual((parsed.uci if parsed else None, how), ("b2e2", "iccs"))
        red = Observation.from_position(pos, "red", [], "马腿被挡，只有红方看得到", 1, "blind")
        black = Observation.from_position(pos, "black", [], None, 0, "blind")
        red_prompt = render_user_prompt(red)
        black_prompt = render_user_prompt(black)
        self.assertIn("马腿被挡", red_prompt)
        self.assertNotIn("马腿被挡", black_prompt)
        self.assertNotIn("可走的坐标", red_prompt)
        self.assertIn("楚河", red_prompt)


class ReportTests(unittest.TestCase):
    def test_summary_splits_seats(self):
        attempts = [
            {"seat": "red", "attempt_index": 0, "ok": False, "error_type": "illegal_geometry"},
            {"seat": "red", "attempt_index": 1, "ok": True},
            {"seat": "black", "attempt_index": 0, "ok": True},
        ]
        summary = summarize_attempts("g", attempts, None, None)
        self.assertEqual(summary["first_try_legal_rate"], 0.5)
        self.assertEqual(summary["by_seat"]["red"]["first_try_legal"], 0)
        self.assertEqual(summary["error_types"]["illegal_geometry"], 1)
        total = aggregate([summary, summary])
        self.assertEqual(total["games"], 2)
        self.assertEqual(total["first_try_total"], 4)


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_players_do_not_see_each_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TrajectoryStore(Path(tmp))
            runner = GameRunner(
                ScriptedPlayer("red", "illegal_once"),
                ScriptedPlayer("black", "legal"),
                store,
                max_plies=2,
                max_retries=2,
            )
            events = []

            async def emit(event):
                events.append(event)

            summary = await runner.run(emit)
            self.assertGreaterEqual(summary["first_try_total"], 1)
            prompts = [item["prompt"] for item in events if item["type"] == "turn_started"]
            self.assertTrue(any("不合法" not in item and "红方" in item for item in prompts))
            black_prompts = [item["prompt"] for item in events if item["type"] == "turn_started" and item["seat"] == "black"]
            self.assertTrue(black_prompts)
            for prompt in black_prompts:
                self.assertNotIn("只属于红方", prompt)
                self.assertNotIn("先试一个不合法", prompt)
            red_fail = next(item for item in events if item["type"] == "attempt" and item["seat"] == "red" and not item["ok"])
            self.assertEqual(red_fail["error_type"], "unchanged")
            self.assertIn("只属于红方", red_fail["reasoning"])


if __name__ == "__main__":
    unittest.main()
