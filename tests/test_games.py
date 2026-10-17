import unittest

import chess

from llmgo.env.chess_game import CHESS, Move as ChessMove, Position as ChessPosition
from llmgo.env.games import get_game
from llmgo.env.gomoku import GOMOKU, Move as GomokuMove


class GomokuTests(unittest.TestCase):
    def test_five_and_occupied(self):
        pos = GOMOKU.initial()
        self.assertEqual(pos.side, "black")
        for uci in ("h8", "h7", "i8", "i7", "j8", "j7", "k8", "k7"):
            move = GomokuMove.parse(uci)
            self.assertTrue(pos.classify(move).ok, uci)
            pos, result, reason = pos.apply(move)
        move = GomokuMove.parse("l8")
        pos, result, reason = pos.apply(move)
        self.assertEqual((result, reason), ("black_win", "five"))
        blocked = GOMOKU.initial()
        blocked, _, _ = blocked.apply(GomokuMove.parse("h8"))
        verdict = blocked.classify(GomokuMove.parse("h8"))
        self.assertEqual(verdict.error_type, "occupied")

    def test_last_coordinate_wins(self):
        pos = GOMOKU.initial()
        parsed, how = GOMOKU.parse_move("<move>h8</move> later <move>a1</move>", pos)
        self.assertEqual((parsed.uci, how), ("a1", "coordinate"))
        prompt = GOMOKU.user_prompt(GOMOKU.observe(pos, "black", [], None, 0, "blind"))
        self.assertIn("a1 is the bottom-left", prompt)
        self.assertNotIn("楚河", prompt)


class ChessTests(unittest.TestCase):
    def test_opening_and_last_move_tag(self):
        pos = CHESS.initial()
        self.assertTrue(pos.classify(ChessMove("e2e4")).ok)
        self.assertFalse(pos.classify(ChessMove("e2e5")).ok)
        parsed, how = CHESS.parse_move("<move>e2e4</move>\n<move>d2d4</move>", pos)
        self.assertEqual((parsed.uci, how), ("d2d4", "uci"))
        prompt = CHESS.user_prompt(CHESS.observe(pos, "white", [], None, 0, "blind"))
        self.assertIn("You are White", prompt)
        self.assertIn("<move>", prompt)

    def test_leaves_check_only_when_the_geometry_is_legal(self):
        in_check = ChessPosition(chess.Board("4k3/8/8/8/r7/8/8/R3K2r w - - 0 1"))
        blocked = in_check.classify(ChessMove("a1a8"))
        self.assertEqual(blocked.error_type, "illegal_move")
        exposed = ChessPosition(chess.Board("4k3/8/8/8/8/8/3r4/4K3 w - - 0 1"))
        self.assertFalse(exposed.board.is_check())
        walked = exposed.classify(ChessMove("e1e2"))
        self.assertEqual(walked.error_type, "leaves_check")

    def test_san_and_hyphen_fallback(self):
        pos = CHESS.initial()
        parsed, how = CHESS.parse_move("<move>e2e4</move> <move>Nf3</move>", pos)
        self.assertEqual((parsed.uci, how), ("g1f3", "san"))
        parsed, how = CHESS.parse_move("<move>e2-e4</move>", pos)
        self.assertEqual((parsed.uci, how), ("e2e4", "uci"))
        castle = ChessPosition(chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"))
        parsed, how = CHESS.parse_move("<move>O-O</move>", castle)
        self.assertEqual((parsed.uci, how), ("e1g1", "san"))

    def test_fools_mate(self):
        pos = CHESS.initial()
        result = reason = None
        for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
            move = ChessMove(uci)
            self.assertTrue(pos.classify(move).ok, uci)
            pos, result, reason = pos.apply(move)
        self.assertEqual((result, reason), ("black_win", "checkmate"))

    def test_registry(self):
        self.assertEqual(get_game("xiangqi").seats, ("red", "black"))
        self.assertEqual(get_game("chess").seats, ("white", "black"))
        self.assertEqual(get_game("gomoku").seats, ("black", "white"))


if __name__ == "__main__":
    unittest.main()
