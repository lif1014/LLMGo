import math
import unittest

from llmgo.eval.elo import game_points, performance_rating, summarize_games


class EloTests(unittest.TestCase):
    def test_even_score_stays_on_the_anchor(self):
        rated = performance_rating(800, 20, 40)
        self.assertAlmostEqual(rated["rating"], 800.0)
        self.assertAlmostEqual(rated["p"], 0.5)

    def test_known_advantage(self):
        rated = performance_rating(800, 22, 40)
        expected = 800 + 400 * math.log10(0.55 / 0.45)
        self.assertAlmostEqual(rated["rating"], expected)
        self.assertGreater(rated["se"], 0)

    def test_sweep_has_no_point_rating(self):
        self.assertEqual(performance_rating(800, 0, 40)["saturated"], "low")
        self.assertIsNone(performance_rating(800, 40, 40)["rating"])

    def test_forfeit_counts_in_the_match_score_and_is_split_out(self):
        self.assertEqual(game_points("black_win", "white"), 0.0)
        self.assertEqual(game_points("draw", "white"), 0.5)
        report = summarize_games(
            [
                {"result": "black_win", "result_reason": "illegal_forfeit", "model_seat": "white", "anchor_elo": 800},
                {"result": "white_win", "result_reason": "checkmate", "model_seat": "white", "anchor_elo": 800},
                {"result": None, "result_reason": None, "model_seat": "white", "anchor_elo": 800},
            ]
        )
        self.assertEqual(report["unfinished"], 1)
        bucket = report["by_anchor"][0]
        self.assertEqual(bucket["forfeits"], 1)
        self.assertEqual(bucket["all"]["games"], 2)
        self.assertEqual(bucket["all"]["score"], 1.0)
        self.assertEqual(bucket["played"]["games"], 1)
        self.assertEqual(bucket["played"]["p"], 1.0)


if __name__ == "__main__":
    unittest.main()
