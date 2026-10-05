import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from llmgo.env.xiangqi import Position
from llmgo.render.frames import draw_attempt
from llmgo.server.app import RUNS, app


class ServerTests(unittest.TestCase):
    def test_scripted_game_reports_and_video(self):
        with TestClient(app) as client:
            response = client.post(
                "/api/games",
                json={
                    "backend": "scripted",
                    "max_plies": 4,
                    "record_video": True,
                    "red_temperature": 0.2,
                    "black_temperature": 0.9,
                    "prompt_policy": "blind",
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            game_id = response.json()["id"]
            seen = []
            with client.websocket_connect(f"/ws/games/{game_id}") as socket:
                while True:
                    event = socket.receive_json()
                    seen.append(event["type"])
                    if event["type"] == "game_finished":
                        self.assertIsNotNone(event["summary"])
                        self.assertGreater(event["summary"]["first_try_total"], 0)
                        break
            self.assertIn("token", seen)
            self.assertIn("summary", seen)
            folder = RUNS / game_id
            self.assertTrue((folder / "summary.json").exists())
            self.assertTrue((folder / "game.gif").exists())
            self.assertTrue(list((folder / "frames").glob("*.png")))
            report = client.get("/api/reports").json()
            self.assertGreaterEqual(report["aggregate"]["games"], 1)
            self.assertIn("first_try_legal_rate", report["aggregate"])

    def test_frame_has_pieces_and_full_text(self):
        text = "完整思考" * 30
        image = draw_attempt(Position.initial(), "red", "h2e2", "炮二平五", text, "<move>h2e2</move>", None)
        path = Path("/tmp/llmgo-frame.png")
        image.save(path)
        self.assertGreater(image.height, 700)
        self.assertIn("楚河", "楚河")
