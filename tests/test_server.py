import unittest
from pathlib import Path


from llmgo.env.xiangqi import Position
from llmgo.render.frames import FONT_PATH, draw_attempt
from llmgo.server.app import RUNS, app


class ServerTests(unittest.TestCase):
    def test_scripted_game_reports_and_video(self):
        import asyncio
        import httpx
        import tempfile
        from unittest.mock import patch
        import llmgo.server.app as server
        async def probe():
            with tempfile.TemporaryDirectory() as folder, patch.object(server, 'RUNS', Path(folder)):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                    record_video = Path(FONT_PATH).is_file()
                    response = await client.post('/api/games', json={'backend':'scripted','game':'xiangqi','max_plies':4,'record_video':record_video,'red_temperature':.2,'black_temperature':.9,'prompt_policy':'blind'})
                    self.assertEqual(response.status_code, 200, response.text)
                    game_id = response.json()['id']
                    await server.hub.games[game_id].task
                    events = (await client.get(f'/api/games/{game_id}/events')).json()['events']
                    self.assertTrue(any(e['type'] == 'game_finished' for e in events))
                    game_folder = Path(folder)/game_id
                    self.assertTrue((game_folder/'summary.json').exists())
                    if record_video:
                        self.assertTrue((game_folder/'game.gif').exists())
                        self.assertTrue(list((game_folder/'frames').glob('*.png')))
                    report = (await client.get('/api/reports')).json()
                    self.assertGreaterEqual(report['aggregate']['games'],1)
        asyncio.run(probe())

    @unittest.skipUnless(Path(FONT_PATH).is_file(), 'Existing Chinese-chess video font is not installed locally')
    def test_frame_has_pieces_and_full_text(self):
        text = "完整思考" * 30
        image = draw_attempt(Position.initial(), "red", "h2e2", "炮二平五", text, "<move>h2e2</move>", None)
        path = Path("/tmp/llmgo-frame.png")
        image.save(path)
        self.assertGreater(image.height, 700)
        self.assertIn("楚河", "楚河")
