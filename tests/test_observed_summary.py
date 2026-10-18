"""Per-game progress for snapshots must never borrow global or another game's metrics."""
import asyncio
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from llmgo.eval.report import observed_summary

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name,turns,requests,tokens,fallbacks', [
    ('20261018-153738-eaa2',2,6,32034,{'incomplete_thinking':2}),
    ('20261018-153740-0a3b',3,7,39945,{'incomplete_thinking':2,'request_timeout':1}),
])
def test_latest_run_api_derives_its_own_progress_without_inventing_an_end(name, turns, requests, tokens, fallbacks):
    import llmgo.server.app as server
    folder = ROOT/'runs/harness_runs'/name
    assert not (folder/'summary.json').exists()
    summary = observed_summary(folder)
    assert summary['game_id'] == name and summary['provisional']
    assert summary['model_turns'] == turns and summary['model_decisions'] == 0 and summary['random_moves'] == turns
    assert summary['request_count'] == requests and summary['pending_requests'] == 1
    assert summary['completed_requests'] == requests - 1 and summary['completion_tokens'] == tokens
    assert summary['fallback_reasons'] == fallbacks and summary['tokens_exact']
    assert summary['result'] is None and summary['token_scope'] == 'completed_requests'
    async def probe():
        with patch.object(server,'RUNS',ROOT/'runs'),patch.object(server,'hub',server.Hub()):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),base_url='http://test') as client:
                info = (await client.get('/api/games/'+name)).json()
                assert info['summary'] == summary
                report = (await client.get('/api/reports')).json()
                row = next(e for e in report['games'] if e['game_id']==name)
                assert row['model_decisions']==0 and row['model_turns']==turns
    asyncio.run(probe())
    assert not (folder/'summary.json').exists()
