"""Original GPU SSE replay plus strict boundary and early-stream regressions."""
import asyncio
import json
from pathlib import Path

import httpx
import pytest
import chess

from llmgo.harness.player import VllmAsker
from llmgo.harness.choices import parse_move
from llmgo.harness.agent import choose_move

ROOT = Path(__file__).resolve().parents[1]
RUNS = ['20261018-153738-eaa2', '20261018-153740-0a3b']


class Body(httpx.AsyncByteStream):
    def __init__(self, raw): self.raw = raw
    async def __aiter__(self):
        for offset in range(0, len(self.raw), 65536):
            yield self.raw[offset:offset+65536]


def tokenizer_file(tmp_path, vocab):
    # A sparse vocabulary is sufficient to identify control tokens. Historical
    # non-control text comes directly from the original SSE, without faking BPE decoding.
    path = tmp_path / 'tokenizer.json'
    path.write_text(json.dumps({'version':'1.0','model':{'type':'WordLevel',
        'vocab': {'[UNK]':0, **vocab}, 'unk_token':'[UNK]'}}))
    return path


def test_full_latest_gpu_responses_recover_all_ten_completed_answers(tmp_path):
    path = tokenizer_file(tmp_path, {'</think>':248069, '<think>':248068, '<|im_end|>':248046})
    async def replay():
        accepted, rejected = [], []
        for name in RUNS:
            with (ROOT/'runs/harness_runs'/name/'events.jsonl').open() as handle:
                for line in handle:
                    event = json.loads(line)
                    if event['type'] != 'model_request' or event.get('status') != 'finished':
                        continue
                    raw = event['raw_response'].encode()
                    def respond(request):
                        return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Body(raw))
                    asker = VllmAsker('http://recorded/v1','Qwen3.5-4B',tokenizer_path=path,
                                      transport=httpx.MockTransport(respond))
                    changes = []
                    async def emit(change): changes.append(change)
                    result = await asker.stream(event['prompt'],event['max_tokens'],emit)
                    assert result.raw_response == event['raw_response']
                    move = parse_move(result.text,set(event['allowed_choices'])) if not result.error else None
                    if event['finish_reason'] == 'stop':
                        assert move and result.normalization_error is None and result.boundary_source == 'token_ids'
                        content = ''
                        for change in changes:
                            if change['channel'] == 'content':
                                content = change['text'] if change.get('replace') else content + change['text']
                        assert content.strip() == move
                        accepted.append((name,event['request_id'],move))
                        if len(accepted) == 1:
                            trace = []
                            async def record(item): trace.append(item)
                            decision = await choose_move(chess.Board(),[],asker,record,seat='white')
                            assert decision.source == 'model' and decision.uci == move
                            assert decision.diagnostics['request_count'] == 1
                            assert not any(e['type']=='fallback' for e in trace)
                            finished = next(e for e in trace if e['type']=='model_request' and e['status']=='finished')
                            assert finished['accepted'] and finished['boundary_source']=='token_ids'
                    else:
                        assert result.error and move is None and result.normalization_error == 'incomplete_thinking'
                        rejected.append((name,event['request_id']))
        assert len(accepted) == 10 and len(rejected) == 1
        assert rejected[0] == (RUNS[1],'4:white:2')
    asyncio.run(replay())


@pytest.mark.parametrize('bundled', [False, True])
def test_suppressed_marker_can_be_separate_or_bundled_without_reading_moves_from_thoughts(tmp_path, bundled):
    path = tokenizer_file(tmp_path, {'thought':10, '</think>':1, 'e2e4':2})
    delta = {'content':'thought  e2e4'}
    choices = [{'delta':delta,'token_ids':[10,1,2],'finish_reason':'stop'}] if bundled else [
        {'delta':{'content':'thought '},'token_ids':[10]},
        {'delta':{},'token_ids':[1]},
        {'delta':{'content':' e2e4'},'token_ids':[2],'finish_reason':'stop'}]
    raw = ''.join('data: '+json.dumps({'choices':[c]})+'\n\n' for c in choices).encode()+b'data: [DONE]\n\n'
    async def probe():
        asker = VllmAsker('http://model/v1','Qwen',tokenizer_path=path,transport=httpx.MockTransport(
            lambda request:httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Body(raw))))
        async def emit(change): pass
        result = await asker.stream('choose',100,emit)
        assert result.text == 'e2e4' and result.reasoning == 'thought' and result.boundary_source == 'token_ids'
    asyncio.run(probe())


def test_wrong_tokenizer_cannot_create_a_final_answer(tmp_path):
    path = tokenizer_file(tmp_path, {'DIFFERENT':10, '</think>':1, 'd2d4':2})
    raw = ('data: '+json.dumps({'choices':[{'delta':{'content':'thought e2e4'},
            'token_ids':[10,1,2],'finish_reason':'stop'}]})+'\n\n').encode()+b'data: [DONE]\n\n'
    async def probe():
        asker = VllmAsker('http://model/v1','Qwen',tokenizer_path=path,transport=httpx.MockTransport(
            lambda request:httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Body(raw))))
        async def emit(change): pass
        result = await asker.stream('choose',100,emit)
        assert result.text == '' and result.normalization_error == 'incomplete_thinking'
        assert result.boundary_error.startswith('token_text_mismatch')
    asyncio.run(probe())


def test_hidden_end_restores_live_final_before_request_finishes(tmp_path):
    path = tokenizer_file(tmp_path, {'</think>':1})
    async def probe():
        gate, seen = asyncio.Event(), asyncio.Event()
        def frame(delta, ids=None, finish=None):
            return ('data: '+json.dumps({'choices':[{'delta':delta,'token_ids':ids,
                'finish_reason':finish}]})+'\n\n').encode()
        class Paused(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield frame({'content':'thought mentioning d2d4'},[])
                yield frame({},[1])
                yield frame({'content':'\n\ne2e4'},[])
                await gate.wait()
                yield frame({},[], 'stop')
                yield b'data: [DONE]\n\n'
        asker = VllmAsker('http://model/v1','Qwen',tokenizer_path=path,transport=httpx.MockTransport(
            lambda request:httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Paused())))
        content = ''
        async def emit(change):
            nonlocal content
            if change['channel']=='content':
                content = change['text'] if change.get('replace') else content + change['text']
                if content.strip()=='e2e4': seen.set()
        task = asyncio.create_task(asker.stream('choose',100,emit))
        try:
            await asyncio.wait_for(seen.wait(),1)
            assert not task.done()
        finally: gate.set()
        result = await task
        assert result.text=='e2e4' and result.boundary_source=='token_ids'
    asyncio.run(probe())
