"""Real-time SSE delivery, partial-response retention and browser transports."""
import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import patch

import chess
import httpx
import pytest
from fastapi.testclient import TestClient

from llmgo.harness.agent import choose_move
from llmgo.harness.config import HarnessConfig
from llmgo.harness.player import VllmAsker
from llmgo.harness.streaming import SSEDecoder
from llmgo.store.jsonl import TrajectoryStore


def event(delta=None, *, finish=None, usage=None):
    data = {'choices': [{'index':0,'delta':delta or {},'finish_reason':finish}]}
    if usage is not None: data = {'choices':[],'usage':usage}
    return ('data: ' + json.dumps(data, ensure_ascii=False) + '\r\n\r\n').encode()


class Fragments(httpx.AsyncByteStream):
    def __init__(self, chunks, gate=None): self.chunks=chunks; self.gate=gate; self.closed=False
    async def __aiter__(self):
        for index, chunk in enumerate(self.chunks):
            if index==1 and self.gate is not None: await self.gate.wait()
            yield chunk
            await asyncio.sleep(0)
    async def aclose(self): self.closed=True


@pytest.mark.parametrize('field', ['reasoning','reasoning_content','mixed','closing_only'])
def test_sse_text_arrives_before_completion_and_is_not_duplicated(field):
    async def probe():
        gate, seen = asyncio.Event(), asyncio.Event()
        first = event({'content':'<think>比較兩個走法。'} if field=='mixed' else
                      {'content':'比較兩個走法。'} if field=='closing_only' else {field:'比較兩個走法。'})
        tail = [event({'content':'</thi'}), event({'content':'nk>\n{"move":"e2e4"}'})] if field in ('mixed','closing_only') else [event({'content':'{"move":"e2e4"}'})]
        body = [first,*tail,event(finish='stop'),event(usage={'prompt_tokens':100,'completion_tokens':20}),b'data: [DONE]\r\n\r\n']
        stream=Fragments(body,gate)
        payloads=[]; changes=[]
        def respond(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=stream)
        asker=VllmAsker('http://model/v1','Qwen3.5-4B',transport=httpx.MockTransport(respond))
        async def emit(change):
            changes.append(change)
            if change['channel']=='reasoning' and '比較' in change['text']: seen.set()
        task=asyncio.create_task(asker.stream('prompt',100,emit))
        try:
            await asyncio.wait_for(seen.wait(),1)
            assert not task.done() and not gate.is_set()
        finally: gate.set()
        result=await asyncio.wait_for(task,1)
        assert result.text=='{"move":"e2e4"}' and result.reasoning=='比較兩個走法。'
        assert result.finish_reason=='stop' and result.tokens==20 and result.tokens_exact
        assert result.raw_response==b''.join(body).decode() and result.streamed and result.stream_chunks>1
        assert result.first_token_ms is not None and result.latency_ms>=result.first_token_ms
        assert payloads[0]['stream'] and payloads[0]['stream_options']['include_usage']
        assert payloads[0]['chat_template_kwargs']['enable_thinking'] is True
        assert payloads[0]['skip_special_tokens'] is False and payloads[0]['return_token_ids'] is True
        shown={'reasoning':'','content':''}
        for change in changes:
            key=change['channel'];shown[key]=change['text'] if change['replace'] else shown[key]+change['text']
        assert shown=={'reasoning':'比較兩個走法。','content':'{"move":"e2e4"}'}
        assert stream.closed
    asyncio.run(probe())


def test_sse_decoder_handles_utf8_byte_boundaries_comments_and_multiline_data():
    decoder=SSEDecoder(); raw=': ping\r\ndata: {"text":\r\ndata: "思考"}\r\n\r\n'.encode()
    events=[]
    for byte in raw: events.extend(decoder.feed(bytes([byte])))
    assert len(events)==1 and json.loads(events[0])=={'text':'思考'}
    assert decoder.feed(b'data: {"partial":',final=True)==[]


@pytest.mark.parametrize('mixed', [False, True])
def test_original_generated_ids_and_chatml_end_marker_survive_transport(mixed):
    async def probe():
        chunks = [
            {'choices':[{'delta':{'content':'<think>compare</think>e2e4' if mixed else 'e2e4'},'token_ids':[1,2], 'finish_reason':None}]},
            {'choices':[{'delta':{'content':'<|im_end|>'},'token_ids':[3], 'finish_reason':'stop'}]},
        ]
        body = [('data: '+json.dumps(chunk)+'\n\n').encode() for chunk in chunks] + [b'data: [DONE]\n\n']
        asker = VllmAsker('http://model/v1','Qwen',transport=httpx.MockTransport(
            lambda request: httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Fragments(body))))
        changes = []
        async def emit(event): changes.append(event)
        result = await asker.stream('choose',100,emit)
        assert result.token_ids == [1,2,3] and result.tokens == 3 and result.tokens_exact
        assert result.text == 'e2e4' and result.normalization_error is None
    asyncio.run(probe())


def test_stop_without_reasoning_boundary_does_not_turn_candidate_json_into_answer():
    async def probe():
        chunks = [event({'content':'Consider {"move":"e2e4"}. Another possibility is d2d4.\n\n{"move":"d2d4"}'}),
                  event(finish='stop'),b'data: [DONE]\n\n']
        asker = VllmAsker('http://model/v1','Qwen',transport=httpx.MockTransport(
            lambda request: httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Fragments(chunks))))
        async def emit(event): pass
        result = await asker.stream('choose',100,emit)
        assert result.text == '' and result.normalization_error == 'incomplete_thinking'
    asyncio.run(probe())


@pytest.mark.parametrize('failure', ['timeout','disconnect','malformed','missing_finish','length','cancel'])
def test_partial_generation_survives_failures_and_never_becomes_a_move(failure):
    async def probe():
        first=event({'reasoning':'partial thought: e2e4 is only a candidate'})
        gate=asyncio.Event();seen=asyncio.Event()
        class Broken(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield first
                if failure in ('timeout','cancel'): await gate.wait()
                elif failure=='disconnect': raise httpx.ReadError('connection lost')
                elif failure=='malformed': yield b'data: {broken json}\n\n'
                elif failure=='missing_finish':
                    yield event({'content':'{"move":"e2e4"}'})
                    yield b'data: [DONE]\n\n'
                else:
                    yield event({'content':'{"move":"e2e4"}'})
                    yield event(finish='length')
                    yield b'data: [DONE]\n\n'
        asker=VllmAsker('http://model/v1','Qwen3.5-4B',transport=httpx.MockTransport(
            lambda request:httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Broken())))
        events=[]
        async def emit(item):
            events.append(item)
            if item['type']=='token':seen.set()
        task=asyncio.create_task(choose_move(chess.Board(),[],asker,emit,seat='white',
                                           config=HarnessConfig(request_timeout_sec=.04)))
        if failure=='cancel':
            await asyncio.wait_for(seen.wait(),1); task.cancel()
            with pytest.raises(asyncio.CancelledError): await task
        else:
            decision=await asyncio.wait_for(task,1)
            assert decision.source=='random'
        finished=next(item for item in events if item['type']=='model_request' and item['status']=='finished')
        assert not finished['accepted'] and finished['reasoning'].startswith('partial thought')
        assert finished['raw_response'].startswith('data: ') and finished['completion_tokens']>0
        assert finished['stream_chunks']>0
        assert any(item['type']=='token' for item in events)
        if failure=='cancel': assert not any(item['type']=='fallback' for item in events)
        if failure=='timeout': assert finished['error']=='request_timeout'
    asyncio.run(probe())


def test_sse_http_error_and_nonstreaming_json_compatibility():
    async def probe():
        async def emit(change): pass
        asker=VllmAsker('http://model/v1','model',transport=httpx.MockTransport(lambda request:httpx.Response(400,text='bad request')))
        bad=await asker.stream('test',20,emit)
        assert bad.http_status==400 and bad.raw_response=='bad request' and bad.error and bad.tokens==0
        asker=VllmAsker('http://model/v1','model',transport=httpx.MockTransport(lambda request:httpx.Response(200,json={
            'choices':[{'message':{'content':'thought</think>{"move":"e2e4"}'},'finish_reason':'stop','token_ids':[1,2,3]}],
            'usage':{'completion_tokens':10}})))
        good=await asker.stream('test',20,emit)
        assert good.text=='{"move":"e2e4"}' and good.reasoning=='thought'
        assert not good.streamed and good.tokens==10 and good.token_ids==[1,2,3]
        nonstream = await asker('test',20)
        assert nonstream.token_ids==[1,2,3] and nonstream.text=='{"move":"e2e4"}'
    asyncio.run(probe())
