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


@pytest.mark.parametrize('stop', [False, True])
def test_websocket_displays_live_thought_while_model_is_still_generating(tmp_path, stop):
    import llmgo.server.app as server
    gate=threading.Event()
    first=event({'reasoning':'live thought visible before finish'})
    class Paused(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield first
            while not gate.is_set(): await asyncio.sleep(.01)
            yield event({'content':'{"move":"e2e4"}'})
            yield event(finish='stop')
            yield event(usage={'prompt_tokens':100,'completion_tokens':20})
            yield b'data: [DONE]\n\n'
    original=VllmAsker.__init__
    def init(self,*args,**kwargs):
        kwargs['transport']=httpx.MockTransport(lambda request:httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Paused()))
        original(self,*args,**kwargs)
    watchdog=threading.Timer(3,gate.set);watchdog.start()
    try:
        with patch.object(server,'RUNS',tmp_path),patch.object(server,'hub',server.Hub()),patch.object(VllmAsker,'__init__',init):
            with TestClient(server.app) as client:
                response=client.post('/api/games',json={'game':'chess','backend':'harness','opponent':'model','max_plies':1})
                assert response.status_code==200
                game_id=response.json()['id']
                with client.websocket_connect('/ws/games/'+game_id) as ws:
                    while True:
                        item=ws.receive_json()
                        assert item['type']!='game_finished','stream was buffered until completion'
                        if item['type']=='token': break
                    assert item['text']=='live thought visible before finish' and item['request_id']=='0:white:1'
                    assert not gate.is_set() and not server.hub.games[game_id].task.done()
                    stored=[json.loads(line) for line in (tmp_path/game_id/'events.jsonl').read_text().splitlines()]
                    assert any(item['type']=='token' for item in stored)
                    assert not any(item['type']=='model_request' and item['status']=='finished' for item in stored)
                    if stop:
                        assert client.post(f'/api/games/{game_id}/stop').status_code==200
                    else:
                        gate.set()
                    while True:
                        item=ws.receive_json()
                        if item['type']=='game_finished':break
                    if stop:
                        assert item['result']=='aborted' and item['result_reason']=='stopped'
                        stored=[json.loads(line) for line in (tmp_path/game_id/'events.jsonl').read_text().splitlines()]
                        finished=next(e for e in stored if e['type']=='model_request' and e['status']=='finished')
                        assert finished['failure_reason']=='request_cancelled' and finished['raw_response']==first.decode()
                        assert not any(e['type']=='fallback' for e in stored)
                        assert json.loads((tmp_path/game_id/'summary.json').read_text())['result']=='aborted'
                        assert not client.get(f'/api/games/{game_id}').json()['live']
                    else:
                        assert item['summary']['random_moves']==0 and item['summary']['request_count']==1
                public=client.get(f'/api/games/{game_id}/events').json()['events']
                assert any(item['type']=='token' for item in public)
    finally: gate.set();watchdog.cancel()


def test_external_cli_game_streams_new_file_records_and_waits_for_partial_lines(tmp_path):
    import llmgo.server.app as server
    store=TrajectoryStore(tmp_path/'harness_runs'/'external-stream')
    store.write_config({'model_seat':'white','model':'Qwen3.5-4B'})
    store.append({'type':'model_request','status':'started','seat':'white','stage':'move','request_id':'0:white:1'})
    store.append({'type':'token','seat':'white','request_id':'0:white:1','channel':'reasoning','text':'first live text','replace':False})
    token={'type':'token','seat':'white','request_id':'0:white:1','channel':'reasoning','text':' 還在思考','replace':False}
    raw=(json.dumps(token,ensure_ascii=False)+'\n').encode()
    # Stop inside a UTF-8 character to check that the reader waits for the entire record.
    split=raw.index('還'.encode())+1
    def finish():
        with store.events_path.open('ab') as handle: handle.write(raw[split:])
        store.append({'type':'game_finished','result':'draw','result_reason':'test','summary':{}})
    timer=None
    with patch.object(server,'RUNS',tmp_path),patch.object(server,'hub',server.Hub()),TestClient(server.app) as client:
        with client.websocket_connect('/ws/games/external-stream') as ws:
            assert ws.receive_json()['type']=='replay_begin'
            assert ws.receive_json()['type']=='model_request'
            first=ws.receive_json()
            assert first['type']=='token' and first['request_id']=='0:white:1' and first['text']=='first live text'
            with store.events_path.open('ab') as handle:handle.write(raw[:split])
            timer=threading.Timer(.2,finish);timer.start()
            following=[]
            while True:
                item=ws.receive_json();following.append(item)
                if item['type']=='game_finished':break
            assert [item['text'] for item in following if item['type']=='token']==[' 還在思考']
    if timer: timer.join()


def test_reconnect_coalescing_preserves_request_ids_and_replacements():
    from llmgo.server.app import _coalesce_tokens
    events=[{'type':'token','seat':'white','request_id':'1','channel':'content','text':'old','replace':False},
            {'type':'token','seat':'white','request_id':'1','channel':'content','text':'','replace':True},
            {'type':'token','seat':'white','request_id':'1','channel':'content','text':'new','replace':False},
            {'type':'token','seat':'white','request_id':'2','channel':'content','text':'other','replace':False}]
    merged=list(_coalesce_tokens(events))
    assert len(merged)==2 and merged[0]['request_id']=='1' and merged[0]['replace'] and merged[0]['text']=='new'
    assert merged[1]['request_id']=='2' and merged[1]['text']=='other'
