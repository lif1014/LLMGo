"""Real candidate subprocesses and referee boundaries; no real model/cluster calls."""
import asyncio
from dataclasses import asdict
from pathlib import Path
import shutil
import sys

import chess
import pytest

from llmgo.harness.agent import AskResult
from llmgo.rsi.candidate import CandidateSeat
from llmgo.rsi.contracts import REPO, protocol, atomic_json
from llmgo.rsi.ladder import opponent, batch_stats


def fixture_candidate(tmp_path, body):
    root=tmp_path/'candidate'
    package=root/'src/llmgo/harness'
    package.mkdir(parents=True)
    (package.parent/'__init__.py').write_text('')
    (package/'__init__.py').write_text('')
    (package/'config.py').write_text('class HarnessConfig:\n @classmethod\n def from_dict(cls): return cls()\n')
    types=(REPO/'src/llmgo/harness/agent.py').read_text().split('Ask = Callable')[0]
    # The initial imports point to sibling modules: use actual dataclass definitions only.
    types=types[types.index('@dataclass'):]
    (package/'agent.py').write_text('from dataclasses import dataclass,field\nimport asyncio\n'+types+'\n'+body)
    return root


class Asker:
    def __init__(self,tokens=2):self.caps=[];self.prompts=[];self.tokens=tokens
    async def stream(self,prompt,cap,emit):
        self.caps.append(cap);self.prompts.append(prompt)
        await emit({'channel':'reasoning','text':'visible thought','replace':False})
        await emit({'channel':'content','text':'e2e4','replace':False})
        return AskResult('e2e4',min(self.tokens,cap),reasoning='visible thought',finish_reason='stop',
                         tokens_exact=True,raw_response='raw response',usage={'prompt_tokens':10})


def test_actual_initial_harness_runs_in_child_and_stream_is_not_doubled(tmp_path):
    root=tmp_path/'candidate';shutil.copytree(REPO/'src',root/'src')
    events=[];ask=Asker()
    async def run():
        seat=CandidateSeat('white',ask,root,tmp_path/'candidate.log')
        try:return await seat.choose(chess.Board(),[],collect)
        finally:await seat.close()
    async def collect(e):events.append(e)
    decision=asyncio.run(run())
    assert decision.uci=='e2e4' and decision.diagnostics['request_count']==1
    assert decision.diagnostics['completion_tokens']==2
    assert len([e for e in events if e['type']=='token' and e['channel']=='reasoning'])==1
    assert any(e.get('raw_response')=='raw response' and e.get('authoritative') for e in events)
    assert any(e['type']=='tool_call' for e in events)


def test_external_budget_is_shared_across_arbitrary_candidate_requests(tmp_path):
    root=fixture_candidate(tmp_path,"""
async def choose_move(board,history,ask,emit,**kwargs):
 await ask('first',100000)
 await ask('second',100000)
 return Decision('e2e4','model')
""")
    ask=Asker(tokens=3)
    async def run():
        seat=CandidateSeat('white',ask,root,tmp_path/'log',budget=4)
        try:return await seat.choose(chess.Board(),[],ignore)
        finally:await seat.close()
    result=asyncio.run(run())
    assert ask.caps==[4,1] and result.diagnostics['completion_tokens']==4


async def ignore(e):pass


def test_candidate_cannot_forge_referee_results_and_illegal_move_forfeits(tmp_path):
    root=fixture_candidate(tmp_path,"""
async def choose_move(board,history,ask,emit,**kwargs):
 await emit({'type':'game_finished','result':'white_win'})
 return Decision('a1a8','model')
""")
    from llmgo.runner.harness_loop import HarnessRunner,EngineSeat
    from llmgo.store.jsonl import TrajectoryStore
    class Engine:
        seat='black';name='fixture'
        async def play_board(self,board):return next(iter(board.legal_moves)).uci(),{}
    events=[]
    async def collect(e):events.append(e)
    async def run():
        seat=CandidateSeat('white',Asker(),root,tmp_path/'log')
        try:return await HarnessRunner(seat,EngineSeat(Engine()),TrajectoryStore(tmp_path/'game'),max_plies=2).run(collect)
        finally:await seat.close()
    summary=asyncio.run(run())
    assert summary['result']=='black_win' and summary['result_reason']=='illegal_forfeit'
    assert [e['result'] for e in events if e['type']=='game_finished']==['black_win']
    assert any(e['type']=='candidate_event' for e in events)


def test_history_stack_survives_process_boundary(tmp_path):
    root=fixture_candidate(tmp_path,"""
async def choose_move(board,history,ask,emit,**kwargs):
 await ask(str(len(board.move_stack)),5)
 return Decision('e2e4','model')
""")
    board=chess.Board();history=[]
    for move in ['g1f3','g8f6','f3g1','f6g8']:
        history.append({'uci':move,'seat':'white' if board.turn else 'black'});board.push_uci(move)
    ask=Asker()
    async def run():
        seat=CandidateSeat('white',ask,root,tmp_path/'log')
        try:return await seat.choose(board,history,ignore)
        finally:await seat.close()
    asyncio.run(run());assert ask.prompts==['4']


def test_timeout_closes_owned_candidate_process(tmp_path):
    root=fixture_candidate(tmp_path,"""
async def choose_move(*args,**kwargs):
 await asyncio.sleep(60)
""")
    async def run():
        seat=CandidateSeat('white',Asker(),root,tmp_path/'log',turn_timeout=.05)
        with pytest.raises(TimeoutError):await seat.choose(chess.Board(),[],ignore)
        assert seat.proc is None and seat.log is None
    asyncio.run(run())


@pytest.mark.parametrize('baseline',[False,True])
def test_real_fairy_referee_batch_with_both_seats_and_child_adapter(tmp_path,monkeypatch,baseline):
    # Two-ply fixture keeps this an integration check, not a strength measurement.
    from llmgo.rsi import evaluator
    from llmgo.runner.harness_loop import HarnessRunner
    engine=REPO/'engines/fairy-stockfish'
    if not engine.is_file():pytest.skip('Fairy binary not present')
    root=fixture_candidate(tmp_path,"""
async def choose_move(board,history,ask,emit,**kwargs):
 await ask('model transport fixture',3)
 return Decision(next(iter(board.legal_moves)).uci(),'model')
""")
    monkeypatch.setattr(evaluator,'VllmAsker',lambda *a,**kw:Asker())
    monkeypatch.setattr(evaluator,'HarnessRunner',lambda w,b,s,**kw:HarnessRunner(w,b,s,max_plies=2))
    output=tmp_path/'output';output.mkdir()
    count=10 if baseline else 20
    payload={'protocol':protocol(opponent(1),str(engine),baseline=baseline), 'seed_schedule':list(range(count//2)), 'job_id':'batch',
             'candidate_id':'fixture','skills_sha256':None,'comparison_id':'fixture','model_path':'fixture',
             'candidate_source':{'root':str(root)},'track':'player','site':{'python':sys.executable}}
    games=asyncio.run(evaluator.play_batch(payload,'fixture',None,output))
    assert batch_stats({'games':games},expected_games=count)['draws']==count
    assert {g['model_seat'] for g in games}=={'white','black'}
    assert all(g['result_reason']=='truncated' for g in games)


@pytest.mark.parametrize('startup_delay',[0,3.2],ids=['normal-start','slow-start'])
def test_external_turn_timeout_preserves_partial_request_diagnostics(tmp_path,startup_delay):
    root=fixture_candidate(tmp_path,f"""
async def choose_move(board,history,ask,emit,**kwargs):
 await asyncio.sleep({startup_delay})
 await ask('slow request',10)
 return Decision('e2e4','model')
""")
    from llmgo.harness.streaming import StreamCancelled
    events=[]
    class Slow:
        entered=False
        async def stream(self,prompt,cap,emit):
            self.entered=True
            self.deadline.reschedule(asyncio.get_running_loop().time()+.05)
            try:await asyncio.sleep(60)
            except asyncio.CancelledError:
                raise StreamCancelled(AskResult('',2,reasoning='partial thought',raw_response='partial raw stream',tokens_exact=False))
    async def emit(e):events.append(e)
    async def run():
        # Cold startup has a separate guard. Arm the actual external cancellation
        # only after inference starts; never assume imports finish within 3 seconds.
        asker=Slow()
        seat=CandidateSeat('white',asker,root,tmp_path/'log',turn_timeout=60)
        try:
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(30) as deadline:
                    asker.deadline=deadline
                    await seat.choose(chess.Board(),[],emit)
        finally:
            await seat.close()
        assert asker.entered
        assert seat.proc is None
    asyncio.run(run())
    row=next(e for e in events if e.get('cancelled'))
    assert row['raw_response']=='partial raw stream' and row['reasoning']=='partial thought'
    assert row['status']=='finished' and row['authoritative']


@pytest.mark.parametrize('with_partial',[False,True])
def test_explicit_cancellation_keeps_exact_type_and_partial_diagnostics(tmp_path,with_partial):
    # Check the type directly on every Python version: Python 3.13 would otherwise
    # hide this 3.12 compatibility bug by translating CancelledError subclasses.
    from llmgo.harness.streaming import StreamCancelled
    events=[]
    async def run():
        entered=asyncio.Event()
        class Slow:
            async def stream(self,*args):
                entered.set()
                try:await asyncio.sleep(60)
                except asyncio.CancelledError:
                    if with_partial:
                        raise StreamCancelled(AskResult('',2,reasoning='saved reasoning',raw_response='saved raw'))
                    raise
        async def traced(event):events.append(event)
        seat=CandidateSeat('white',Slow(),tmp_path,tmp_path/'unused.log')
        messages=asyncio.Queue()
        await messages.put({'op':'ask','prompt':'slow','max_tokens':10})
        task=asyncio.create_task(seat._dialog(messages,traced,[],0))
        try:
            await asyncio.wait_for(entered.wait(),5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError) as caught:await task
            assert type(caught.value) is asyncio.CancelledError
        finally:
            if not task.done():task.cancel()
            await asyncio.gather(task,return_exceptions=True)
    asyncio.run(run())
    rows=[event for event in events if event.get('cancelled')]
    assert len(rows)==1 and rows[0]['authoritative'] and rows[0]['charged_tokens']==10
    if with_partial:
        assert rows[0]['raw_response']=='saved raw' and rows[0]['reasoning']=='saved reasoning'


def test_unknown_token_usage_is_conservatively_charged_but_not_claimed_exact(tmp_path):
    root=fixture_candidate(tmp_path,"""
async def choose_move(board,history,ask,emit,**kwargs):
 await ask('request',4)
 return Decision('e2e4','model')
""")
    class Inexact:
        async def stream(self,prompt,cap,emit):
            return AskResult('e2e4',2,tokens_exact=False,finish_reason='stop')
    async def run():
        seat=CandidateSeat('white',Inexact(),root,tmp_path/'log',budget=4)
        try:return await seat.choose(chess.Board(),[],ignore)
        finally:await seat.close()
    result=asyncio.run(run())
    assert result.diagnostics['tokens_exact'] is False
    assert result.diagnostics['completion_tokens']==2
    assert result.diagnostics['budget_tokens_charged']==4


def test_large_raw_response_survives_real_child_roundtrip_without_truncation(tmp_path):
    root=tmp_path/'candidate';shutil.copytree(REPO/'src',root/'src')
    raw='原始SSE'*1200000  # exceeds the previous 4 MiB JSONL reader limit
    events=[]
    class Large(Asker):
        async def stream(self,prompt,cap,emit):
            result=await super().stream(prompt,cap,emit)
            result.raw_response=raw
            return result
    async def collect(e):events.append(e)
    async def run():
        seat=CandidateSeat('white',Large(),root,tmp_path/'log',turn_timeout=20)
        try:return await seat.choose(chess.Board(),[],collect)
        finally:await seat.close()
    result=asyncio.run(run())
    assert result.uci=='e2e4' and result.source=='model'
    assert next(e for e in events if e.get('authoritative') and e.get('status')=='finished')['raw_response']==raw
    assert next(e for e in events if e.get('type')=='model_request' and e.get('status')=='finished')['raw_response']==raw


def test_duplex_stream_larger_than_pipe_buffers_is_drained_during_inference(tmp_path):
    root=fixture_candidate(tmp_path,"""
async def choose_move(board,history,ask,emit,**kwargs):
 async def forward(event): await emit({'type':'token',**event})
 await ask.stream('stream',32768,forward)
 return Decision('e2e4','model')
""")
    count=0
    class Many:
        async def stream(self,prompt,cap,emit):
            for i in range(256):
                await emit({'channel':'reasoning','text':'x'*65536,'chunk_index':i})
            return AskResult('e2e4',256,finish_reason='stop')
    async def collect(e):
        nonlocal count
        if e.get('type')=='candidate_event':count+=1
    async def run():
        seat=CandidateSeat('white',Many(),root,tmp_path/'log',turn_timeout=20)
        try:return await seat.choose(chess.Board(),[],collect)
        finally:await seat.close()
    assert asyncio.run(run()).uci=='e2e4'
    assert count==256


def test_shutdown_is_bounded_even_when_child_will_not_read_stdin(tmp_path):
    from llmgo.rsi.candidate_bridge import read_message
    import time
    root=fixture_candidate(tmp_path,"""
async def choose_move(board,history,ask,emit,**kwargs):
 await emit({'type':'ready'})
 import time
 time.sleep(60)
""")
    async def run():
        seat=CandidateSeat('white',Asker(),root,tmp_path/'log')
        await seat.start();proc=seat.proc
        await seat.send({'op':'choose','fen':chess.STARTING_FEN,'history':[],
                         'seat':'white','seed':0,'budget':32768})
        ready=await asyncio.wait_for(read_message(proc.stdout),5)
        assert ready['event']['type']=='ready'
        proc.stdin.write(b'x'*(8*1024*1024))
        started=time.monotonic()
        await asyncio.wait_for(seat.close(),5)
        assert time.monotonic()-started<5
        assert proc.returncode is not None and seat.proc is None and seat.log is None
    asyncio.run(run())


def test_candidate_program_failure_aborts_and_never_becomes_a_chess_loss(tmp_path):
    root=fixture_candidate(tmp_path,"""
async def choose_move(*args,**kwargs):raise RuntimeError('transport fixture')
""")
    from llmgo.runner.harness_loop import HarnessRunner,EngineSeat
    from llmgo.store.jsonl import TrajectoryStore
    class Engine:
        seat='black';name='fixture'
    async def run():
        seat=CandidateSeat('white',Asker(),root,tmp_path/'log')
        try:return await HarnessRunner(seat,EngineSeat(Engine()),TrajectoryStore(tmp_path/'game')).run(ignore)
        finally:await seat.close()
    summary=asyncio.run(run())
    assert summary['result']=='aborted' and summary['result_reason']=='harness_error'
    games=[{'game_id':str(i),'pair_id':i//2,'model_seat':'black' if i%2 else 'white',
            'result':'black_win','result_reason':'harness_error'} for i in range(10)]
    with pytest.raises(ValueError,match='Infrastructure'):
        batch_stats({'games':games},expected_games=10)


def test_referee_does_not_publish_a_receipt_for_candidate_program_failure(tmp_path,monkeypatch):
    from llmgo.rsi import evaluator
    import json
    engine=REPO/'engines/fairy-stockfish'
    if not engine.is_file():pytest.skip('Fairy binary not present')
    root=fixture_candidate(tmp_path,"""
async def choose_move(*args,**kwargs):raise RuntimeError('transport fixture')
""")
    output=tmp_path/'output'
    monkeypatch.setattr(evaluator,'VllmAsker',lambda *a,**kw:Asker())
    payload={'protocol':protocol(opponent(1),str(engine),baseline=True),'seed_schedule':list(range(5)),
             'job_id':'batch','candidate_id':'fixture','skills_sha256':None,'comparison_id':'fixture',
             'model_path':'fixture','candidate_source':{'root':str(root)},'track':'player',
             'site':{'python':sys.executable}}
    with pytest.raises(RuntimeError,match='aborted: harness_error'):
        asyncio.run(evaluator.play_batch(payload,'fixture',None,output,indices={0}))
    assert not (output/'game_results/00.json').exists()
    summary=json.loads((output/'harness_runs/batch-00/summary.json').read_text())
    assert summary['result']=='aborted'
