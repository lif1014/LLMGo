import asyncio
import importlib.util
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import chess
import chess.pgn
import httpx
import pytest
from llmgo.harness.agent import AskResult, choose_move
from llmgo.harness.config import HarnessConfig
from llmgo.harness.player import VllmAsker
from llmgo.harness.skills import load_cards, select_skills
from llmgo.harness.tools import capture_threats, king_line, preview_move

ROOT = Path(__file__).resolve().parents[1]


def decide(ask, board=None, config=None, budget=32768):
    events = []
    async def emit(event): events.append(event)
    decision = asyncio.run(choose_move(board or chess.Board(), [], ask, emit, seat='white', config=config, budget=budget))
    return decision, events


def test_preview_identifies_recorded_queen_blunder_without_mutation():
    board = chess.Board('rn1qkbnr/3b2pp/2p2p2/pp1p4/1Q2P3/8/PPP2PPP/RNB1KBNR w KQkq - 0 12')
    before = board.fen()
    result = preview_move(board, chess.Move.from_uci('b4a5'))
    reply = next(r for r in result['recaptures'] if r['uci'] == 'a8a5')
    assert reply['material_delta_cp'] == -800
    assert board.fen() == before and not board.move_stack


def test_preview_finds_recorded_mate_and_next_move_mate():
    board = chess.Board('2b1k2r/1p2Pp1p/2p2b1n/2P5/8/6KP/4q1PR/4r3 b k - 0 29')
    assert preview_move(board, chess.Move.from_uci('e2e3'))['checkmate']
    board = chess.Board()
    board.push_uci('f2f3'); board.push_uci('e7e5')
    assert 'd8h4' in preview_move(board, chess.Move.from_uci('g2g4'))['opponent_mate_in_one']


def test_preview_handles_en_passant_promotion_and_draw():
    board = chess.Board('7k/8/8/8/3p4/8/4P3/7K w - - 0 1')
    result = preview_move(board, chess.Move.from_uci('e2e4'))
    assert any(r['uci'] == 'd4e3' and r['material_delta_cp'] == -100 for r in result['recaptures'])
    board = chess.Board('7k/P7/8/8/8/8/8/7K w - - 0 1')
    assert preview_move(board, chess.Move.from_uci('a7a8q'))['material_delta_cp'] == 800
    board = chess.Board('7k/8/8/8/3p4/2B5/8/K7 w - - 0 1')
    assert preview_move(board, chess.Move.from_uci('c3d4'))['terminal'] == 'insufficient_material'
    with pytest.raises(ValueError): preview_move(chess.Board(), chess.Move.from_uci('e2e5'))


def test_pinned_geometric_defender_does_not_hide_queen_loss():
    board = chess.Board('3rr2k/8/8/6b1/8/8/3QR3/4K3 w - - 0 1')
    assert board.is_pinned(chess.WHITE, chess.E2)
    threat = next(x for x in capture_threats(board) if x['capture'] == 'g5d2')
    assert not threat['recaptures'] and threat['local_material_loss_cp'] == 900


def test_defended_queen_can_still_lose_material_to_pawn():
    board = chess.Board('k7/8/8/2p5/3Q4/4P3/8/7K w - - 0 1')
    threat = next(x for x in capture_threats(board) if x['capture'] == 'c5d4')
    assert 'e3d4' in threat['recaptures'] and threat['local_material_loss_cp'] == 800


def test_castling_rights_and_current_legality_are_distinct():
    lines = king_line(chess.Board())
    assert 'castling_rights kingside queenside' in lines[1]
    assert 'castling_legal_now none' in lines[1]
    board = chess.Board('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1')
    assert 'e1g1' in king_line(board)[1] and 'e1c1' in king_line(board)[1]


def test_tools_preview_requests_and_budgets_are_logged():
    count = 0
    async def ask(prompt, cap):
        nonlocal count
        count += 1
        if '\nTools\n' in prompt:
            answer = 'F' if count == 1 else 'E'
        elif '\nPreview candidate\n' in prompt:
            answer = next(line.split()[0] for line in prompt.splitlines() if 'e2e4 e4' in line)
        elif '\nPick\n' in prompt:
            answer = next(line.split()[0] for line in prompt.splitlines() if line.endswith('pawn e2'))
        else: answer = 'B'
        return AskResult(answer, 20, 'stop', output='thought\n' + answer, reasoning='thought', usage={'prompt_tokens': 100, 'completion_tokens':20}, raw_response='raw-body')
    decision, events = decide(ask)
    assert decision.uci == 'e2e4' and decision.source == 'model'
    requests = [e for e in events if e['type'] == 'model_request' and e['status'] == 'finished']
    assert len(requests) == 5 and all(e['raw_response'] == 'raw-body' for e in requests)
    assert all(e['enable_thinking'] and e['max_tokens'] <= getattr(HarnessConfig(), e['stage'] + '_tokens') for e in requests)
    result = next(e for e in events if e['type'] == 'tool_call' and e['status'] == 'finished')
    assert result['tool'] == 'preview' and result['data']['uci'] == 'e2e4'
    assert decision.tokens == 100 and decision.diagnostics['prompt_tokens'] == 500
    assert decision.diagnostics['request_count'] == 5


def test_length_without_answer_remains_visible_despite_legal_fallback():
    async def ask(prompt, cap): return AskResult('', cap, 'length', output='unfinished thinking', reasoning='unfinished thinking')
    decision, events = decide(ask)
    assert decision.source == 'random'
    assert decision.diagnostics['length_count'] == 2
    assert decision.diagnostics['fallback_reason'] == 'length_without_choice'
    assert any(e['type'] == 'route_recovery' for e in events)
    assert any(e['type'] == 'fallback' and e['reason'] == 'length_without_choice' for e in events)


def test_timeout_and_piece_reserve_are_bounded():
    async def asleep(prompt, cap): await asyncio.sleep(1); return AskResult('A',1)
    decision, events = decide(asleep, config=HarnessConfig(request_timeout_sec=.01, turn_timeout_sec=.05))
    assert decision.source == 'random'
    assert decision.diagnostics['fallback_reason'] in {'request_timeout', 'turn_timeout'}
    assert any(e.get('failure_reason') == 'request_timeout' for e in events)


def test_every_historical_skill_evidence_replays_and_retrieval_is_conditional():
    games = {}
    with (ROOT/'src/llmgo/harness/skills/master_games.pgn').open() as handle:
        while (game := chess.pgn.read_game(handle)) is not None:
            assert not game.errors
            games[game.headers['GameId']] = game
    assert len(games) == 3
    for card in load_cards():
        assert card['when'] and card['counterexample'] and card['sources']
        for evidence in card.get('evidence', []):
            game = games[evidence['game']]; board = game.board(); moves = list(game.mainline_moves())
            for move in moves[:evidence['before_ply']]: board.push(move)
            assert board.san(moves[evidence['before_ply']]) == evidence['san']
    opening = select_skills(chess.Board(), 'piece')
    ending = select_skills(chess.Board('8/8/6k1/8/8/2K5/8/R7 w - - 0 1'), 'piece')
    assert any(c['id'] == 'development' for c in opening)
    assert any(c['id'] == 'endgame-activity' for c in ending)
    assert all(c['id'] != 'development' for c in ending)
    assert len(opening) <= 3 and select_skills(chess.Board(), 'tool') == []


def test_opening_examples_are_legal_and_enter_only_relevant_questions():
    from llmgo.harness.prompt import pick_prompt, render_board
    cards = [card for card in load_cards() if card['id'].startswith('opening-')]
    assert len(cards) == 6
    for card in cards:
        assert card['stages'] == ['piece', 'move']
        for example in card.get('opening_examples', []):
            game = chess.pgn.read_game(io.StringIO(example['pgn'] + ' *'))
            assert game is not None and not game.errors and list(game.mainline_moves())
            assert example['pgn'] in card['rule'] and example['source'] in card['sources']
    board = chess.Board()
    initial = select_skills(board, 'piece')
    assert [card['id'] for card in initial] == ['opening-white-centre', 'development']
    assert select_skills(board, 'tool') == []
    assert all(not card['id'].startswith('opening-') for card in select_skills(board, 'preview'))
    prompt = pick_prompt('White', render_board(board), [], [], [('A', 'pawn e2')], skills=initial)
    assert '1. e4 e5 2. Nf3' in prompt and 'Counterexample:' in prompt


@pytest.mark.parametrize('line,expected', [
    ('e4', 'opening-black-central-reply'),
    ('d4', 'opening-black-central-reply'),
    ('e4 e5', 'opening-open-game'),
    ('e4 e5 Nf3', 'opening-open-game'),
    ('e4 c5', 'opening-white-sicilian'),
    ('e4 c5 Nf3 Nc6 d4 cxd4', 'opening-white-sicilian'),
    ('e4 c6', 'opening-white-caro-kann'),
    ('e4 c6 d4 d5 Nd2 dxe4', 'opening-white-caro-kann'),
    ('d4 d5', 'opening-queen-pawn'),
    ('d4 d5 c4 e6 Nc3', 'opening-queen-pawn'),
])
def test_opening_knowledge_follows_actual_structure_including_recaptures(line, expected):
    board = chess.Board()
    for san in line.split(): board.push_san(san)
    actual = [card['id'] for card in select_skills(board, 'piece') if card['id'].startswith('opening-')]
    assert actual == [expected]


def test_opening_advice_yields_to_check_and_expires_after_early_moves():
    board = chess.Board()
    for san in 'e4 e5 Bc4 d6 Bb5+'.split(): board.push_san(san)
    cards = select_skills(board, 'piece')
    assert cards[0]['id'] == 'answer-check'
    assert not any(card['id'].startswith('opening-') for card in cards)
    # Castling and move horizon each independently end the concrete opening hints.
    board = chess.Board()
    for san in 'e4 e5 Nf3 Nc6 Bc4 Bc5 O-O'.split(): board.push_san(san)
    assert not any(card['id'].startswith('opening-') for card in select_skills(board, 'piece'))
    board = chess.Board(); board.push_san('e4'); board.push_san('e5'); board.fullmove_number = 7
    assert not any(card['id'].startswith('opening-') for card in select_skills(board, 'piece'))
    # Against a different centre, do not supply an unrelated Sicilian/Caro/open-game line.
    board = chess.Board(); board.push_san('e4'); board.push_san('e6')
    assert not any(card['id'].startswith('opening-') for card in select_skills(board, 'piece'))


def test_vllm_payload_usage_raw_body_and_http_error():
    payloads = []
    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={'choices':[{'finish_reason':'length','message':{'content':'','reasoning_content':'still thinking'}}], 'usage':{'prompt_tokens':12,'completion_tokens':16}})
    asker = VllmAsker('http://local/v1', 'Qwen3.5-4B', transport=httpx.MockTransport(handle))
    result = asyncio.run(asker('test',16))
    assert payloads[0]['chat_template_kwargs']['enable_thinking'] is True
    assert result.tokens == 16 and result.tokens_exact and result.finish_reason == 'length'
    assert result.reasoning == 'still thinking' and 'still thinking' in result.raw_response
    failed = VllmAsker('http://local/v1','Qwen3.5-4B', transport=httpx.MockTransport(lambda r:httpx.Response(400,text='context too long')))
    result = asyncio.run(failed('test',16))
    assert result.http_status == 400 and result.raw_response == 'context too long' and result.error


def test_model_and_engine_pipeline_with_mock_transport(tmp_path):
    from llmgo.runner.harness_loop import HarnessRunner, HarnessSeat, EngineSeat
    from llmgo.runtime.player import StreamEvent
    import subprocess
    from llmgo.store.jsonl import TrajectoryStore
    # Network is mocked; the actual UCI opponent runs locally on CPU.
    def handle(request):
        prompt = json.loads(request.content)['messages'][0]['content']
        answer = 'E' if '\nTools\n' in prompt else 'A'
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':answer,'reasoning_content':'smoke'}}], 'usage':{'prompt_tokens':20,'completion_tokens':3}})
    asker = VllmAsker('http://local/v1','Qwen3.5-4B',transport=httpx.MockTransport(handle))
    class NativeOpponent:
        # Synchronous native UCI transport avoids cross-thread socket wakeups
        # blocked by the local sandbox; the runner uses its normal EngineSeat.
        seat = 'black'
        name = 'native-fairy-smoke'
        def __init__(self):
            self.proc = subprocess.Popen([str(ROOT/'engines/fairy-stockfish')], cwd=ROOT/'engines', stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            self.send('uci'); self.read_until('uciok')
            self.send('setoption name Skill Level value -9')
            self.send('isready'); self.read_until('readyok')
        def send(self, line):
            self.proc.stdin.write(line + '\n'); self.proc.stdin.flush()
        def read_until(self, prefix):
            while True:
                line = self.proc.stdout.readline().strip()
                if line.startswith(prefix): return line
                if not line and self.proc.poll() is not None: raise RuntimeError('UCI engine stopped')
        async def play(self, obs):
            self.send('position fen ' + obs.fen)
            self.send('go depth 5 movetime 50')
            move = self.read_until('bestmove').split()[1]
            yield StreamEvent('content', f'<move>{move}</move>')
            yield StreamEvent('done', finish_reason='stop')
        def close(self):
            self.send('quit'); self.proc.wait(timeout=5)
    engine = NativeOpponent()
    async def emit(event): pass
    async def game():
        try:
            return await HarnessRunner(HarnessSeat('white',asker),EngineSeat(engine),TrajectoryStore(tmp_path/'game'),max_plies=4).run(emit)
        finally: engine.close()
    summary = asyncio.run(game())
    assert summary['attempts'] == 4 and summary['model_turns'] == 2
    assert summary['random_moves'] == 0 and summary['request_count'] > 0
    records = [json.loads(line) for line in (tmp_path/'game/events.jsonl').read_text().splitlines()]
    assert all(e.get('timestamp') and e.get('sequence') for e in records)
    assert any(e.get('source') == 'engine' for e in records)


def test_stage_allowance_is_shared_across_tool_questions():
    async def ask(prompt, cap):
        return AskResult('A', cap if '\nTools\n' in prompt else 1, 'stop')
    decision, events = decide(ask, config=HarnessConfig(tool_tokens=10,piece_tokens=20,preview_tokens=10,move_tokens=30),budget=100)
    assert decision.source == 'model'
    assert decision.diagnostics['stage_tokens']['tool'] == 10
    assert len([e for e in events if e['type']=='model_request' and e['status']=='finished' and e['stage']=='tool']) == 1
    assert any(e['type']=='route_recovery' and e['reason']=='stage_budget_exhausted' for e in events)


def test_tool_failure_is_not_hidden_by_legal_fallback():
    async def ask(prompt, cap): return AskResult('A',1,'stop')
    with patch('llmgo.harness.agent.tool_result',side_effect=ValueError('broken diagnostic tool')):
        decision, events = decide(ask)
    assert decision.source == 'random' and decision.diagnostics['fallback_reason']=='tool_error'
    failure=next(e for e in events if e['type']=='tool_call' and e['status']=='finished')
    assert 'broken diagnostic tool' in failure['error'] and failure['call_id']


def test_external_match_incremental_events_keep_partial_writes_out(tmp_path):
    import llmgo.server.app as server
    folder=tmp_path/'harness_runs'/'external-test';folder.mkdir(parents=True)
    (folder/'config.json').write_text(json.dumps({'model_seat':'white','model':'Qwen3.5-4B'}))
    first={'type':'model_request','status':'finished','seat':'white','stage':'tool','raw_response':'full raw response','usage':{'completion_tokens':42}}
    second={'type':'tool_call','status':'finished','seat':'white','tool':'preview','result':['full tool output']}
    events=folder/'events.jsonl';events.write_text(json.dumps(first)+'\n'+json.dumps(second)+'\n'+ '{"type":')
    async def probe():
        with patch.object(server,'RUNS',tmp_path):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),base_url='http://test') as client:
                info=(await client.get('/api/games/external-test')).json()
                assert info['external'] and info['live'] and info['model_seat']=='white'
                data=(await client.get('/api/games/external-test/events?after=1')).json()
                assert data['cursor']==2 and len(data['events'])==1 and data['events'][0]['result']==['full tool output']
                assert (await client.get('/api/games/external-test/events?after=-1')).status_code==400
                assert (await client.get('/api/games',params={})).status_code in {404,405}
                rejected=await client.post('/api/games',json={'game':'chess','backend':'harness','opponent':'model','max_plies':201})
                assert rejected.status_code==400
    asyncio.run(probe())


def test_malformed_budget_is_rejected_before_starting_a_game():
    for data in ({'piece_tokens':'lots'},{'max_lookups':True},{'request_timeout_sec':float('nan')}):
        with pytest.raises(ValueError):HarnessConfig.from_dict(data)


def test_engine_failure_is_not_a_model_win_or_partial_score(tmp_path):
    from llmgo.runner.harness_loop import HarnessRunner, HarnessSeat, EngineSeat
    from llmgo.store.jsonl import TrajectoryStore
    class BrokenOpponent:
        seat='white';name='broken-opponent'
        async def play(self, obs):
            raise RuntimeError('engine crashed')
            yield
    async def ask(prompt,cap):return AskResult('A',1)
    async def emit(event):pass
    summary=asyncio.run(HarnessRunner(EngineSeat(BrokenOpponent()),HarnessSeat('black',ask),TrajectoryStore(tmp_path/'game'),max_plies=2).run(emit))
    assert summary['result']=='aborted' and summary['result_reason']=='engine_error'
    spec=importlib.util.spec_from_file_location('harness_eval_errors',ROOT/'scripts/eval_harness.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(RuntimeError,match='no partial batch score'):
        module.collect_metrics({},[{'game_id':'broken','result':summary['result'],'result_reason':summary['result_reason'],'model_seat':'black'}],None)


def test_evaluation_entry_reuses_server_and_persists_batch_metrics(tmp_path):
    import sys
    spec = importlib.util.spec_from_file_location('harness_eval_entry', ROOT/'scripts/eval_harness.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    protocol = json.loads((ROOT.parent/'PostTrainBench/protocol.json').read_text())
    protocol['engine'] = str(ROOT/'engines/fairy-stockfish')
    protocol_path = tmp_path/'protocol.json'; protocol_path.write_text(json.dumps(protocol))
    output = tmp_path/'metrics.json'
    async def play(settings, urls, engine, runs, white, black):
        assert urls == ['http://127.0.0.1:8001/v1'] and (white, black) == (1, 1)
        assert runs == tmp_path/'harness_runs' and engine.is_file()
        assert settings['enable_thinking'] and settings['max_plies'] == 200
        assert settings['harness_artifact']['sha256'] == module.harness_manifest()['sha256']
        return [{'game_id': seat, 'model_seat': seat, 'result': result, 'random_moves': random,
                 'diagnostics': {'model_turns': 2, 'request_count': 4, 'completion_tokens': 80,
                                 'fallback_reasons': {'invalid_choice': 1} if random else {}}}
                for seat, result, random in [('white', 'white_win', 0), ('black', 'draw', 1)]]
    argv = ['eval_harness.py', '--protocol', str(protocol_path), '--output', str(output),
            '--base-url', 'http://127.0.0.1:8001/v1/', '--games', '2', '--batch', 'test-batch']
    with patch.object(sys, 'argv', argv), patch.object(module, '_run_games', side_effect=play), \
         patch.object(module, 'visible_gpu_ids', side_effect=AssertionError('Existing server needs no GPU discovery')):
        assert module.main() == 0
    metrics = json.loads(output.read_text())
    assert metrics['primary_score'] == .75 and metrics['random_rate'] == .25
    assert metrics['request_count'] == 8 and metrics['completion_tokens'] == 160
    assert metrics['fallback_reasons'] == {'invalid_choice': 1}
    assert json.loads((tmp_path/'harness_batches/test-batch.json').read_text()) == metrics
