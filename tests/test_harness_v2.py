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


def test_automatic_tools_and_one_complete_move_request_are_logged():
    async def ask(prompt, cap):
        return AskResult('{"move":"e2e4"}', 20, 'stop', output='thought\nfinal', reasoning='thought',
                         usage={'prompt_tokens':100,'completion_tokens':20}, raw_response='raw-body')
    decision, events = decide(ask)
    assert decision.uci == 'e2e4' and decision.source == 'model'
    requests = [e for e in events if e['type']=='model_request' and e['status']=='finished']
    assert len(requests)==1 and requests[0]['stage']=='move' and requests[0]['raw_response']=='raw-body'
    assert requests[0]['enable_thinking'] and requests[0]['max_tokens']==32768
    tools = [e for e in events if e['type']=='tool_call' and e['status']=='finished']
    assert [e['tool'] for e in tools]==['position','candidates']
    assert len(tools[-1]['data']['candidates'])==20
    assert decision.tokens==20 and decision.diagnostics['prompt_tokens']==100
    assert not any(e['type']=='route_recovery' or e.get('stage') in ('tool','piece','preview') for e in events)


def test_length_without_final_is_not_guessed_from_thoughts():
    async def ask(prompt, cap):
        return AskResult('<think>I prefer e2e4. {"move":"e2e4"}', cap, 'length')
    decision, events = decide(ask)
    assert decision.source=='random' and decision.diagnostics['length_count']==1
    assert decision.diagnostics['fallback_reason']=='length_without_choice'
    assert any(e['type']=='fallback' for e in events)


def test_timeout_is_reported_and_does_not_loop():
    async def asleep(prompt, cap): await asyncio.sleep(1); return AskResult('{"move":"e2e4"}',1)
    decision, events = decide(asleep, config=HarnessConfig(request_timeout_sec=.01,turn_timeout_sec=.05))
    assert decision.source=='random' and decision.diagnostics['fallback_reason']=='request_timeout'
    assert decision.diagnostics['request_count']==1


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
    opening = select_skills(chess.Board(), 'move')
    ending = select_skills(chess.Board('8/8/6k1/8/8/2K5/8/R7 w - - 0 1'), 'move')
    assert any(c['id'] == 'opening-white-centre' for c in opening)
    assert all(c['id'] != 'development' for c in opening)
    later = chess.Board(); later.fullmove_number = 7
    assert any(c['id'] == 'development' for c in select_skills(later, 'move'))
    assert any(c['id'] == 'endgame-activity' for c in ending)
    assert all(c['id'] != 'development' for c in ending)
    assert len(opening) <= 3 and select_skills(chess.Board(), 'tool') == []


def test_opening_examples_are_legal_and_enter_only_relevant_questions():
    from llmgo.harness.prompt import move_prompt
    cards = [card for card in load_cards() if card['id'].startswith('opening-')]
    assert len(cards) == 6
    for card in cards:
        assert card['stages'] == ['move']
        for example in card.get('opening_examples', []):
            game = chess.pgn.read_game(io.StringIO(example['pgn'] + ' *'))
            assert game is not None and not game.errors and list(game.mainline_moves())
            assert example['pgn'] in card['rule'] and example['source'] in card['sources']
    board = chess.Board()
    initial = select_skills(board, 'move')
    assert any(card['id']=='opening-white-centre' for card in initial)
    assert select_skills(board, 'tool') == []
    assert all(not card['id'].startswith('opening-') for card in select_skills(board, 'preview'))
    from llmgo.harness.tools import prepare_candidates
    candidates, observations = prepare_candidates(board)
    initial = select_skills(board, 'move', candidates=candidates)
    assert [card['id'] for card in initial] == ['opening-white-centre']
    prompt = move_prompt(board, [], candidates, observations, skills=initial)
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
    actual = [card['id'] for card in select_skills(board, 'move', limit=20) if card['id'].startswith('opening-')]
    assert actual == [expected]


def test_opening_advice_yields_to_check_and_expires_after_early_moves():
    board = chess.Board()
    for san in 'e4 e5 Bc4 d6 Bb5+'.split(): board.push_san(san)
    cards = select_skills(board, 'move')
    assert cards[0]['id'] == 'answer-check'
    assert not any(card['id'].startswith('opening-') for card in cards)
    # Castling and move horizon each independently end the concrete opening hints.
    board = chess.Board()
    for san in 'e4 e5 Nf3 Nc6 Bc4 Bc5 O-O'.split(): board.push_san(san)
    assert not any(card['id'].startswith('opening-') for card in select_skills(board, 'move'))
    board = chess.Board(); board.push_san('e4'); board.push_san('e5'); board.fullmove_number = 7
    assert not any(card['id'].startswith('opening-') for card in select_skills(board, 'move'))
    # Against a different centre, do not supply an unrelated Sicilian/Caro/open-game line.
    board = chess.Board(); board.push_san('e4'); board.push_san('e6')
    assert not any(card['id'].startswith('opening-') for card in select_skills(board, 'move'))


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


def test_tool_failure_is_not_hidden_by_legal_fallback():
    async def ask(prompt, cap): raise AssertionError('must not call the model after a tool failure')
    with patch('llmgo.harness.agent.prepare_candidates',side_effect=ValueError('broken candidate tool')):
        decision, events = decide(ask)
    assert decision.source=='random' and decision.diagnostics['fallback_reason']=='tool_error'
    failure = next(e for e in events if e['type']=='tool_call' and e.get('error'))
    assert 'broken candidate tool' in failure['error'] and failure['status']=='finished'




def test_malformed_budget_is_rejected_before_starting_a_game():
    for data in ({'move_tokens':'lots'},{'attempts_per_choice':True},{'request_timeout_sec':float('nan')}):
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
    from llmgo.rsi.ladder import batch_stats
    with pytest.raises(ValueError,match="Incomplete/aborted"):
        batch_stats({'games':[{'game_id':str(i),'result':summary['result'],'result_reason':summary['result_reason'],'model_seat':'black'} for i in range(20)]})
