"""Acceptance regressions for the complete-move harness and real UCI pipeline."""
import asyncio
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import chess
import httpx
import pytest

from llmgo.harness.agent import AskResult, choose_move
from llmgo.harness.choices import parse_choice, parse_move, split_thinking
from llmgo.harness.config import HarnessConfig
from llmgo.harness.player import VllmAsker
from llmgo.harness.prompt import move_prompt
from llmgo.harness.skills import select_skills
from llmgo.harness.tools import prepare_candidates

ROOT = Path(__file__).resolve().parents[1]


def run_choice(board, ask, **kwargs):
    events = []
    async def emit(event): events.append(event)
    result = asyncio.run(choose_move(board, [], ask, emit, seat='white' if board.turn else 'black', **kwargs))
    return result, events


def test_all_legal_moves_are_kept_in_fixed_groups_with_no_position_mutation():
    board = chess.Board()
    # A nonempty stack must also survive preprocessing, for repetition adjudication.
    for san in ('Nf3', 'Nf6', 'Ng1', 'Ng8'): board.push_san(san)
    before, stack = board.fen(), list(board.move_stack)
    candidates, observations = prepare_candidates(board)
    assert len(candidates) == 20 and {item['uci'] for item in candidates} == {move.uci() for move in board.legal_moves}
    assert [item['uci'] for item in candidates[:4]] == ['b1a3','b1c3','g1f3','g1h3']
    assert (board.fen(), board.move_stack) == (before, stack)
    assert prepare_candidates(board) == (candidates, observations)
    prompt = move_prompt(board, [], candidates, observations)
    assert 'pawn e2: e2e3 (e3), e2e4 (e4)' in prompt
    assert all(len(re.findall(r'\b' + item['uci'] + r' \(', prompt)) == 1 for item in candidates)
    assert not any(text in prompt for text in ('Tools\n', 'Pick\n', 'one letter', 'FEN ', '8 r n b'))


@pytest.mark.parametrize('fen,expected', [
    ('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1', {'e5d6'}),
    ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', {'e1g1','e1c1'}),
    ('7k/P7/8/8/8/8/8/7K w - - 0 1', {'a7a8q','a7a8r','a7a8b','a7a8n'}),
])
def test_special_moves_are_distinct_options(fen, expected):
    board = chess.Board(fen)
    candidates, _ = prepare_candidates(board)
    assert expected <= {item['uci'] for item in candidates}
    assert len(candidates) == board.legal_moves.count()
    assert all(item['tags'] for item in candidates if item['uci'] in expected)


def test_candidate_list_has_no_alphabet_limit_in_a_busy_position():
    board = chess.Board('r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1')
    candidates, _ = prepare_candidates(board)
    assert len(candidates) > 26
    assert {item['uci'] for item in candidates} == {move.uci() for move in board.legal_moves}


def test_tactical_previews_inform_without_pruning_bad_candidates():
    board = chess.Board('rn1qkbnr/3b2pp/2p2p2/pp1p4/1Q2P3/8/PPP2PPP/RNB1KBNR w KQkq - 0 12')
    candidates, notes = prepare_candidates(board)
    assert 'b4a5' in {item['uci'] for item in candidates}
    assert any('b4a5:' in line and 'a8a5' in line and '-8' in line for line in notes)
    board = chess.Board(); board.push_san('f3'); board.push_san('e5')
    candidates, notes = prepare_candidates(board)
    assert any('g2g4:' in line and 'd8h4' in line and 'mate-in-one' in line for line in notes)
    assert 'defend-mate' in {card['id'] for card in select_skills(board, candidates=candidates)}
    # The warning is only triggered when an actual one-reply mate has been found.
    initial, _ = prepare_candidates(chess.Board())
    assert 'defend-mate' not in {card['id'] for card in select_skills(chess.Board(), candidates=initial)}


@pytest.mark.parametrize('text,expected', [
    ('{"move":"e2e4"}', 'e2e4'),
    ('e2e4', 'e2e4'),
    ('<think>A and B are alternatives. e2e3 is possible.</think>\n{"move":"e2e4"}', 'e2e4'),
    ('Thoughts mentioning A, B, e2e3.\n</think>\n{"move":"e2e4"}', 'e2e4'),
    ('```json\n{"move":"e2e4"}\n```', 'e2e4'),
    ('<think>{"move":"e2e4"}', None),
    ('I considered e2e3 and choose e2e4.', None),
    ('{"move":"e2e3","move":"e2e4"}', None),
    ('{"move":"e2e4","why":"best"}', None),
    ('{"move":"e2e5"}', None),
    ('{"move":"e2e4"}\n{"move":"e2e3"}', None),
    ('["e2e4"]', None),
    ('G', None),
])
def test_only_explicit_complete_legal_final_answers_are_accepted(text, expected):
    assert parse_move(text, {'e2e3','e2e4'}) == expected


def test_actual_three_rejected_g_responses_now_have_a_separate_final_answer():
    records = json.loads((ROOT/'tests/fixtures/qwen_mixed_reasoning.json').read_text())['cases']
    assert len(records) == 3
    for record in records:
        payloads = []
        def respond(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, text=record['raw_response'], headers={'content-type':'application/json'})
        asker = VllmAsker('http://test/v1','Qwen3.5-4B',transport=httpx.MockTransport(respond))
        result = asyncio.run(asker('historical replay',8192))
        assert result.text == 'G' and result.reasoning and result.normalization_error is None
        assert parse_choice(result.text, set(record['allowed_choices'])) == 'G'
        assert result.raw_response == record['raw_response'] and result.tokens == record['completion_tokens']
        assert payloads[0]['chat_template_kwargs']['enable_thinking'] is True
        # The old piece answer is not a complete move in the new protocol.
        assert parse_move(result.text, {'e2e4'}) is None


@pytest.mark.parametrize('separated', [False, True])
def test_model_request_accepts_complete_move_with_both_reasoning_response_styles(separated):
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        message = {'content':'{"move":"e2e4"}', 'reasoning':'I compared e2e3 and e2e4.'} if separated else {
            'content':'I discussed A, B, and e2e3.\n</think>\n{"move":"e2e4"}', 'reasoning':None}
        return httpx.Response(200,json={'choices':[{'message':message,'finish_reason':'stop'}],
                                       'usage':{'prompt_tokens':500,'completion_tokens':80}})
    decision, events = run_choice(chess.Board(),VllmAsker('http://test/v1','Qwen3.5-4B',transport=httpx.MockTransport(respond)))
    assert decision.uci == 'e2e4' and decision.source == 'model' and len(requests)==1
    assert requests[0]['chat_template_kwargs']['enable_thinking'] is True
    finished = next(e for e in events if e['type']=='model_request' and e['status']=='finished')
    assert finished['accepted'] and finished['text']=='{"move":"e2e4"}' and finished['reasoning']
    assert finished['raw_response'] and finished['usage']['completion_tokens']==80
    assert decision.diagnostics['request_count']==1


def test_retry_cannot_spend_more_than_the_shared_move_budget():
    caps = []
    async def ask(prompt, cap):
        caps.append(cap)
        return AskResult('A' if len(caps)==1 else '{"move":"e2e4"}',8 if len(caps)==1 else 2,'stop')
    decision, _ = run_choice(chess.Board(),ask,budget=100,config=HarnessConfig(move_tokens=10))
    assert caps == [10,2] and decision.tokens==10 and decision.source=='model'
    assert decision.diagnostics['stage_tokens']=={'move':10}


def test_forced_move_does_not_consume_model_tokens():
    board=chess.Board('7k/5K2/5Q2/8/8/8/8/8 b - - 0 1')
    assert board.legal_moves.count()==1
    async def ask(prompt,cap): raise AssertionError('forced move must not ask the model')
    decision, events = run_choice(board,ask)
    assert decision.source=='forced' and decision.uci=='h8h7' and decision.tokens==0
    assert decision.diagnostics['request_count']==0
    assert not any(e['type']=='model_request' for e in events)


def test_even_complete_json_is_rejected_when_the_server_reports_length():
    async def ask(prompt,cap): return AskResult('{"move":"e2e4"}',cap,'length')
    decision, _ = run_choice(chess.Board(),ask,budget=20)
    assert decision.source=='random' and decision.diagnostics['fallback_reason']=='length_without_choice'


def test_eval_cli_runs_real_fairy_stockfish_against_local_http_model(tmp_path):
    """Actual CLI + HTTP transport + production UciPlayer, without a GPU or Qwen weights."""
    payloads = []
    class ModelServer(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            assert self.path == '/v1/chat/completions'
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            payloads.append(body)
            prompt = body['messages'][0]['content']
            options = re.findall(r'\b([a-h][1-8][a-h][1-8][qrbn]?) \(', prompt.split('Legal moves (',1)[1])
            assert options and 'Tools\n' not in prompt and 'Pick\n' not in prompt
            assert body['chat_template_kwargs']['enable_thinking'] is True
            assert body['stream'] and body['stream_options']['include_usage']
            self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
            chunks = [
                {'choices':[{'delta':{'content':'I compared candidate alternatives.'},'finish_reason':None}]},
                {'choices':[{'delta':{'content':'\n</think>\n' + json.dumps({'move':options[0]})},'finish_reason':None}]},
                {'choices':[{'delta':{},'finish_reason':'stop'}]},
                {'choices':[],'usage':{'prompt_tokens':500,'completion_tokens':50}},
            ]
            for chunk in chunks:
                self.wfile.write(('data: ' + json.dumps(chunk) + '\n\n').encode()); self.wfile.flush()
            self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush()
    server=ThreadingHTTPServer(('127.0.0.1',0),ModelServer)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    # Server paths remain untouched; only this temporary test protocol uses local fixtures.
    protocol=json.loads((ROOT.parent/'PostTrainBench/protocol.json').read_text())
    protocol.update(engine=str(ROOT/'engines/fairy-stockfish'),llmgo_root=str(ROOT),max_plies=6)
    protocol_path=tmp_path/'protocol.json';protocol_path.write_text(json.dumps(protocol))
    output=tmp_path/'metrics.json'
    try:
        completed=subprocess.run([sys.executable,str(ROOT/'scripts/eval_harness.py'),'--protocol',str(protocol_path),
                                  '--base-url',f'http://127.0.0.1:{server.server_port}/v1','--games','2','--output',str(output)],
                                 cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'CUDA_VISIBLE_DEVICES':''},
                                 capture_output=True,text=True,timeout=60)
        assert completed.returncode==0,completed.stdout+'\n'+completed.stderr
        candidate=tmp_path/'candidate'
        shutil.copytree(ROOT/'src/llmgo/harness',candidate,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        settings=json.loads((candidate/'settings.json').read_text())
        settings.update(move_tokens=1024,skills_limit=1)
        (candidate/'settings.json').write_text(json.dumps(settings))
        probe_output=tmp_path/'probe/metrics.json'
        probe=subprocess.run([sys.executable,str(ROOT.parent/'PostTrainBench/evaluate.py'),
                              '--protocol',str(protocol_path),'--model-path',str(tmp_path/'unused-local-test-weights'),
                              '--harness-dir',str(candidate),'--base-url',f'http://127.0.0.1:{server.server_port}/v1',
                              '--games','1','--output',str(probe_output)],cwd=ROOT,
                             env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'CUDA_VISIBLE_DEVICES':''},
                             capture_output=True,text=True,timeout=60)
        assert probe.returncode==0,probe.stdout+'\n'+probe.stderr
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)
    assert completed.returncode==0,completed.stdout+'\n'+completed.stderr
    metrics=json.loads(output.read_text())
    assert len(metrics['games'])==2 and metrics['random_moves']==0 and metrics['random_rate']==0
    assert metrics['request_count']==6 and metrics['model_decisions']==6 and len(payloads)==9
    assert metrics['forced_moves']==0 and metrics['primary_score']==.5
    for row in metrics['games']:
        folder=tmp_path/'harness_runs'/row['game_id']
        records=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        assert len([e for e in records if e['type']=='attempt' and e['source']=='engine'])==3
        assert all(e['stage']=='move' for e in records if e['type']=='model_request')
        assert len([e for e in records if e['type']=='tool_call' and e['status']=='finished'])==6
        assert all(e['accepted'] for e in records if e['type']=='model_request' and e['status']=='finished')
        assert any(e['type']=='token' and e['channel']=='reasoning' for e in records)
        assert all(e['streamed'] for e in records if e['type']=='model_request' and e['status']=='finished')
        assert json.loads((folder/'config.json').read_text())['harness_config']['request_timeout_sec']==540
    probe_metrics=json.loads(probe_output.read_text())
    assert probe_metrics['request_count']==3 and probe_metrics['random_moves']==0
    assert all(payload['max_tokens']==1024 for payload in payloads[6:])
    row=probe_metrics['games'][0]
    probe_config=json.loads((probe_output.parent/'harness_runs'/row['game_id']/'config.json').read_text())
    assert probe_config['harness_config']['skills_limit']==1 and probe_config['harness_config']['move_tokens']==1024
