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
    assert 'pawn e2: e2e3 e2e4' in prompt
    legal_section = prompt.split('Legal moves (', 1)[1].split('\n\nFinal answer:', 1)[0]
    assert {m for m in re.findall(r'\b[a-h][1-8][a-h][1-8][qrbn]?\b', legal_section)} == {m.uci() for m in board.legal_moves}
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
