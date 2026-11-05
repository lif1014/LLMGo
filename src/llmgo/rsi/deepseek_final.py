"""Held-out Qwen versus DeepSeek Flash non-thinking match."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import time
from pathlib import Path

import chess

from llmgo.training.chess_r1 import ANSWER_PARSER, extract_answer, messages
from llmgo.rsi.game import ChessEpisode
from .contracts import MAX_GAME_PLIES, atomic_json, file_hash, inside, model_identity, read_json, utc
from .deepseek_client import extract_loose_answer, opponent_identity, request as deepseek_request
from .evaluator import candidate_messages, request_completion
from .service import VllmServer, _pick_ports, _wait_servers, visible_gpu_ids

OPPONENT = {'level': 'deepseek-flash', 'level_id': 'DeepSeek Flash',
            'version': 'non-thinking-last-box-san-v2', 'skill': None,
            'depth': None, 'movetime_sec': None, 'nodes': None,
            'threads': 1, 'hash_mb': 0, 'thinking': 'disabled'}
PROTOCOL = {'model_name': 'Qwen3.5-4B', 'max_tokens': 16384,
            'temperature': 1.0, 'top_p': .95, 'top_k': 20,
            'presence_penalty': 1.5, 'repetition_penalty': 1.0,
            'enable_thinking': False}
START_FEN = chess.STARTING_FEN


class DeepSeekPlayer:
    def __init__(self, side, game_id, seed):
        self.side, self.game_id = side, game_id
        self.rng = random.Random(seed)

    def _play_board(self, board):
        fen = board.fen()
        request_id = f'{self.game_id}-opponent-{len(board.move_stack)}-{time.time_ns()}'
        response = deepseek_request(messages(fen, enable_thinking=False), request_id, timeout=None)
        choice = response['choices'][0]
        if (choice['message'].get('reasoning_content') or '').strip():
            raise ValueError('Final opponent returned reasoning despite non-thinking request')
        content = choice['message'].get('content') or ''
        parsed = extract_loose_answer(content, fen, finish_reason=choice.get('finish_reason'))
        fallback = None
        if parsed.legal:
            uci = parsed.uci
        else:
            fallback = parsed.error or 'invalid_final_answer'
            uci = self.rng.choice(sorted(move.uci() for move in board.legal_moves))
        return uci, {**opponent_identity(), 'response_model':response.get('model'),
                     'request_id': request_id, 'raw_output': content,
                     'finish_reason': choice.get('finish_reason'),
                     'parsed_san': parsed.san, 'parsed_uci': parsed.uci,
                     'fallback_reason': fallback,
                     'response_tokens': int((response.get('usage') or {}).get('completion_tokens') or 0)}

    def close(self):
        pass


def play_game(index, output, url, payload):
    side = 'white' if index % 2 == 0 else 'black'
    pair = index // 2
    seed = 417000 + pair
    game_id = f'deepseek-final-{index:02d}'
    episode = ChessEpisode(game_id, payload['candidate_id'], side, 2, None,
        output/'games'/game_id, seed=seed, opponent_spec=OPPONENT,
        engine_factory=lambda side, _path, **_kwargs: DeepSeekPlayer(side, game_id, seed))
    try:
        episode.advance()
        while not episode.done:
            fen = episode.board.fen()
            prompt = candidate_messages(payload, fen, episode.board,
                messages(fen, enable_thinking=False))
            request_id = f'{game_id}-student-{len(episode.board.move_stack)}-{time.time_ns()}'
            episode.event('model_request_started', request_id=request_id, messages=prompt)
            response = request_completion(url, PROTOCOL, prompt, timeout=None)
            choice = response['choices'][0]
            content = choice['message'].get('content') or ''
            reasoning = choice['message'].get('reasoning_content') or ''
            parsed = extract_answer(content, fen, reasoning=reasoning,
                                    finish_reason=choice.get('finish_reason'))
            receipt = {'request_id': request_id, 'response': content, 'raw_output': content,
                       'reasoning': reasoning, 'finish_reason': choice.get('finish_reason'),
                       'response_tokens': int((response.get('usage') or {}).get('completion_tokens') or 0),
                       'failure_reason': parsed.error, 'format_valid': parsed.format_valid,
                       'legal': parsed.legal, 'uci': parsed.uci if parsed.legal else None}
            episode.event('model_response', **receipt)
            episode.act(receipt['uci'], [receipt], fallback_reason=parsed.error)
        summary = read_json(episode.output/'summary.json')
        if not summary['completed'] or summary['aborted']:
            raise RuntimeError('Incomplete final-test game')
        return {'game_id': game_id, 'index': index, 'pair_id': pair,
                'seed': seed, 'student_color': side, 'result': summary['result'],
                'plies': summary['plies'], 'reason': summary['reason'],
                'events': str(episode.output/'events.jsonl'),
                'events_sha256': file_hash(episode.output/'events.jsonl'),
                'student_random_moves': summary['random_moves']}
    except Exception as exc:
        if not episode.done:
            episode.abort(str(exc))
        raise
    finally:
        episode.close()


def run(model, candidate, output, *, inference_python=None, games=10, parallel_games=1):
    if type(games) is not int or games < 2 or games % 2:
        raise ValueError('Final test games must be a positive even number, at least 2')
    if type(parallel_games) is not int or not 1 <= parallel_games <= 40:
        raise ValueError('parallel_games must be an integer between 1 and 40')
    concurrency = min(games, parallel_games)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Final test output already exists: {output}')
    if output.name != 'final-test':
        raise ValueError('Use var/runs/<run-id>/final-test as output')
    model, candidate = Path(model).resolve(), Path(candidate).resolve()
    output.mkdir(parents=True)
    skills = candidate/'skills.json'
    harness = candidate/'harness.py'
    source_hashes = {p.name: file_hash(p) for p in (skills, harness) if p.is_file()}
    identity = model_identity(model)
    gpu_ids = visible_gpu_ids()
    if len(gpu_ids) != 1:
        raise RuntimeError('Held-out test requires exactly one allocated GPU')
    payload = {'candidate_id': candidate.name, 'candidate_source': {'root': str(candidate)},
               'research_scope': 'phase1_rl_only',
               'skills_path': str(skills) if skills.is_file() else None}
    atomic_json(output/'protocol.json', {'role': 'held_out_final_test',
        'opponent': {**OPPONENT,**opponent_identity()}, 'student_protocol': PROTOCOL,
        'answer_parser':ANSWER_PARSER, 'opponent_answer_parser':ANSWER_PARSER, 'games': games,
        'white': games//2, 'black': games//2, 'seed_pairs': [417000+i for i in range(games//2)],
        'parallel_games': concurrency,
        'student_vllm_max_num_seqs':max(8,concurrency),
        'student_request_timeout_seconds': None,
        'opponent_request_timeout_seconds': None,
        'student_async_scheduling': False,
        'student_compilation_config': {'cudagraph_mode': 'PIECEWISE'},
        'max_plies': MAX_GAME_PLIES, 'fivefold_repetition': True, 'invalid_move_policy': 'random_legal',
        'student_model_identity': identity, 'candidate': str(candidate),
        'candidate_source_hashes': source_hashes, 'started': utc()})
    port = _pick_ports(8000, 1)[0]
    service = VllmServer(model, PROTOCOL, output/'vllm.log', port, gpu_ids[0],
                         python=inference_python, max_num_seqs=max(8, concurrency), async_scheduling=False,
                         compilation_config={'cudagraph_mode':'PIECEWISE'})
    rows = []
    try:
        service.start()
        error = _wait_servers([service], 900)
        if error:
            raise RuntimeError(error)
        atomic_json(output/'progress.json', {'stage': 'playing', 'completed': 0, 'total': games,
                                            'parallel_games': concurrency, 'updated': utc()})
        # Each game owns its board, RNG and event files. Only this coordinator
        # writes progress and result receipts, so parallel games cannot race on
        # an atomic JSON temporary file.
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending = {pool.submit(play_game, index, output, service.base_url, payload): index
                       for index in range(games)}
            for future in as_completed(pending):
                index = pending[future]
                row = future.result()
                if row.get('index') != index:
                    raise ValueError('Final-test game receipt has the wrong index')
                rows.append(row)
                atomic_json(output/'game_results'/f'{index:02d}.json', row)
                atomic_json(output/'progress.json', {'stage': 'playing', 'completed': len(rows),
                            'total': games, 'parallel_games': concurrency, 'updated': utc()})
    except Exception as error:
        atomic_json(output/'failure.json', {'type': type(error).__name__, 'error': str(error), 'updated': utc()})
        atomic_json(output/'progress.json', {'stage': 'failed', 'completed': len(rows), 'total': games, 'updated': utc()})
        raise
    finally:
        service.stop()
    if model_identity(model)['sha256'] != identity['sha256']:
        raise ValueError('Student model changed during final test')
    if {p.name: file_hash(p) for p in (skills, harness) if p.is_file()} != source_hashes:
        raise ValueError('Candidate prompt or skills changed during final test')
    rows.sort(key=lambda row: row['index'])
    score = sum((row['result'] + 1)/2 for row in rows)/games
    result = {'status': 'completed', 'role': 'held_out_final_test',
              'opponent': opponent_identity(), 'thinking': 'disabled',
              'games': games, 'white': games//2, 'black': games//2,
              'wins': sum(x['result'] == 1 for x in rows),
              'draws': sum(x['result'] == 0 for x in rows),
              'losses': sum(x['result'] == -1 for x in rows),
              'win_rate': sum(x['result'] == 1 for x in rows)/games,
              'score_rate': score, 'rows': rows, 'completed': utc()}
    atomic_json(output/'result.json', result)
    atomic_json(output/'progress.json', {'stage': 'completed', 'completed': games, 'total': games, 'updated': utc()})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--inference-python')
    parser.add_argument('--games', type=int, default=10, help='Even total; half white and half black (default: 10)')
    parser.add_argument('--parallel-games', type=int, default=1, help='Concurrent games on the single vLLM replica (1–40)')
    parser.add_argument('--opponent-model',default=os.environ.get('DEEPSEEK_MODEL','deepseek-v4.1-flash'))
    parser.add_argument('--opponent-base-url',default=os.environ.get('DEEPSEEK_BASE_URL','https://api.deepseek.com'))
    args = parser.parse_args(argv)
    os.environ['DEEPSEEK_MODEL']=args.opponent_model
    os.environ['DEEPSEEK_BASE_URL']=args.opponent_base_url.rstrip('/')
    root = Path('/user/lyh4170')
    for path in (args.model, args.candidate, args.output):
        inside(root, path)
    repo = root/'LLMGo!'
    if args.output.resolve().parent.parent != (repo/'var/runs').resolve():
        raise ValueError('Final test output must be LLMGo!/var/runs/<run-id>/final-test')
    result = run(args.model, args.candidate, args.output,
                 inference_python=args.inference_python, games=args.games, parallel_games=args.parallel_games)
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, ensure_ascii=False))
    return 0
