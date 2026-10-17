"""One thinking-enabled choice over all legal moves, with complete traces."""
from __future__ import annotations
import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable
import chess
from llmgo.harness.choices import parse_move, split_thinking
from llmgo.harness.config import HarnessConfig
from llmgo.harness.prompt import move_prompt
from llmgo.harness.skills import select_skills
from llmgo.harness.tools import TOOL_LIBRARY, king_line, material_line, prepare_candidates

@dataclass
class AskResult:
    text: str
    tokens: int
    finish_reason: str | None = None
    error: str | None = None
    output: str = ''
    reasoning: str = ''
    usage: dict = field(default_factory=dict)
    tokens_exact: bool = True
    raw_response: str = ''
    request_payload: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    http_status: int | None = None
    normalization_error: str | None = None

@dataclass
class Decision:
    uci: str
    source: str
    tools: list[str] = field(default_factory=list)
    tokens: int = 0
    prompt: str = ''
    reasoning: str = ''
    diagnostics: dict = field(default_factory=dict)

Ask = Callable[[str, int], Awaitable[AskResult]]
Emit = Callable[[dict], Awaitable[None]]

async def choose_move(board, history, ask, emit, *, seat, budget=32768, rng=None, config=None) -> Decision:
    cfg = config or HarnessConfig.from_dict()
    picker = rng or random.Random()
    started = time.monotonic()
    turn_id = f'{len(history)}:{seat}'
    legal = sorted(board.legal_moves, key=lambda move: move.uci())
    if not legal:
        raise ValueError('Cannot choose a move from a terminal position')
    allowed = {move.uci() for move in legal}
    spent, requests, tools = 0, [], []
    prompt, observations = '', []
    allowance = min(budget, cfg.move_tokens)

    async def send(event):
        await emit({'turn_id': turn_id, 'ply': len(history), 'seat': seat, **event})

    def diagnostics(reason=None):
        return {'request_count': len(requests), 'completion_tokens': spent,
                'tokens_exact': all(item['tokens_exact'] for item in requests),
                'prompt_tokens': sum(int(item['usage'].get('prompt_tokens') or 0) for item in requests),
                'latency_ms': (time.monotonic() - started) * 1000,
                'finish_reasons': [item['finish_reason'] for item in requests],
                'length_count': sum(item['finish_reason'] == 'length' for item in requests),
                'request_errors': sum(bool(item['error']) for item in requests),
                'fallback_reason': reason, 'fallback_stage': 'move' if reason else None,
                'recoveries': [], 'stage_tokens': {'move': spent}, 'config': cfg.as_dict()}

    def choice(status, **extra):
        return {'type': 'choice', 'stage': 'move', 'status': status, 'prompt': prompt,
                'options': [{'uci': item['uci'], 'label': item['label']} for item in candidates],
                'blackboard': observations, 'skills': cards, **extra}

    async def fallback(reason):
        move = picker.choice(legal)
        await send({'type': 'fallback', 'stage': 'move', 'reason': reason,
                    'scope': 'all_legal_moves', 'uci': move.uci(), 'budget_spent': spent, 'budget_total': budget})
        await send(choice('answered', uci=move.uci(), source='random', fallback_reason=reason,
                          origins=[chess.square_name(move.from_square)], targets=[chess.square_name(move.to_square)]))
        return Decision(move.uci(), 'random', tools, spent, prompt,
                        '\n\n'.join(item['output'] for item in requests if item['output']), diagnostics(reason))

    await send({'type': 'turn_started', 'attempt_index': 0, 'prompt': '', 'framework': 'single_choice',
                'fen': board.fen(), 'tool_library': TOOL_LIBRARY, 'harness_config': cfg.as_dict(),
                'budget_total': budget, 'enable_thinking': True})
    candidates, cards = [], []
    for name in ('position', 'candidates'):
        call_id = f'{turn_id}:tool:{name}'
        parameters = {'fen': board.fen(), 'mode': 'automatic'}
        tick = time.monotonic()
        await send({'type': 'tool_call', 'status': 'started', 'tool': name, 'parameters': parameters, 'call_id': call_id})
        try:
            if name == 'position':
                lines, data = king_line(board) + material_line(board), {'fen': board.fen()}
            else:
                candidates, observations = prepare_candidates(board)
                if {item['uci'] for item in candidates} != allowed or len(candidates) != len(allowed):
                    raise ValueError('Candidate list must contain every legal move exactly once')
                lines, data = observations, {'candidates': candidates, 'legal_count': len(candidates)}
        except Exception as exc:
            await send({'type': 'tool_call', 'status': 'finished', 'tool': name, 'parameters': parameters,
                        'call_id': call_id, 'error': f'{type(exc).__name__}: {exc}',
                        'latency_ms': (time.monotonic() - tick) * 1000})
            return await fallback('tool_error')
        tools.append(name)
        await send({'type': 'tool_call', 'status': 'finished', 'tool': name, 'parameters': parameters,
                    'call_id': call_id, 'result': lines, 'data': data, 'blackboard': observations,
                    'latency_ms': (time.monotonic() - tick) * 1000})
    cards = select_skills(board, 'move', cfg.skills_limit, candidates=candidates)
    correction = None
    reason = 'invalid_choice'
    for attempt in range(cfg.attempts_per_choice):
        prompt = move_prompt(board, history, candidates, observations, skills=cards,
                             history_plies=cfg.history_plies, correction=correction)
        await send(choice('asking', attempt_index=attempt,
                          origins=sorted({item['from'] for item in candidates}),
                          targets=sorted({item['to'] for item in candidates})))
        if len(legal) == 1:
            move = legal[0]
            await send(choice('answered', uci=move.uci(), source='forced',
                              origins=[chess.square_name(move.from_square)], targets=[chess.square_name(move.to_square)]))
            return Decision(move.uci(), 'forced', tools, 0, prompt, '', diagnostics())
        remaining_time = cfg.turn_timeout_sec - (time.monotonic() - started)
        if remaining_time <= 0:
            return await fallback('turn_timeout')
        cap = allowance - spent
        if cap <= 0:
            return await fallback('turn_budget_exhausted')
        request_id = f'{turn_id}:{attempt + 1}'
        payload_builder = getattr(ask, 'build_payload', None)
        payload = payload_builder(prompt, cap) if payload_builder else {}
        base = {'type': 'model_request', 'stage': 'move', 'request_id': request_id, 'attempt_index': attempt,
                'prompt': prompt, 'allowed_choices': sorted(allowed), 'max_tokens': cap,
                'enable_thinking': True, 'budget_total': budget}
        await send({**base, 'status': 'started', 'budget_spent': spent, 'request_payload': payload})
        tick = time.monotonic()
        try:
            result = await asyncio.wait_for(ask(prompt, cap), timeout=min(cfg.request_timeout_sec, remaining_time))
        except TimeoutError:
            result = AskResult('', 0, error='request_timeout', tokens_exact=False)
        except Exception as exc:
            result = AskResult('', 0, error=f'{type(exc).__name__}: {exc}', tokens_exact=False)
        used = max(0, result.tokens)
        spent += used
        final, reasoning, normalization_error = split_thinking(result.text, result.reasoning)
        normalization_error = result.normalization_error or normalization_error
        uci = parse_move(final, allowed) if not (result.error or normalization_error or result.finish_reason == 'length') else None
        timeout_error = result.error and any(name in result.error for name in ('request_timeout', 'ReadTimeout', 'ConnectTimeout', 'WriteTimeout', 'PoolTimeout'))
        reason = None if uci else ('request_timeout' if timeout_error else
                  'request_error' if result.error else 'length_without_choice' if result.finish_reason == 'length' else
                  normalization_error or 'invalid_choice')
        record = {**base, 'status': 'finished', 'text': final, 'reasoning': reasoning,
                  'output': result.output or '\n\n'.join(part for part in (reasoning, final) if part),
                  'raw_response': result.raw_response, 'request_payload': result.request_payload or payload,
                  'http_status': result.http_status, 'usage': result.usage, 'completion_tokens': used,
                  'tokens_exact': result.tokens_exact, 'latency_ms': (time.monotonic() - tick) * 1000,
                  'finish_reason': result.finish_reason, 'error': result.error, 'normalization_error': normalization_error,
                  'uci': uci, 'accepted': uci is not None, 'failure_reason': reason, 'budget_spent': spent}
        requests.append(record)
        await send(record)
        if uci:
            move = chess.Move.from_uci(uci)
            await send(choice('answered', uci=uci, source='model',
                              origins=[chess.square_name(move.from_square)], targets=[chess.square_name(move.to_square)]))
            return Decision(uci, 'model', tools, spent, prompt,
                            '\n\n'.join(item['output'] for item in requests if item['output']), diagnostics())
        if result.error or result.finish_reason == 'length':
            return await fallback(reason)
        correction = ('The previous final answer was not accepted (' + reason + '). '
                      'Return only {"move":"<one exact UCI from Legal moves>"}; do not return a piece, option letter, or several moves.')
    return await fallback(reason)
