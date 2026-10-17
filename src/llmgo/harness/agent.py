"""Thinking-enabled tool -> piece -> move decisions with complete request traces."""
from __future__ import annotations
import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable
import chess
from llmgo.harness.choices import LETTER_TOOL, available_tools, blackboard, option_labels, parse_choice
from llmgo.harness.config import HarnessConfig
from llmgo.harness.prompt import move_prompt, pick_prompt, played_lines, preview_prompt, render_board, side_name, tool_letter, tool_prompt
from llmgo.harness.skills import select_skills
from llmgo.harness.tools import TOOL_LIBRARY, move_label, movable_pieces, preview_lines, preview_move, tool_result

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
    notes, outputs, requests, recoveries = {}, [], [], []
    stage_spent = {stage: 0 for stage in ('tool', 'preview', 'piece', 'move')}
    spent, request_no = 0, 0
    started = time.monotonic()
    last_prompt, last_stage = '', 'tool'
    board_text, side = render_board(board), side_name(board.turn)
    played = played_lines(history, cfg.history_plies)
    turn_id = f'{len(history)}:{seat}'
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError('Cannot choose a move from a terminal position')

    async def send(event):
        await emit({'turn_id': turn_id, 'ply': len(history), **event})

    def diagnostics(reason=None):
        return {'request_count': len(requests), 'completion_tokens': spent,
                'tokens_exact': all(r['tokens_exact'] for r in requests),
                'prompt_tokens': sum(int(r['usage'].get('prompt_tokens') or 0) for r in requests),
                'latency_ms': (time.monotonic() - started) * 1000,
                'finish_reasons': [r['finish_reason'] for r in requests],
                'length_count': sum(r['finish_reason'] == 'length' for r in requests),
                'request_errors': sum(bool(r['error']) for r in requests),
                'fallback_reason': reason, 'fallback_stage': last_stage if reason else None,
                'recoveries': recoveries, 'stage_tokens': dict(stage_spent), 'config': cfg.as_dict()}

    async def fallback(moves, reason):
        chosen = picker.choice(moves)
        scope = 'selected_piece' if last_stage == 'move' else 'all_legal_moves'
        await send({'type': 'fallback', 'seat': seat, 'stage': last_stage, 'reason': reason,
                    'scope': scope, 'uci': chosen.uci(), 'budget_spent': spent, 'budget_total': budget})
        await send(_choice(seat, 'move', 'answered', last_prompt, [], blackboard(notes),
                           uci=chosen.uci(), source='random', fallback_reason=reason,
                           origins=[chess.square_name(chosen.from_square)], targets=[chess.square_name(chosen.to_square)]))
        return Decision(chosen.uci(), 'random', list(notes), spent, last_prompt,
                        '\n\n'.join(outputs), diagnostics(reason))

    async def spend(stage, prompt, allowed):
        nonlocal spent, request_no, last_prompt, last_stage
        last_prompt, last_stage = prompt, stage
        reserve = 0
        if stage in {'tool', 'preview'}:
            reserve = min(cfg.piece_tokens, budget // 4) + min(cfg.move_tokens, budget // 4)
        elif stage == 'piece':
            reserve = min(cfg.move_tokens, budget // 4)
        stage_left = min(getattr(cfg, stage + '_tokens') - stage_spent[stage], max(0, budget - spent - reserve))
        last_reason = 'invalid_choice'
        for attempt in range(cfg.attempts_per_choice):
            if time.monotonic() - started >= cfg.turn_timeout_sec:
                return 'turn_timeout', None
            cap = min(stage_left, budget - spent - reserve)
            if cap <= 0:
                return ('stage_budget_exhausted' if stage_left <= 0 else 'turn_budget_reserved'), None
            request_no += 1
            request_id = f'{turn_id}:{request_no}'
            build_payload = getattr(ask, 'build_payload', None)
            payload = build_payload(prompt, cap) if build_payload else {}
            await send({'type': 'model_request', 'status': 'started', 'seat': seat, 'stage': stage,
                        'request_id': request_id, 'attempt_index': attempt, 'prompt': prompt,
                        'allowed_choices': sorted(allowed), 'max_tokens': cap, 'enable_thinking': True,
                        'budget_spent': spent, 'budget_total': budget, 'reserved_tokens': reserve,
                        'request_payload': payload})
            tick = time.monotonic()
            timeout = min(cfg.request_timeout_sec, max(0.001, cfg.turn_timeout_sec - (tick - started)))
            try:
                result = await asyncio.wait_for(ask(prompt, cap), timeout=timeout)
            except TimeoutError:
                result = AskResult('', 0, error='request_timeout', tokens_exact=False)
            except Exception as exc:
                result = AskResult('', 0, error=f'{type(exc).__name__}: {exc}', tokens_exact=False)
            used = max(0, result.tokens)
            spent += used
            stage_left -= used
            stage_spent[stage] += used
            shown = result.output or result.reasoning or result.text
            if shown: outputs.append(shown)
            letter = parse_choice(result.text, allowed) if not result.error else None
            timeout_error = result.error and any(name in result.error for name in ('request_timeout', 'ReadTimeout', 'ConnectTimeout', 'WriteTimeout', 'PoolTimeout'))
            reason = None if letter else ('request_timeout' if timeout_error else
                     'request_error' if result.error else 'length_without_choice' if result.finish_reason == 'length' else 'invalid_choice')
            record = {'type': 'model_request', 'status': 'finished', 'seat': seat, 'stage': stage,
                      'request_id': request_id, 'attempt_index': attempt, 'prompt': prompt,
                      'allowed_choices': sorted(allowed), 'max_tokens': cap, 'enable_thinking': True,
                      'text': result.text, 'reasoning': result.reasoning, 'output': shown,
                      'raw_response': result.raw_response, 'request_payload': result.request_payload or payload,
                      'http_status': result.http_status, 'usage': result.usage,
                      'completion_tokens': used, 'tokens_exact': result.tokens_exact,
                      'latency_ms': (time.monotonic() - tick) * 1000,
                      'finish_reason': result.finish_reason, 'error': result.error,
                      'letter': letter, 'accepted': letter is not None, 'failure_reason': reason,
                      'budget_spent': spent, 'budget_total': budget}
            requests.append(record)
            await send(record)
            if letter: return 'ok', letter
            last_reason = reason
            if result.error or result.finish_reason == 'length':
                return reason, None
        return last_reason, None

    def cards(stage):
        return select_skills(board, stage, cfg.skills_limit)

    await send({'type': 'turn_started', 'seat': seat, 'attempt_index': 0, 'prompt': '',
                'tool_library': TOOL_LIBRARY, 'harness_config': cfg.as_dict(), 'budget_total': budget,
                'enable_thinking': True})
    while len(notes) < cfg.max_lookups:
        names = available_tools(set(notes))
        prompt = tool_prompt(side, board_text, played, blackboard(notes), set(notes), max_lookups=cfg.max_lookups)
        await send(_choice(seat, 'tool', 'asking', prompt, names, blackboard(notes)))
        status, letter = await spend('tool', prompt, {tool_letter(name) for name in names})
        if status != 'ok':
            recoveries.append({'stage': 'tool', 'reason': status, 'action': 'continue_to_piece'})
            await send({'type': 'route_recovery', 'seat': seat, **recoveries[-1]})
            break
        name = LETTER_TOOL[letter]
        await send(_choice(seat, 'tool', 'answered', prompt, names, blackboard(notes), letter=letter, tool=name))
        if name == 'piece': break
        parameters, result_data = {}, None
        if name == 'preview':
            labels = option_labels(len(legal))
            options = [(label, move_label(board, move)) for label, move in zip(labels, legal)]
            chosen_cards = cards('preview')
            prompt = preview_prompt(side, board_text, played, blackboard(notes), options, skills=chosen_cards)
            await send(_choice(seat, 'preview', 'asking', prompt, [], blackboard(notes), options=options, skills=chosen_cards))
            status, letter = await spend('preview', prompt, set(labels))
            if status != 'ok':
                recoveries.append({'stage': 'preview', 'reason': status, 'action': 'continue_to_piece'})
                await send({'type': 'route_recovery', 'seat': seat, **recoveries[-1]})
                break
            candidate = legal[labels.index(letter)]
            parameters = {'uci': candidate.uci()}
            await send(_choice(seat, 'preview', 'answered', prompt, [], blackboard(notes), options=options, letter=letter, uci=candidate.uci()))
        tick = time.monotonic()
        call_id = f'{turn_id}:tool:{name}'
        await send({'type': 'tool_call', 'status': 'started', 'seat': seat, 'tool': name, 'parameters': parameters, 'call_id': call_id})
        try:
            if name == 'preview':
                result_data = preview_move(board, candidate)
                lines = preview_lines(result_data)
            else:
                lines = tool_result(name, board)
        except Exception as exc:
            await send({'type': 'tool_call', 'status': 'finished', 'seat': seat, 'tool': name,
                        'parameters': parameters, 'call_id': call_id, 'error': f'{type(exc).__name__}: {exc}',
                        'latency_ms': (time.monotonic() - tick) * 1000})
            return await fallback(legal, 'tool_error')
        notes[name] = lines
        await send({'type': 'tool_call', 'status': 'finished', 'seat': seat, 'tool': name,
                    'parameters': parameters, 'call_id': call_id, 'result': lines, 'data': result_data,
                    'latency_ms': (time.monotonic() - tick) * 1000, 'blackboard': blackboard(notes)})
    else:
        await send(_choice(seat, 'tool', 'answered', '', ['piece'], blackboard(notes), letter='E', tool='piece', source='automatic'))

    pieces = movable_pieces(board)
    labels = option_labels(len(pieces))
    options = [(label, piece[1]) for label, piece in zip(labels, pieces)]
    chosen_cards = cards('piece')
    prompt = pick_prompt(side, board_text, played, blackboard(notes), options, skills=chosen_cards)
    last_prompt, last_stage = prompt, 'piece'
    await send(_choice(seat, 'piece', 'asking', prompt, [], blackboard(notes), options=options, skills=chosen_cards,
                       origins=[chess.square_name(piece[0]) for piece in pieces]))
    if len(labels) == 1:
        letter = labels[0]
        automatic = True
    else:
        automatic = False
        status, letter = await spend('piece', prompt, set(labels))
        if status != 'ok': return await fallback(legal, status)
    square, piece_name, moves = pieces[labels.index(letter)]
    await send(_choice(seat, 'piece', 'answered', prompt, [], blackboard(notes), options=options, letter=letter,
                       origins=[chess.square_name(square)], source='automatic' if automatic else 'model'))
    labels = option_labels(len(moves))
    options = [(label, move_label(board, move)) for label, move in zip(labels, moves)]
    chosen_cards = cards('move')
    prompt = move_prompt(side, board_text, played, blackboard(notes), piece_name, options, skills=chosen_cards)
    last_prompt, last_stage = prompt, 'move'
    await send(_choice(seat, 'move', 'asking', prompt, [], blackboard(notes), options=options, skills=chosen_cards,
                       origins=[chess.square_name(square)], targets=[chess.square_name(move.to_square) for move in moves]))
    if len(labels) == 1:
        letter = labels[0]
        automatic = True
    else:
        automatic = False
        status, letter = await spend('move', prompt, set(labels))
        if status != 'ok': return await fallback(moves, status)
    chosen = moves[labels.index(letter)]
    await send(_choice(seat, 'move', 'answered', prompt, [], blackboard(notes), options=options, letter=letter,
                       uci=chosen.uci(), source='automatic' if automatic else 'model',
                       origins=[chess.square_name(square)], targets=[chess.square_name(chosen.to_square)]))
    # Automatic forced choices still form part of the model's turn, never a random fallback.
    return Decision(chosen.uci(), 'model', list(notes), spent, prompt, '\n\n'.join(outputs), diagnostics())

def _choice(seat, stage, status, prompt, tools, result, *, letter=None, tool=None,
            origins=None, targets=None, uci=None, source=None, options=None, output='', **extra):
    if options is None and tools: options = [(tool_letter(name), name) for name in tools]
    return {'type': 'choice', 'seat': seat, 'stage': stage, 'status': status, 'prompt': prompt,
            'options': [{'letter': a, 'label': b} for a, b in (options or [])], 'letter': letter,
            'tool': tool, 'blackboard': result, 'origins': origins or [], 'targets': targets or [],
            'uci': uci, 'source': source, 'output': output, **extra}
