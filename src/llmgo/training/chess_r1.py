"""Chess-R1 FEN/SAN interface, independent of LLMGo Harness and skills.

Prompt adapted from krafton-ai/Chess-R1 (Apache-2.0), commit 830c0597,
data_preprocess/lichess_quiz.py: qwen_instruct_reastemp_fen_legal_rule.
Only the final answer adds a required LaTeX box. This is not a trainer.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import chess

MAX_RESPONSE_TOKENS = 16384
ANSWER_PROTOCOL = 'chess-r1-boxed-san-v2'
SYSTEM_PROMPT = r"""You are a helpful assistant who plays chess professionally.
The assistant first thinks through the reasoning process internally and then provides the user with the best move.
The reasoning process should be enclosed within <think> </think> tags. After thinking, put the final SAN move in \boxed{...}.
The reasoning process should describe how you analyze the position and decide on the best move, including:
  - A strategic evaluation of the position.
  - A comparison of key candidate moves.
  - For each candidate, consider the opponent's likely response and outcome.
  - Conclude with a clear justification for your final choice.
The answer must be in SAN notation, strictly using the moving piece and the destination square (e.g., Nf3, Rxf2, c5).
Now, the user provides a FEN string, and a list of legal moves for the given board.
After analyzing the position, clearly state the best move in SAN notation inside \boxed{...}. i.e., \boxed{Nf3}

Reminder of chess rules:
- Bishops move diagonally.
- Rooks move horizontally or vertically.
- Knights jump in an L-shape.
- Queens combine rook and bishop movement.
- Kings move one square in any direction.
- Pawns move forward, capture diagonally, and can promote.

Output exactly one SAN move inside \boxed{...} on the last line after thinking.
The box must contain only that SAN move, not UCI, a variation, or an explanation.
Do not put the final answer box inside <think>. Do not use XML answer tags."""

NONTHINKING_SYSTEM_PROMPT = r"""You are a helpful assistant who plays chess professionally.
Given the FEN position and the list of legal moves, choose the best move.
Output exactly one legal SAN move inside \boxed{...}, for example \boxed{Nf3}.
The box must contain only the SAN move; output nothing else."""


def messages(fen, legal_moves_san=None, *, enable_thinking=True):
    board = chess.Board(fen)
    if not board.is_valid() or board.is_game_over():
        raise ValueError('Need a valid nonterminal chess position')
    legal = [board.san(move) for move in board.legal_moves]
    if legal_moves_san is not None:
        supplied = legal_moves_san.split() if isinstance(legal_moves_san, str) else list(legal_moves_san)
        if len(supplied) != len(legal) or set(supplied) != set(legal):
            raise ValueError('Source legal SAN list does not match the board')
        legal = supplied  # Preserve upstream ordering, not a new move ranking.
    system = SYSTEM_PROMPT if enable_thinking else NONTHINKING_SYSTEM_PROMPT
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': f'Current FEN string: {fen}\nLegal moves: {" ".join(legal)}'}]


@dataclass(frozen=True)
class Answer:
    san: str | None = None
    uci: str | None = None
    format_valid: bool = False
    legal: bool = False
    error: str | None = None


def extract_answer(response, fen, *, reasoning='', finish_reason=None):
    """Read only the final answer; never rescue it from a thought or earlier box.

    The generation adapter passes raw generated assistant text (not its prompt).
    Qwen may emit only </think> after an open thinking prefill; API-separated
    reasoning is also supported. The reasoning argument is never searched.
    """
    del reasoning
    if finish_reason in ('length', 'max_tokens'):
        return Answer(error='truncated')
    text = response.strip()
    for suffix in ('<|im_end|>', '<|endoftext|>'):
        text = text.removesuffix(suffix).strip()
    closed_thinking = '</think>' in text
    if closed_thinking:
        text = text.rsplit('</think>', 1)[1].strip()
    if '<think>' in text or '</think>' in text:
        return Answer(error='incomplete_thinking')
    if '<answer>' in text or '</answer>' in text:
        return Answer(error='legacy_answer_tags')
    body = text.strip()
    if len(re.findall(r'\\boxed\s*\{', body)) != 1:
        return Answer(error='missing_or_multiple_boxes')
    start = re.match(r'\\boxed\s*\{', body)
    if not start:
        # The prompt requires a boxed SAN on the last line. A short conclusion
        # after </think> must not hide that answer. Without a closing thinking
        # boundary we cannot distinguish this prefix from unfinished thoughts.
        last_line = body.rsplit('\n', 1)[-1].strip()
        if not closed_thinking or not re.match(r'\\boxed\s*\{', last_line):
            return Answer(error='box_must_be_entire_answer')
        body = last_line
        start = re.match(r'\\boxed\s*\{', body)
    depth, end = 1, start.end()
    while end < len(body) and depth:
        depth += (body[end] == '{') - (body[end] == '}')
        end += 1
    if depth or body[end:].strip():
        return Answer(error='unclosed_box_or_extra_text')
    san = body[start.end():end - 1].strip()
    # Harmless math-font wrappers are allowed, but not arbitrary nested content.
    wrapped = re.fullmatch(r'\\(?:text|mathrm)\s*\{\s*([^{}]+?)\s*\}', san)
    if wrapped:
        san = wrapped[1].strip()
    if not san or any(c.isspace() for c in san) or any(c in san for c in '{}\\'):
        return Answer(error='not_one_san')
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError('Invalid reward FEN')
    # python-chess also accepts long algebraic/UCI in parse_san; exact canonical
    # SAN membership intentionally rejects that cross-protocol shortcut.
    legal_san = {board.san(move): move for move in board.legal_moves}
    if san not in legal_san:
        return Answer(san=san, format_valid=True, error='illegal_or_noncanonical_san')
    return Answer(san=san, uci=legal_san[san].uci(), format_valid=True, legal=True)
