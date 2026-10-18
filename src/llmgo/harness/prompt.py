"""A single complete-move question with compact coordinate-based context."""
from __future__ import annotations
import chess
from llmgo.harness.skills import skill_lines


def side_name(color: chess.Color) -> str:
    return 'White' if color else 'Black'


def render_board(board: chess.Board) -> str:
    rows = []
    for color in (chess.WHITE, chess.BLACK):
        rows.append(side_name(color) + ' pieces: ' + '; '.join(
            chess.piece_name(kind) + ' ' + ' '.join(chess.square_name(square) for square in board.pieces(kind, color))
            for kind in (chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)
            if board.pieces(kind, color)))
    rows.append('In check: ' + ('yes' if board.is_check() else 'no'))
    rows.append('White pawns move toward rank 8; Black pawns toward rank 1.')
    return '\n'.join(rows)


def played_lines(history: list[dict], limit: int = 12) -> list[str]:
    start = max(0, len(history) - limit) if limit else len(history)
    return [f"ply {i + 1} {history[i].get('seat', 'white' if i % 2 == 0 else 'black')} {history[i]['uci']}" for i in range(start, len(history))]


def move_prompt(board: chess.Board, history: list[dict], candidates: list[dict], observations: list[str], *, skills=(), history_plies=12, correction=None):
    side = side_name(board.turn)
    parts = [f'You play {side} in a normal chess game. Choose the best practical legal move; a forced win may not exist. '
             'Briefly compare a few promising moves, check the opponent reply, then decide. '
             'UCI = source square + target square (+ q/r/b/n for promotion).',
             'Position\n' + render_board(board)]
    played = played_lines(history, history_plies)
    if played:
        parts.append('Recent moves\n' + '\n'.join(played))
    if observations:
        parts.append('One-reply checks: unlisted moves are NOT proven safe. Negative immediate delta may be recovered by your next recapture; it is not a forced loss.\n' + '\n'.join(observations))
    if skills:
        parts.append('Relevant skills (general advice, not a hint to a unique correct answer)\n' + '\n'.join(skill_lines(list(skills))))
    groups = {}
    for item in candidates:
        key = item['piece'] + ' ' + item['from']
        groups.setdefault(key, []).append(item['uci'])
    lines = [key + ': ' + ' '.join(labels) for key, labels in groups.items()]
    parts.append(f'Legal moves ({len(candidates)} total)\n' + '\n'.join(lines))
    if correction:
        parts.append('Correction\n' + correction)
    parts.append('Final answer: output exactly one UCI from Legal moves. No explanation or JSON.')
    return '\n\n'.join(parts)
