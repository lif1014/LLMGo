"""A single complete-move question with compact coordinate-based context."""
from __future__ import annotations
import chess
from llmgo.harness.skills import skill_lines


def side_name(color: chess.Color) -> str:
    return 'White' if color else 'Black'


def render_board(board: chess.Board) -> str:
    rows = []
    for color in (chess.WHITE, chess.BLACK):
        rows.append(side_name(color) + ' pieces: ' + ', '.join(
            f'{chess.piece_name(piece.piece_type)} {chess.square_name(square)}'
            for square, piece in sorted(board.piece_map().items()) if piece.color == color))
    rows.append('In check: ' + ('yes' if board.is_check() else 'no'))
    rows.append('White pawns move toward rank 8; Black pawns toward rank 1.')
    return '\n'.join(rows)


def played_lines(history: list[dict], limit: int = 12) -> list[str]:
    start = max(0, len(history) - limit) if limit else len(history)
    return [f"ply {i + 1} {history[i].get('seat', 'white' if i % 2 == 0 else 'black')} {history[i]['uci']}" for i in range(start, len(history))]


def move_prompt(board: chess.Board, history: list[dict], candidates: list[dict], observations: list[str], *, skills=(), history_plies=12, correction=None):
    side = side_name(board.turn)
    parts = [f'You are {side}. Choose the strongest complete move from the legal list below to win this chess game. '
             'Consider the opponent reply. Every listed option is legal; observations cover only one opponent reply and do not prove safety.',
             'Position\n' + render_board(board)]
    played = played_lines(history, history_plies)
    if played:
        parts.append('Recent moves\n' + '\n'.join(played))
    if observations:
        parts.append('Candidate observations\n' + '\n'.join(observations))
    if skills:
        parts.append('Relevant skills\n' + '\n'.join(skill_lines(list(skills))))
    groups = {}
    for item in candidates:
        key = item['piece'] + ' ' + item['from']
        groups.setdefault(key, []).append(item['label'])
    lines = [key + ': ' + ', '.join(labels) for key, labels in groups.items()]
    parts.append(f'Legal moves ({len(candidates)} total; UCI with SAN in parentheses)\n' + '\n'.join(lines))
    if correction:
        parts.append('Correction\n' + correction)
    parts.append('Final answer: return only {"move":"YOUR_CHOSEN_UCI"}. '
                 'Replace YOUR_CHOSEN_UCI with one exact UCI string from Legal moves.')
    return '\n\n'.join(parts)
