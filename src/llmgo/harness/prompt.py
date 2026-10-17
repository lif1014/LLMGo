"""Short multiple-choice cards with explicit coordinates and conditional skills."""
from __future__ import annotations
import chess
from llmgo.harness.choices import TOOL_LETTER, available_tools
from llmgo.harness.skills import skill_lines
from llmgo.harness.tools import TOOL_LIBRARY

_HELP = {item['name']: [item['letter'] + ' ' + item['name'], 'reply ' + item['letter'], item['description']] for item in TOOL_LIBRARY}
_HELP['legal'] += ['captures means a move that takes an enemy piece, including en passant', 'checks means a move that attacks the enemy king', 'quiet means any other legal move', 'A move can be both a capture and a check. UCI e7e8q promotes; e1g1 is kingside castling.']

def side_name(color: chess.Color) -> str:
    return 'White' if color else 'Black'

def render_board(board: chess.Board) -> str:
    rows = ['Uppercase White, lowercase Black; a1 bottom-left; White pawns go to higher ranks.']
    for rank in range(7, -1, -1):
        rows.append(f'{rank + 1} ' + ' '.join(board.piece_at(chess.square(file, rank)).symbol() if board.piece_at(chess.square(file, rank)) else '.' for file in range(8)))
    rows.append('  a b c d e f g h')
    for color in (chess.WHITE, chess.BLACK):
        rows.append(side_name(color) + ' pieces: ' + ', '.join(f'{chess.piece_name(p.piece_type)} {chess.square_name(sq)}' for sq, p in sorted(board.piece_map().items()) if p.color == color))
    rows.extend(['FEN ' + board.fen(), 'In check: ' + ('yes' if board.is_check() else 'no')])
    return '\n'.join(rows)

def played_lines(history: list[dict], limit: int = 12) -> list[str]:
    start = max(0, len(history) - limit) if limit else len(history)
    return [f"ply {i + 1} {history[i].get('seat', 'white' if i % 2 == 0 else 'black')} {history[i]['uci']}" for i in range(start, len(history))]

def _card(side, board_text, played, result, blocks, skills=()):
    parts = ['Goal\nPlay standard chess to win as ' + side + '. Compare the opponent response before committing. Select only from this question. A lookup or preview does not play a move.', 'Side\n' + side]
    if played: parts.append('Recent played plies\n' + '\n'.join(played))
    parts.append('Board\n' + board_text)
    if result: parts.append('Result\n' + '\n'.join(result))
    if skills: parts.append('Relevant skills\n' + '\n'.join(skill_lines(list(skills))))
    parts.extend(heading + '\n' + '\n'.join(lines) for heading, lines in blocks)
    parts.append('Reply\none letter')
    return '\n\n'.join(parts)

def tool_prompt(side, board_text, played, result, used, *, skills=(), max_lookups=2):
    left = max(0, max_lookups - len(used))
    lines = [f'you can look up {left} more tools on this turn', 'choosing E does not use a lookup']
    names = available_tools(used) if left else ['piece']
    for name in names: lines.extend(_HELP[name])
    return _card(side, board_text, played, result, [('Tools', lines)], skills)

def pick_prompt(side, board_text, played, result, options, *, skills=()):
    lines = ['reply one letter to pick up that piece'] + [f'{letter} {label}' for letter, label in options]
    return _card(side, board_text, played, result, [('Pick', lines)], skills)

def move_prompt(side, board_text, played, result, piece_label, options, *, skills=()):
    lines = ['reply one letter to play that move'] + [f'{letter} {label}' for letter, label in options]
    return _card(side, board_text, played, result, [('Piece', [piece_label]), ('Moves', lines)], skills)

def preview_prompt(side, board_text, played, result, options, *, skills=()):
    lines = ['reply one letter to inspect this candidate without playing it'] + [f'{letter} {label}' for letter, label in options]
    return _card(side, board_text, played, result, [('Preview candidate', lines)], skills)

def tool_letter(name: str) -> str:
    return TOOL_LETTER[name]
