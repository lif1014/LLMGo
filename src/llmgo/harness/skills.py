"""Conditionally retrieve short, inspectable cards. No move lookup at runtime."""
from functools import lru_cache
import json
from pathlib import Path
import chess
from llmgo.harness.tools import material_value

@lru_cache(maxsize=1)
def load_cards() -> tuple[dict, ...]:
    cards = json.loads((Path(__file__).with_name('skills') / 'cards.json').read_text(encoding='utf-8'))
    return validate_cards(cards)


TRIGGERS = {'in_check', 'checks_available', 'captures_available', 'opening', 'inactive_rook', 'endgame',
            'candidate_allows_mate', 'opening_white_centre', 'opening_black_reply', 'opening_open_game',
            'opening_sicilian', 'opening_caro_kann', 'opening_queen_pawn'}


def validate_cards(cards) -> tuple[dict, ...]:
    if not isinstance(cards, (list, tuple)) or len(cards) > 64:
        raise ValueError('Skills must be an array with at most 64 cards')
    seen = set()
    for card in cards:
        if not isinstance(card, dict):
            raise ValueError('Every skill must be an object')
        if any(not isinstance(card.get(key), str) or not card[key].strip()
               for key in ('id', 'trigger', 'title', 'when', 'rule', 'counterexample')):
            raise ValueError('Skill identifiers, trigger and prose must be nonempty strings')
        if card['id'] in seen or card['trigger'] not in TRIGGERS:
            raise ValueError('Invalid or duplicate skill: ' + card['id'])
        for field in ('title', 'when', 'rule', 'counterexample', 'sources', 'stages'):
            if not card.get(field): raise ValueError(f'Missing {field} in {card["id"]}')
        if type(card.get('priority')) is not int or not 0 <= card['priority'] <= 100:
            raise ValueError('Skill priority must be an integer from 0 to 100')
        if card['stages'] != ['move']:
            raise ValueError('Only the complete-move stage is supported')
        if not isinstance(card['sources'], list) or any(not isinstance(s, str) or not s.strip() for s in card['sources']):
            raise ValueError('Skill sources must be a nonempty list of references')
        if any(not isinstance(card[key], str) for key in ('id', 'title', 'when', 'rule', 'counterexample')):
            raise ValueError('Skill prose fields must be strings')
        if len('\n'.join(card[key] for key in ('title', 'when', 'rule', 'counterexample'))) > 2200:
            raise ValueError('Skill card exceeds fixed prose budget')
        seen.add(card['id'])
    return tuple(cards)

def _opening_signals(board: chess.Board) -> dict[str, bool]:
    """Broad early pawn structures select prose knowledge, never a book move."""
    def has(square, kind, color):
        return board.piece_at(square) == chess.Piece(kind, color)
    early = (board.fullmove_number <= 6 and not board.is_check()
             and has(chess.E1, chess.KING, chess.WHITE) and has(chess.E8, chess.KING, chess.BLACK)
             and has(chess.D1, chess.QUEEN, chess.WHITE) and has(chess.D8, chess.QUEEN, chess.BLACK))
    white_e4 = has(chess.E4, chess.PAWN, chess.WHITE)
    white_d4 = has(chess.D4, chess.PAWN, chess.WHITE)
    black_e5 = has(chess.E5, chess.PAWN, chess.BLACK)
    return {
        'opening_white_centre': early and board.turn == chess.WHITE and board.fullmove_number <= 3
            and has(chess.E2, chess.PAWN, chess.WHITE) and has(chess.D2, chess.PAWN, chess.WHITE),
        'opening_black_reply': early and board.turn == chess.BLACK and board.fullmove_number <= 3
            and (white_e4 or white_d4) and has(chess.E7, chess.PAWN, chess.BLACK) and has(chess.D7, chess.PAWN, chess.BLACK)
            and not has(chess.C6, chess.PAWN, chess.BLACK) and not has(chess.C5, chess.PAWN, chess.BLACK),
        'opening_open_game': early and white_e4 and black_e5,
        'opening_sicilian': early and board.turn == chess.WHITE and white_e4 and not black_e5
            and (has(chess.C5, chess.PAWN, chess.BLACK) or
                 (has(chess.D4, chess.PAWN, chess.BLACK) and not has(chess.C7, chess.PAWN, chess.BLACK))),
        'opening_caro_kann': early and board.turn == chess.WHITE and not black_e5
            and has(chess.C6, chess.PAWN, chess.BLACK)
            and (white_e4 or (white_d4 and has(chess.E4, chess.PAWN, chess.BLACK)
                             and not has(chess.D7, chess.PAWN, chess.BLACK))),
        'opening_queen_pawn': early and white_d4 and has(chess.D5, chess.PAWN, chess.BLACK)
            and not (white_e4 and black_e5),
    }

def select_skills(board: chess.Board, stage: str = 'move', limit: int = 2, *, candidates=None, cards=None) -> list[dict]:
    legal = list(board.legal_moves)
    opening = board.fullmove_number <= 12
    endgame = not board.queens and sum(material_value(board, c) - 100 * len(board.pieces(chess.PAWN, c)) for c in chess.COLORS) <= 2000
    back = chess.BB_RANK_1 if board.turn else chess.BB_RANK_8
    signals = {'in_check': board.is_check(), 'checks_available': any(board.gives_check(m) for m in legal),
               'captures_available': any(board.is_capture(m) for m in legal),
               'opening': opening and bool((board.knights | board.bishops) & board.occupied_co[board.turn] & back),
               'inactive_rook': not opening and not endgame and bool(board.pieces_mask(chess.ROOK, board.turn) & back), 'endgame': endgame,
               'candidate_allows_mate': False,
               **_opening_signals(board)}
    if candidates is not None:
        signals['candidate_allows_mate'] = any(item['preview']['opponent_mate_in_one'] for item in candidates)
    cards = [card for card in (load_cards() if cards is None else cards)
             if stage in card['stages'] and signals[card['trigger']]]
    if any(card['id'].startswith('opening-') for card in cards):
        cards = [card for card in cards if card['id'] != 'development']
    return sorted(cards, key=lambda card: (-card['priority'], card['id']))[:limit]

def skill_lines(cards: list[dict]) -> list[str]:
    lines = []
    for card in cards:
        lines.extend([f"{card['id']}: {card['title']}", 'Trigger: ' + card['when'], 'Action: ' + card['rule'], 'Counterexample: ' + card['counterexample']])
    return lines
