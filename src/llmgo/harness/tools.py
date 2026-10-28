"""Finite chess-rule observations: no engine, opening book or tablebase."""
from __future__ import annotations
import chess

from llmgo.env.rules import chess_outcome

VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
NAMES = {kind: chess.piece_name(kind) for kind in VALUES}
TOOL_LIBRARY = [
    {"name": "position", "mode": "automatic", "description": "Compute in-check status, castling rights/current legality and material. Full details stay in the trace."},
    {"name": "candidates", "mode": "automatic", "description": "Enumerate ALL legal moves in fixed piece/source/target order. Preview each for legal captures of the moved piece, one-reply material changes, opponent mate-in-one and terminal outcomes. No engine, ranking or pruning."},
]

def _uci_list(moves) -> str:
    return " ".join(sorted(move.uci() for move in moves)) or "none"

def captured_square(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return move.to_square + (-8 if board.turn == chess.WHITE else 8)
    return move.to_square

def captured_value(board: chess.Board, move: chess.Move) -> int:
    victim = board.piece_at(captured_square(board, move))
    return VALUES[victim.piece_type] if victim and board.is_capture(move) else 0

def material_value(board: chess.Board, color: chess.Color) -> int:
    return sum(VALUES[kind] * len(board.pieces(kind, color)) for kind in VALUES)

def material_balance(board: chess.Board, color: chess.Color) -> int:
    return material_value(board, color) - material_value(board, not color)

def legal_lines(board: chess.Board) -> list[str]:
    groups = {"captures": [], "checks": [], "quiet": []}
    for move in board.legal_moves:
        takes, checks = board.is_capture(move), board.gives_check(move)
        if takes: groups["captures"].append(move)
        if checks: groups["checks"].append(move)
        if not takes and not checks: groups["quiet"].append(move)
    return [f"legal: {name} {_uci_list(moves)}" for name, moves in groups.items()]

def capture_threats(board: chess.Board) -> list[dict]:
    """Hypothetical enemy turn plus one legal recapture, not SEE or a legal pass."""
    ours = board.turn
    enemy = board.copy(stack=False)
    enemy.turn = not ours
    enemy.ep_square = None
    threats = []
    for move in list(enemy.legal_moves):
        victim_square = captured_square(enemy, move)
        victim = enemy.piece_at(victim_square)
        if not enemy.is_capture(move) or not victim or victim.color != ours or victim.piece_type == chess.KING:
            continue
        after = enemy.copy(stack=False)
        after.push(move)
        replies = [reply for reply in after.legal_moves if after.is_capture(reply) and captured_square(after, reply) == move.to_square]
        best_balance = material_balance(after, ours)
        for reply in replies:
            after.push(reply)
            best_balance = max(best_balance, material_balance(after, ours))
            after.pop()
        loss = material_balance(board, ours) - best_balance
        threats.append({"piece": NAMES[victim.piece_type], "square": chess.square_name(victim_square), "capture": move.uci(), "recaptures": sorted(r.uci() for r in replies), "local_material_loss_cp": loss})
    return sorted(threats, key=lambda item: (-item["local_material_loss_cp"], item["square"], item["capture"]))

def hanging_line(board: chess.Board) -> list[str]:
    threats = capture_threats(board)
    lines = ["hanging: local capture/recapture estimate only; no full exchange or positional safety guarantee"]
    if not threats:
        return lines + ["hanging: no legal capture threats found in the hypothetical enemy-turn view"]
    for item in threats:
        replies = " ".join(item["recaptures"]) or "none"
        lines.append(f"hanging: {item['piece']} {item['square']}; enemy {item['capture']}; legal recaptures {replies}; local material loss {item['local_material_loss_cp'] / 100:+.2f}")
    return lines

def king_line(board: chess.Board) -> list[str]:
    color = board.turn
    rights = [name for name, exists in (("kingside", board.has_kingside_castling_rights(color)), ("queenside", board.has_queenside_castling_rights(color))) if exists]
    castles = [move for move in board.legal_moves if board.is_castling(move)]
    return [f"king: in_check {'yes' if board.is_check() else 'no'}; this is not a safety assessment", f"king: castling_rights {' '.join(rights) or 'none'}; castling_legal_now {_uci_list(castles)}", "king: candidate observations separately check one opponent reply"]

def material_line(board: chess.Board) -> list[str]:
    def counts(color):
        return " ".join(f"{chess.piece_symbol(k).upper() if color else chess.piece_symbol(k)}{len(board.pieces(k, color))}" for k in VALUES)
    balance = material_value(board, board.turn) - material_value(board, not board.turn)
    return [f"material: white {counts(chess.WHITE)}, black {counts(chess.BLACK)}", f"material: your approximate balance {balance / 100:+.2f}; P1 N3.2 B3.3 R5 Q9; not a position evaluation"]

def move_label(board: chess.Board, move: chess.Move) -> str:
    tags = []
    if board.is_capture(move): tags.append(f"capture {captured_value(board, move) / 100:g}")
    if board.gives_check(move): tags.append("check")
    if board.is_castling(move): tags.append("castle")
    if move.promotion: tags.append("promote " + NAMES[move.promotion])
    return f"{move.uci()} {board.san(move)}" + ("; " + ", ".join(tags) if tags else "")

def preview_move(board: chess.Board, move: chess.Move) -> dict:
    if move not in board.legal_moves:
        raise ValueError(f"Illegal preview: {move.uci()}")
    color, san = board.turn, board.san(move)
    baseline = material_balance(board, color)
    after = board.copy()
    after.push(move)
    gain = material_balance(after, color) - baseline
    outcome = chess_outcome(after)
    recaptures, mates = [], []
    if outcome is None:
        for reply in list(after.legal_moves):
            takes_mover = after.is_capture(reply) and captured_square(after, reply) == move.to_square
            reply_san = after.san(reply) if takes_mover else None
            after.push(reply)
            if takes_mover:
                recaptures.append({"uci": reply.uci(), "san": reply_san, "material_delta_cp": material_balance(after, color) - baseline})
            if after.is_checkmate(): mates.append(reply.uci())
            after.pop()
    return {"uci": move.uci(), "san": san, "before_fen": board.fen(), "after_fen": after.fen(), "gives_check": after.is_check(), "checkmate": after.is_checkmate(), "material_delta_cp": gain, "recaptures": recaptures, "opponent_mate_in_one": sorted(mates), "terminal": outcome.termination.name.lower() if outcome else None, "winner": ("white" if outcome.winner else "black") if outcome and outcome.winner is not None else None, "mover": "white" if color else "black", "scope": "exact legal moves and one opponent reply; no engine, no full exchange search"}

def preview_lines(result: dict) -> list[str]:
    recaptures = "; ".join(f"{r['uci']} {r['san']} your material delta {r['material_delta_cp'] / 100:+.2f}" for r in result["recaptures"]) or "none"
    return [f"preview {result['uci']} {result['san']}: terminal {result['terminal'] or 'no'}; checkmate {'yes' if result['checkmate'] else 'no'}", f"preview: immediate material delta {result['material_delta_cp'] / 100:+.2f}; legal captures of moved piece {recaptures}", f"preview: opponent mate-in-one {' '.join(result['opponent_mate_in_one']) or 'none found'}", f"preview: after FEN {result['after_fen']}", "preview: no mate-in-one found does not mean safe; other captures and deeper tactics remain"]

def tool_result(name: str, board: chess.Board) -> list[str]:
    return {"legal": legal_lines, "hanging": hanging_line, "king": king_line, "material": material_line}[name](board)

def movable_pieces(board: chess.Board) -> list[tuple[int, str, list[chess.Move]]]:
    grouped: dict[int, list[chess.Move]] = {}
    for move in board.legal_moves:
        grouped.setdefault(move.from_square, []).append(move)
    return [(sq, f"{NAMES[board.piece_type_at(sq)]} {chess.square_name(sq)}", sorted(grouped[sq], key=lambda m: m.uci())) for sq in sorted(grouped)]


def prepare_candidates(board: chess.Board) -> tuple[list[dict], list[str]]:
    """No heuristic filtering: every legal move receives the same rule checks."""
    order = {kind: index for index, kind in enumerate((chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING, chess.PAWN))}
    legal = sorted(board.legal_moves, key=lambda move: (order[board.piece_type_at(move.from_square)], move.from_square, move.uci()))
    candidates, observations = [], []
    for move in legal:
        preview = preview_move(board, move)
        tags = []
        if board.is_capture(move): tags.append(f"capture {captured_value(board, move) / 100:g}")
        if board.is_en_passant(move): tags.append('en passant')
        if preview['gives_check']: tags.append('check')
        if board.is_castling(move): tags.append('castle')
        if move.promotion: tags.append('promote ' + NAMES[move.promotion])
        label = f"{move.uci()} ({preview['san']})" + (' [' + '; '.join(tags) + ']' if tags else '')
        candidates.append({'uci': move.uci(), 'san': preview['san'], 'piece': NAMES[board.piece_type_at(move.from_square)],
                           'from': chess.square_name(move.from_square), 'to': chess.square_name(move.to_square),
                           'label': label, 'tags': tags, 'preview': preview})
        facts = []
        if preview['checkmate']:
            facts.append('delivers checkmate')
        elif preview['terminal']:
            facts.append('terminal ' + preview['terminal'])
        if preview['opponent_mate_in_one']:
            facts.append('opponent mate-in-one: ' + ' '.join(preview['opponent_mate_in_one']))
        if preview['recaptures']:
            replies = preview['recaptures']
            worst = min(reply['material_delta_cp'] for reply in replies) / 100
            facts.append('opponent can capture moved piece: ' + ' '.join(reply['uci'] for reply in replies)
                         + f'; immediate delta {worst:+g}')
        if facts:
            observations.append(move.uci() + ': ' + '; '.join(facts))
    return candidates, observations
