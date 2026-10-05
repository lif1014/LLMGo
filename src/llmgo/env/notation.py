"""棋盘文本、中文记谱。红方用中文数字，黑方用阿拉伯数字。"""

from __future__ import annotations

import re
import unicodedata

from llmgo.env.xiangqi import BLACK, RED, Move, Piece, Position

RED_CHAR = {
    "king": "帅",
    "advisor": "仕",
    "elephant": "相",
    "horse": "马",
    "rook": "车",
    "cannon": "炮",
    "pawn": "兵",
}
BLACK_CHAR = {
    "king": "将",
    "advisor": "士",
    "elephant": "象",
    "horse": "马",
    "rook": "车",
    "cannon": "炮",
    "pawn": "卒",
}
CHAR = {RED: RED_CHAR, BLACK: BLACK_CHAR}
CN_NUM = "一二三四五六七八九"
NUM_VALUE = {str(i): i for i in range(1, 10)}
NUM_VALUE.update({char: index + 1 for index, char in enumerate(CN_NUM)})
MOVE_RE = re.compile(
    r"([前后中])?(帅|将|仕|士|相|象|马|车|炮|兵|卒)([1-9一二三四五六七八九])?(进|退|平)([1-9一二三四五六七八九])"
)
ICCS_RE = re.compile(r"[a-iA-I][0-9][a-iA-I][0-9]")
TAG_RE = re.compile(r"<move>\s*(.*?)\s*</move>", re.IGNORECASE | re.DOTALL)


def piece_char(piece: Piece) -> str:
    return CHAR[piece.color][piece.kind]


def piece_disc(piece: Piece) -> str:
    char = piece_char(piece)
    if piece.color == RED:
        return f"（{char}）"
    return f"【{char}】"


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def _fit(text: str, width: int) -> str:
    pad = max(0, width - _display_width(text))
    left = pad // 2
    return " " * left + text + " " * (pad - left)


def render_board(pos: Position) -> str:
    cell = 6
    header = "   " + "".join(_fit(char, cell) for char in "abcdefghi")
    lines = [
        "红方在下，黑方在上。（）是红方棋子，【】是黑方棋子。",
        header,
    ]
    for rank in range(9, -1, -1):
        cells = []
        for file in range(9):
            piece = pos.board[rank][file]
            cells.append(_fit(piece_disc(piece) if piece else "·", cell))
        lines.append(f"{rank} " + "".join(cells))
        if rank == 5:
            river = "楚河" + " " * 8 + "汉界"
            lines.append("  " + _fit(river, cell * 9))
    lines.append("九宫、河界和棋子路线都以这张图为准。楚河汉界在第 5 段和第 4 段之间。")
    return "\n".join(lines)


def _file_label(color: str, file: int) -> str:
    if color == RED:
        return CN_NUM[8 - file]
    return str(file + 1)


def _file_index(color: str, number: int) -> int:
    if color == RED:
        return 9 - number
    return number - 1


def _step_label(color: str, steps: int) -> str:
    if color == RED:
        return CN_NUM[steps - 1]
    return str(steps)


def chinese_notation(pos: Position, move: Move) -> str:
    piece = pos.board[move.src[1]][move.src[0]]
    if piece is None:
        return move.uci
    name = piece_char(piece)
    same = [
        (file, rank)
        for rank in range(10)
        for file in range(9)
        if (found := pos.board[rank][file]) is not None and found.color == piece.color and found.kind == piece.kind
    ]
    on_file = [item for item in same if item[0] == move.src[0]]
    if len(on_file) <= 1:
        prefix = name + _file_label(piece.color, move.src[0])
    else:
        ordered = sorted(on_file, key=lambda item: item[1], reverse=(piece.color == RED))
        index = ordered.index(move.src)
        if len(ordered) == 2:
            locator = "前" if index == 0 else "后"
        elif len(ordered) == 3:
            locator = ("前", "中", "后")[index]
        else:
            locator = ("前", "二", "三", "四", "五")[index] if index < 5 else str(index + 1)
        prefix = locator + name
    df = move.dst[0] - move.src[0]
    dr = move.dst[1] - move.src[1]
    forward = dr > 0 if piece.color == RED else dr < 0
    if dr == 0:
        verb = "平"
        dest = _file_label(piece.color, move.dst[0])
    elif piece.kind in {"horse", "elephant", "advisor"}:
        verb = "进" if forward else "退"
        dest = _file_label(piece.color, move.dst[0])
    else:
        verb = "进" if forward else "退"
        dest = _step_label(piece.color, abs(dr))
    return prefix + verb + dest


def _kind_from_char(color: str, char: str) -> str | None:
    table = RED_CHAR if color == RED else BLACK_CHAR
    for kind, label in table.items():
        if label == char:
            return kind
    return None


def parse_chinese(pos: Position, text: str) -> Move | None:
    matched = MOVE_RE.search(text.replace("砲", "炮").replace("傌", "马").replace("俥", "车").replace("車", "车"))
    if not matched:
        return None
    locator, char, origin, action, dest = matched.groups()
    color = pos.side
    kind = _kind_from_char(color, char)
    if kind is None or dest not in NUM_VALUE:
        return None
    dest_n = NUM_VALUE[dest]
    candidates = [
        (file, rank)
        for rank in range(10)
        for file in range(9)
        if (piece := pos.board[rank][file]) is not None and piece.color == color and piece.kind == kind
    ]
    if origin is not None and locator is None:
        origin_file = _file_index(color, NUM_VALUE[origin])
        candidates = [item for item in candidates if item[0] == origin_file]
    elif locator is not None:
        pool = candidates
        if origin is not None:
            origin_file = _file_index(color, NUM_VALUE[origin])
            narrowed = [item for item in pool if item[0] == origin_file]
            if narrowed:
                pool = narrowed
        ordered = sorted(pool, key=lambda item: item[1], reverse=(color == RED))
        pick = {"前": 0, "中": 1 if len(ordered) == 3 else 0, "后": len(ordered) - 1}.get(locator, 0)
        if not ordered or pick >= len(ordered):
            return None
        candidates = [ordered[pick]]
    found: list[Move] = []
    for src in candidates:
        move = _destination_from_words(pos, src, kind, color, action, dest_n)
        if move is not None:
            found.append(move)
    if len(found) == 1:
        return found[0]
    legal = {item.uci: item for item in pos.legal_moves()}
    agreed = [item for item in found if item.uci in legal]
    if len(agreed) == 1:
        return agreed[0]
    return None


def _destination_from_words(
    pos: Position,
    src: tuple[int, int],
    kind: str,
    color: str,
    action: str,
    dest_n: int,
) -> Move | None:
    direction = 1 if color == RED else -1
    if action == "退":
        direction = -direction
    if action == "平":
        dst = (_file_index(color, dest_n), src[1])
    elif kind in {"horse", "elephant", "advisor"}:
        dst_file = _file_index(color, dest_n)
        options = [
            dest
            for dest in pos.pseudo_destinations(*src)
            if dest[0] == dst_file and (dest[1] - src[1]) * direction > 0
        ]
        if len(options) != 1:
            return None
        dst = options[0]
    else:
        dst = (src[0], src[1] + direction * dest_n)
    if not (0 <= dst[0] < 9 and 0 <= dst[1] < 10):
        return None
    return Move(f"{'abcdefghi'[src[0]]}{src[1]}{'abcdefghi'[dst[0]]}{dst[1]}", src, dst)


def parse_model_move(text: str, pos: Position) -> tuple[Move | None, str | None]:
    """从模型原文取出走法。返回 (move, parsed_by)。"""
    chunks = TAG_RE.findall(text or "")
    sources = chunks or [text or ""]
    for chunk in sources:
        iccs = ICCS_RE.search(chunk)
        if iccs:
            move = Move.parse(iccs.group(0))
            if move is not None:
                return move, "iccs"
        move = parse_chinese(pos, chunk)
        if move is not None:
            return move, "chinese"
    return None, None
