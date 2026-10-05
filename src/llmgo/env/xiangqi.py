"""中国象棋规则。坐标用 ICCS：红方在下，a–i 从左到右，0–9 从下到上。"""

from __future__ import annotations

from dataclasses import dataclass

FILES = "abcdefghi"
RED = "red"
BLACK = "black"
KINDS = ("king", "advisor", "elephant", "horse", "rook", "cannon", "pawn")

FEN_OF = {
    (RED, "king"): "K",
    (RED, "advisor"): "A",
    (RED, "elephant"): "B",
    (RED, "horse"): "N",
    (RED, "rook"): "R",
    (RED, "cannon"): "C",
    (RED, "pawn"): "P",
    (BLACK, "king"): "k",
    (BLACK, "advisor"): "a",
    (BLACK, "elephant"): "b",
    (BLACK, "horse"): "n",
    (BLACK, "rook"): "r",
    (BLACK, "cannon"): "c",
    (BLACK, "pawn"): "p",
}
PIECE_OF = {value: key for key, value in FEN_OF.items()}

INITIAL_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"

ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))
HORSE = ((-1, 2), (1, 2), (-2, 1), (2, 1), (-2, -1), (2, -1), (-1, -2), (1, -2))


def opponent(color: str) -> str:
    return BLACK if color == RED else RED


def on_board(file: int, rank: int) -> bool:
    return 0 <= file < 9 and 0 <= rank < 10


def parse_uci(text: str) -> tuple[int, int, int, int] | None:
    raw = text.strip().lower()
    if len(raw) != 4:
        return None
    if raw[0] not in FILES or raw[2] not in FILES:
        return None
    if raw[1] not in "0123456789" or raw[3] not in "0123456789":
        return None
    return FILES.index(raw[0]), int(raw[1]), FILES.index(raw[2]), int(raw[3])


def to_uci(src: tuple[int, int], dst: tuple[int, int]) -> str:
    return f"{FILES[src[0]]}{src[1]}{FILES[dst[0]]}{dst[1]}"


@dataclass(frozen=True)
class Piece:
    color: str
    kind: str

    @property
    def fen(self) -> str:
        return FEN_OF[(self.color, self.kind)]


@dataclass(frozen=True)
class Move:
    uci: str
    src: tuple[int, int]
    dst: tuple[int, int]

    @classmethod
    def parse(cls, text: str) -> Move | None:
        point = parse_uci(text)
        if point is None:
            return None
        f1, r1, f2, r2 = point
        return cls(to_uci((f1, r1), (f2, r2)), (f1, r1), (f2, r2))


@dataclass(frozen=True)
class Verdict:
    ok: bool
    error_type: str | None = None
    error_detail: str | None = None
    move: Move | None = None


class Position:
    def __init__(self, board: list[list[Piece | None]], side: str, seen: dict[str, int] | None = None):
        self.board = board
        self.side = side
        self.seen = dict(seen or {})
        key = self.key()
        self.seen[key] = self.seen.get(key, 0) + (0 if seen is not None else 1)
        if seen is None:
            self.seen[key] = 1

    @classmethod
    def initial(cls) -> Position:
        return cls.from_fen(INITIAL_FEN)

    @classmethod
    def from_fen(cls, fen: str) -> Position:
        parts = fen.split()
        rows = parts[0].split("/")
        if len(rows) != 10:
            raise ValueError(f"FEN 行数不对: {fen}")
        board: list[list[Piece | None]] = [[None] * 9 for _ in range(10)]
        for row_index, row in enumerate(rows):
            rank = 9 - row_index
            file = 0
            for char in row:
                if char.isdigit():
                    file += int(char)
                    continue
                if char not in PIECE_OF:
                    raise ValueError(f"无法识别的棋子: {char}")
                color, kind = PIECE_OF[char]
                if file > 8:
                    raise ValueError(f"FEN 超出棋盘: {fen}")
                board[rank][file] = Piece(color, kind)
                file += 1
            if file != 9:
                raise ValueError(f"FEN 列数不对: {fen}")
        side = RED if len(parts) < 2 or parts[1] == "w" else BLACK
        return cls(board, side)

    def key(self) -> str:
        return f"{self.board_fen()} {self.side}"

    def board_fen(self) -> str:
        rows = []
        for rank in range(9, -1, -1):
            empty = 0
            row = []
            for file in range(9):
                piece = self.board[rank][file]
                if piece is None:
                    empty += 1
                    continue
                if empty:
                    row.append(str(empty))
                    empty = 0
                row.append(piece.fen)
            if empty:
                row.append(str(empty))
            rows.append("".join(row))
        return "/".join(rows)

    def fen(self) -> str:
        side = "w" if self.side == RED else "b"
        return f"{self.board_fen()} {side} - - 0 1"

    def piece_at(self, file: int, rank: int) -> Piece | None:
        if not on_board(file, rank):
            return None
        return self.board[rank][file]

    def find_king(self, color: str) -> tuple[int, int] | None:
        for rank in range(10):
            for file in range(9):
                piece = self.board[rank][file]
                if piece and piece.color == color and piece.kind == "king":
                    return file, rank
        return None

    def kings_face(self) -> bool:
        red = self.find_king(RED)
        black = self.find_king(BLACK)
        if red is None or black is None or red[0] != black[0]:
            return False
        low, high = sorted((red[1], black[1]))
        for rank in range(low + 1, high):
            if self.board[rank][red[0]] is not None:
                return False
        return True

    def clone(self) -> Position:
        board = [row[:] for row in self.board]
        return Position(board, self.side, dict(self.seen))

    def pseudo_destinations(self, file: int, rank: int) -> list[tuple[int, int]]:
        piece = self.board[rank][file]
        if piece is None:
            return []
        if piece.kind == "king":
            return self._king_steps(piece, file, rank)
        if piece.kind == "advisor":
            return self._advisor_steps(piece, file, rank)
        if piece.kind == "elephant":
            return self._elephant_steps(piece, file, rank)
        if piece.kind == "horse":
            return self._horse_steps(piece, file, rank)
        if piece.kind == "rook":
            return self._rook_steps(piece, file, rank)
        if piece.kind == "cannon":
            return self._cannon_steps(piece, file, rank)
        return self._pawn_steps(piece, file, rank)

    def _can_land(self, color: str, file: int, rank: int) -> bool:
        piece = self.board[rank][file]
        return piece is None or piece.color != color

    def _king_steps(self, piece: Piece, file: int, rank: int) -> list[tuple[int, int]]:
        found = []
        for df, dr in ORTHO:
            nf, nr = file + df, rank + dr
            if self._in_palace(piece.color, nf, nr) and self._can_land(piece.color, nf, nr):
                found.append((nf, nr))
        return found

    def _advisor_steps(self, piece: Piece, file: int, rank: int) -> list[tuple[int, int]]:
        if not self._in_palace(piece.color, file, rank):
            return []
        found = []
        for df in (-1, 1):
            for dr in (-1, 1):
                nf, nr = file + df, rank + dr
                if self._in_palace(piece.color, nf, nr) and self._can_land(piece.color, nf, nr):
                    found.append((nf, nr))
        return found

    def _elephant_steps(self, piece: Piece, file: int, rank: int) -> list[tuple[int, int]]:
        if not self._own_side(piece.color, rank):
            return []
        found = []
        for df in (-2, 2):
            for dr in (-2, 2):
                nf, nr = file + df, rank + dr
                eye_f, eye_r = file + df // 2, rank + dr // 2
                if not on_board(nf, nr) or not self._own_side(piece.color, nr):
                    continue
                if self.board[eye_r][eye_f] is not None:
                    continue
                if self._can_land(piece.color, nf, nr):
                    found.append((nf, nr))
        return found

    def _horse_steps(self, piece: Piece, file: int, rank: int) -> list[tuple[int, int]]:
        found = []
        for df, dr in HORSE:
            nf, nr = file + df, rank + dr
            if not on_board(nf, nr) or not self._can_land(piece.color, nf, nr):
                continue
            leg_f = file + (0 if abs(df) == 1 else (1 if df > 0 else -1))
            leg_r = rank + (0 if abs(dr) == 1 else (1 if dr > 0 else -1))
            if self.board[leg_r][leg_f] is not None:
                continue
            found.append((nf, nr))
        return found

    def _rook_steps(self, piece: Piece, file: int, rank: int) -> list[tuple[int, int]]:
        found = []
        for df, dr in ORTHO:
            nf, nr = file + df, rank + dr
            while on_board(nf, nr):
                blocker = self.board[nr][nf]
                if blocker is None:
                    found.append((nf, nr))
                else:
                    if blocker.color != piece.color:
                        found.append((nf, nr))
                    break
                nf += df
                nr += dr
        return found

    def _cannon_steps(self, piece: Piece, file: int, rank: int) -> list[tuple[int, int]]:
        found = []
        for df, dr in ORTHO:
            nf, nr = file + df, rank + dr
            while on_board(nf, nr) and self.board[nr][nf] is None:
                found.append((nf, nr))
                nf += df
                nr += dr
            if not on_board(nf, nr):
                continue
            nf += df
            nr += dr
            while on_board(nf, nr) and self.board[nr][nf] is None:
                nf += df
                nr += dr
            if on_board(nf, nr):
                target = self.board[nr][nf]
                if target is not None and target.color != piece.color:
                    found.append((nf, nr))
        return found

    def _pawn_steps(self, piece: Piece, file: int, rank: int) -> list[tuple[int, int]]:
        forward = 1 if piece.color == RED else -1
        crossed = rank >= 5 if piece.color == RED else rank <= 4
        deltas = [(0, forward)]
        if crossed:
            deltas.extend([(1, 0), (-1, 0)])
        found = []
        for df, dr in deltas:
            nf, nr = file + df, rank + dr
            if on_board(nf, nr) and self._can_land(piece.color, nf, nr):
                found.append((nf, nr))
        return found

    def square_attacked(self, file: int, rank: int, by_color: str) -> bool:
        for y in range(10):
            for x in range(9):
                piece = self.board[y][x]
                if piece is None or piece.color != by_color:
                    continue
                if (file, rank) in self.pseudo_destinations(x, y):
                    return True
        king = self.find_king(by_color)
        target = self.piece_at(file, rank)
        if king and target and target.kind == "king" and target.color != by_color:
            return self.kings_face()
        return False

    def in_check(self, color: str) -> bool:
        king = self.find_king(color)
        if king is None:
            return True
        return self.square_attacked(king[0], king[1], opponent(color))

    def legal_moves(self) -> list[Move]:
        moves = []
        for rank in range(10):
            for file in range(9):
                piece = self.board[rank][file]
                if piece is None or piece.color != self.side:
                    continue
                for dest in self.pseudo_destinations(file, rank):
                    move = Move(to_uci((file, rank), dest), (file, rank), dest)
                    if self.classify(move).ok:
                        moves.append(move)
        moves.sort(key=lambda item: item.uci)
        return moves

    def classify(self, move: Move) -> Verdict:
        f1, r1 = move.src
        f2, r2 = move.dst
        if (f1, r1) == (f2, r2):
            return Verdict(False, "unchanged", "起点和终点相同", move)
        piece = self.board[r1][f1]
        if piece is None:
            return Verdict(False, "empty_origin", "起点没有棋子", move)
        if piece.color != self.side:
            return Verdict(False, "not_own_piece", "起点不是己方棋子", move)
        if move.dst not in self.pseudo_destinations(f1, r1):
            return Verdict(False, "illegal_geometry", self.explain_geometry(piece, move), move)
        nxt = self._pushed(move)
        if nxt.kings_face():
            return Verdict(False, "king_face", "造成将帅照面", move)
        if nxt.in_check(self.side):
            return Verdict(False, "leaves_check", "走完后己方被将军", move)
        return Verdict(True, move=move)

    def apply(self, move: Move) -> tuple[Position, str | None, str | None]:
        verdict = self.classify(move)
        if not verdict.ok:
            raise ValueError(verdict.error_detail or verdict.error_type)
        nxt = self._pushed(move)
        if nxt.find_king(opponent(self.side)) is None:
            return nxt, f"{self.side}_win", "capture_king"
        if nxt.seen.get(nxt.key(), 0) >= 3:
            return nxt, "draw", "repetition"
        if not nxt.legal_moves():
            if nxt.in_check(nxt.side):
                return nxt, f"{self.side}_win", "checkmate"
            return nxt, f"{self.side}_win", "stalemate"
        return nxt, None, None

    def _pushed(self, move: Move) -> Position:
        board = [row[:] for row in self.board]
        piece = board[move.src[1]][move.src[0]]
        board[move.src[1]][move.src[0]] = None
        board[move.dst[1]][move.dst[0]] = piece
        nxt = Position(board, opponent(self.side), dict(self.seen))
        nxt.seen[nxt.key()] = nxt.seen.get(nxt.key(), 0) + 1
        return nxt

    def explain_geometry(self, piece: Piece, move: Move) -> str:
        f1, r1 = move.src
        f2, r2 = move.dst
        target = self.board[r2][f2] if on_board(f2, r2) else None
        if target is not None and target.color == piece.color:
            return "目标格是己方棋子"
        if piece.kind == "horse":
            return self._explain_horse(f1, r1, f2, r2)
        if piece.kind == "elephant":
            return self._explain_elephant(piece, f1, r1, f2, r2)
        if piece.kind == "advisor":
            return "仕或士只能在九宫内走一格斜线"
        if piece.kind == "king":
            if not self._in_palace(piece.color, f2, r2):
                return "帅或将不能出九宫"
            return "帅或将每次只能走一格直线"
        if piece.kind == "rook":
            if f1 != f2 and r1 != r2:
                return "车必须走直线"
            return "车的路线被挡住"
        if piece.kind == "cannon":
            return self._explain_cannon(f1, r1, f2, r2)
        return self._explain_pawn(piece, f1, r1, f2, r2)

    def _explain_horse(self, f1: int, r1: int, f2: int, r2: int) -> str:
        df, dr = f2 - f1, r2 - r1
        if (df, dr) not in HORSE or not on_board(f2, r2):
            return "马必须走日字"
        leg_f = f1 + (0 if abs(df) == 1 else (1 if df > 0 else -1))
        leg_r = r1 + (0 if abs(dr) == 1 else (1 if dr > 0 else -1))
        if self.board[leg_r][leg_f] is not None:
            return "马腿被挡"
        return "马不能走到该格"

    def _explain_elephant(self, piece: Piece, f1: int, r1: int, f2: int, r2: int) -> str:
        df, dr = f2 - f1, r2 - r1
        if not on_board(f2, r2) or abs(df) != 2 or abs(dr) != 2:
            return "象必须走田字"
        if not self._own_side(piece.color, r2):
            return "象不能过河"
        eye = self.board[r1 + dr // 2][f1 + df // 2]
        if eye is not None:
            return "塞象眼"
        return "象不能走到该格"

    def _explain_cannon(self, f1: int, r1: int, f2: int, r2: int) -> str:
        if f1 != f2 and r1 != r2:
            return "炮必须走直线"
        if not on_board(f2, r2):
            return "炮不能走出棋盘"
        screens = self._screens_between(f1, r1, f2, r2)
        target = self.board[r2][f2]
        if target is None and screens:
            return "炮移动时路线上不能有棋子"
        if target is not None and screens != 1:
            return "炮吃子必须隔恰好一个棋子"
        return "炮不能走到该格"

    def _explain_pawn(self, piece: Piece, f1: int, r1: int, f2: int, r2: int) -> str:
        forward = 1 if piece.color == RED else -1
        crossed = r1 >= 5 if piece.color == RED else r1 <= 4
        if (f2 - f1, r2 - r1) == (0, -forward):
            return "兵或卒不能后退"
        if f1 != f2 and not crossed:
            return "兵或卒未过河不能横走"
        return "兵或卒不能这样走"

    def _screens_between(self, f1: int, r1: int, f2: int, r2: int) -> int:
        df = 0 if f1 == f2 else (1 if f2 > f1 else -1)
        dr = 0 if r1 == r2 else (1 if r2 > r1 else -1)
        count = 0
        f, r = f1 + df, r1 + dr
        while (f, r) != (f2, r2):
            if self.board[r][f] is not None:
                count += 1
            f += df
            r += dr
        return count

    @staticmethod
    def _in_palace(color: str, file: int, rank: int) -> bool:
        if not (3 <= file <= 5 and on_board(file, rank)):
            return False
        if color == RED:
            return rank <= 2
        return rank >= 7

    @staticmethod
    def _own_side(color: str, rank: int) -> bool:
        if color == RED:
            return rank <= 4
        return rank >= 5

    def pieces(self) -> list[dict]:
        found = []
        for rank in range(10):
            for file in range(9):
                piece = self.board[rank][file]
                if piece is None:
                    continue
                found.append(
                    {
                        "file": file,
                        "rank": rank,
                        "color": piece.color,
                        "kind": piece.kind,
                    }
                )
        return found
