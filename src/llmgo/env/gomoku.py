"""五子棋规则。15×15，黑先，横竖斜连成五子或以上即胜。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from llmgo.runtime.prompt import Observation

FILES = "abcdefghijklmno"
SIZE = 15
BLACK = "black"
WHITE = "white"
TAG_RE = re.compile(r"<move>\s*(.*?)\s*</move>", re.IGNORECASE | re.DOTALL)
COORD_RE = re.compile(r"[a-oA-O](?:1[0-5]|[1-9])")
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))

SYSTEM_PROMPT = """You are playing Gomoku as one side only. You can see the board and the moves already played. You cannot see the opponent's reasoning, draft, or raw output. Do not guess why the opponent played a move. Decide from the current board.

Rules:
- The board has 15 by 15 intersections.
- Files a to o run from left to right. Ranks 1 to 15 run from bottom to top. a1 is the bottom-left intersection.
- Black plays first. On your turn, place one of your stones on an empty intersection.
- Stones never move and never capture.
- You win by making a continuous line of five or more of your own stones horizontally, vertically, or diagonally.
- If every intersection is filled and neither side has such a line, the game is drawn.

B is a black stone. W is a white stone. A dot is empty.
After thinking, output one tag:
<move>h8</move>
"""


def opponent(color: str) -> str:
    return WHITE if color == BLACK else BLACK


def on_board(file: int, rank: int) -> bool:
    return 0 <= file < SIZE and 0 <= rank < SIZE


@dataclass(frozen=True)
class Move:
    uci: str
    file: int
    rank: int

    @classmethod
    def parse(cls, text: str) -> Move | None:
        found = COORD_RE.fullmatch((text or "").strip())
        if found is None:
            return None
        raw = found.group(0).lower()
        file = FILES.index(raw[0])
        rank = int(raw[1:]) - 1
        if not on_board(file, rank):
            return None
        return cls(f"{FILES[file]}{rank + 1}", file, rank)


@dataclass(frozen=True)
class Verdict:
    ok: bool
    error_type: str | None = None
    error_detail: str | None = None
    move: Move | None = None


class Position:
    def __init__(self, board: list[list[str | None]], side: str):
        self.board = board
        self.side = side

    @classmethod
    def initial(cls) -> Position:
        return cls([[None for _ in range(SIZE)] for _ in range(SIZE)], BLACK)

    def fen(self) -> str:
        rows = []
        for rank in range(SIZE - 1, -1, -1):
            cells = []
            for file in range(SIZE):
                stone = self.board[rank][file]
                cells.append("b" if stone == BLACK else "w" if stone == WHITE else "1")
            rows.append("".join(cells))
        side = "b" if self.side == BLACK else "w"
        return f"{'/'.join(rows)} {side}"

    def pieces(self) -> list[dict]:
        found = []
        for rank in range(SIZE):
            for file in range(SIZE):
                stone = self.board[rank][file]
                if stone is None:
                    continue
                found.append({"file": file, "rank": rank, "color": stone, "kind": "stone"})
        return found

    def in_check(self, color: str) -> bool:
        return False

    def legal_moves(self) -> list[Move]:
        moves = []
        for rank in range(SIZE):
            for file in range(SIZE):
                if self.board[rank][file] is None:
                    moves.append(Move(f"{FILES[file]}{rank + 1}", file, rank))
        return moves

    def classify(self, move: Move) -> Verdict:
        if not on_board(move.file, move.rank):
            return Verdict(False, "off_board", "That coordinate is off the board.", move)
        if self.board[move.rank][move.file] is not None:
            return Verdict(False, "occupied", "That intersection is already occupied.", move)
        return Verdict(True, move=move)

    def apply(self, move: Move) -> tuple[Position, str | None, str | None]:
        board = [row[:] for row in self.board]
        board[move.rank][move.file] = self.side
        nxt = Position(board, opponent(self.side))
        if _has_five(board, move.file, move.rank, self.side):
            return nxt, f"{self.side}_win", "five"
        if all(cell is not None for row in board for cell in row):
            return nxt, "draw", "full"
        return nxt, None, None


def _has_five(board: list[list[str | None]], file: int, rank: int, color: str) -> bool:
    for df, dr in DIRECTIONS:
        count = 1
        for sign in (1, -1):
            step = 1
            while True:
                nf = file + df * sign * step
                nr = rank + dr * sign * step
                if not on_board(nf, nr) or board[nr][nf] != color:
                    break
                count += 1
                step += 1
        if count >= 5:
            return True
    return False


def render_board(pos: Position) -> str:
    lines = ["B is black, W is white, and a dot is empty. a1 is the bottom-left intersection."]
    for rank in range(SIZE - 1, -1, -1):
        cells = []
        for file in range(SIZE):
            stone = pos.board[rank][file]
            cells.append("B" if stone == BLACK else "W" if stone == WHITE else ".")
        lines.append(f"{rank + 1:2d} " + " ".join(cells))
    lines.append("   " + " ".join(FILES))
    return "\n".join(lines)


def parse_model_move(text: str, pos: Position) -> tuple[Move | None, str | None]:
    """有多个 <move> 时用最后一个标签里的最后一个坐标。"""
    chunks = TAG_RE.findall(text or "")
    chunk = chunks[-1] if chunks else (text or "")
    hits = COORD_RE.findall(chunk)
    if not hits:
        return None, None
    move = Move.parse(hits[-1])
    if move is None:
        return None, None
    return move, "coordinate"


def _seat_name(seat: str) -> str:
    return "Black" if seat == BLACK else "White"


def _system_prompt(seat: str) -> str:
    return SYSTEM_PROMPT + f"\nYou are {_seat_name(seat)}. The opponent's reasoning is not in this message."


def _user_prompt(
    pos: Position,
    seat: str,
    history: list[dict],
    feedback: str | None,
    legal: tuple[str, ...],
    policy: str,
) -> str:
    lines = [
        render_board(pos),
        "",
        f"Turn: {_seat_name(seat)}",
        f"You are {_seat_name(seat)}. Play one move for this side only.",
    ]
    if history:
        lines.append("Moves already played:")
        for index, item in enumerate(history, start=1):
            lines.append(f"{index}. {_seat_name(item['seat'])} {item['uci']}")
    else:
        lines.append("No stones have been placed.")
    if policy == "legal_list":
        lines.append("Legal coordinates: " + " ".join(legal))
    if feedback:
        lines.extend(
            [
                "",
                "Your own previous attempt was rejected. The opponent cannot see this note:",
                feedback,
                "Output one <move> tag again.",
            ]
        )
    else:
        lines.append("Output one <move> tag.")
    return "\n".join(lines)


class GomokuGame:
    """五子棋对局入口。规则、解析和英文提示都在这个文件里。"""

    name = "gomoku"
    files = SIZE
    ranks = SIZE
    seats = (BLACK, WHITE)

    def initial(self) -> Position:
        return Position.initial()

    def opponent(self, seat: str) -> str:
        return opponent(seat)

    def seat_name(self, seat: str) -> str:
        return _seat_name(seat)

    def observe(
        self,
        pos: Position,
        seat: str,
        history: list[dict],
        feedback: str | None,
        attempt_index: int,
        policy: str,
    ) -> Observation:
        legal = tuple(move.uci for move in pos.legal_moves())
        user = _user_prompt(pos, seat, history, feedback, legal, policy)
        return Observation(
            seat=seat,
            fen=pos.fen(),
            board_text=render_board(pos),
            history=tuple(history),
            feedback=feedback,
            legal_moves=legal,
            ply=len(history),
            attempt_index=attempt_index,
            policy=policy,
            system_text=_system_prompt(seat),
            user_text=user,
        )

    def user_prompt(self, obs: Observation) -> str:
        return obs.user_text or ""

    def llm_input(self, obs: Observation, user_prompt: str) -> str:
        return f"【system】\n{obs.system_text}\n\n【user】\n{user_prompt}"

    def parse_move(self, text: str, pos: Position) -> tuple[Move | None, str | None]:
        return parse_model_move(text, pos)

    def label(self, pos: Position, move: Move) -> str | None:
        return None

    def parse_fail_detail(self) -> str:
        return "No <move> coordinate was found."

    def parse_fail_feedback(self) -> str:
        return "No playable move was found. Output one tag such as <move>h8</move>."

    def illegal_feedback(self, uci: str, detail: str | None) -> str:
        return f"Move {uci} cannot be played: {detail}"

    def request_feedback(self, exc: BaseException) -> str:
        return f"Request failed: {exc}"


GOMOKU = GomokuGame()
