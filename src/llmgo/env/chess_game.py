"""国际象棋规则。走子判断交给 chess 套件，提示和解析留在这里。"""

from __future__ import annotations

import re
from dataclasses import dataclass

import chess

from llmgo.runtime.prompt import Observation

WHITE = "white"
BLACK = "black"
TAG_RE = re.compile(r"<move>\s*(.*?)\s*</move>", re.IGNORECASE | re.DOTALL)
UCI_RE = re.compile(r"[a-hA-H][1-8][a-hA-H][1-8][qrbnQRBN]?")
HYPHEN_RE = re.compile(r"[a-hA-H][1-8]\s*[-x]\s*[a-hA-H][1-8]\s*=?\s*[qrbnQRBN]?")
SAN_RE = re.compile(
    r"(?:O-O-O|0-0-0|O-O|0-0|"
    r"[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=?[QRBN])?|"
    r"[a-h]x[a-h][1-8](?:=?[QRBN])?|"
    r"[a-h][1-8](?:=?[QRBN])?)"
    r"[+#]?",
    re.IGNORECASE,
)

KIND = {
    chess.KING: "king",
    chess.QUEEN: "queen",
    chess.ROOK: "rook",
    chess.BISHOP: "bishop",
    chess.KNIGHT: "knight",
    chess.PAWN: "pawn",
}
LETTER = {
    (chess.WHITE, chess.KING): "K",
    (chess.WHITE, chess.QUEEN): "Q",
    (chess.WHITE, chess.ROOK): "R",
    (chess.WHITE, chess.BISHOP): "B",
    (chess.WHITE, chess.KNIGHT): "N",
    (chess.WHITE, chess.PAWN): "P",
    (chess.BLACK, chess.KING): "k",
    (chess.BLACK, chess.QUEEN): "q",
    (chess.BLACK, chess.ROOK): "r",
    (chess.BLACK, chess.BISHOP): "b",
    (chess.BLACK, chess.KNIGHT): "n",
    (chess.BLACK, chess.PAWN): "p",
}
DRAW_REASON = {
    chess.Termination.STALEMATE: "stalemate",
    chess.Termination.INSUFFICIENT_MATERIAL: "insufficient_material",
    chess.Termination.THREEFOLD_REPETITION: "repetition",
    chess.Termination.FIVEFOLD_REPETITION: "repetition",
    chess.Termination.FIFTY_MOVES: "fifty_moves",
    chess.Termination.SEVENTYFIVE_MOVES: "seventyfive_moves",
}

SYSTEM_PROMPT = """You are playing chess as one side only. You can see the board and the moves already played. You cannot see the opponent's reasoning, draft, or raw output. Do not guess why the opponent played a move. Decide from the current board.

Rules:
- White is at the bottom and plays first. Files a to h run from left to right. Ranks 1 to 8 run from bottom to top. a1 is White's left corner.
- The king moves one square. The queen moves any number of squares in a straight line. Rooks move orthogonally, bishops diagonally, and knights in an L shape. Knights may jump.
- Pawns move one square forward, or two from their starting rank, and capture one square diagonally forward. A pawn that reaches the last rank must promote. An en-passant capture is written as the square the pawn lands on.
- Castle kingside with e1g1 or e8g8, and queenside with e1c1 or e8c8, when that move is legal.
- A move that leaves your own king in check is illegal. Checkmate wins. Stalemate, insufficient material, threefold repetition, and the fifty-move rule are draws.

Uppercase letters are White pieces and lowercase letters are Black pieces: K king, Q queen, R rook, B bishop, N knight, P pawn.
Write a move in UCI. Promotion adds the piece letter, for example e7e8q.
After thinking, output one tag:
<move>e2e4</move>
"""


def _color_name(color: chess.Color) -> str:
    return WHITE if color == chess.WHITE else BLACK


def _seat_name(seat: str) -> str:
    return "White" if seat == WHITE else "Black"


@dataclass(frozen=True)
class Move:
    uci: str


@dataclass(frozen=True)
class Verdict:
    ok: bool
    error_type: str | None = None
    error_detail: str | None = None
    move: Move | None = None


class Position:
    def __init__(self, board: chess.Board):
        self.board = board

    @classmethod
    def initial(cls) -> Position:
        return cls(chess.Board())

    @property
    def side(self) -> str:
        return _color_name(self.board.turn)

    def fen(self) -> str:
        return self.board.fen()

    def pieces(self) -> list[dict]:
        found = []
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None:
                continue
            found.append(
                {
                    "file": chess.square_file(square),
                    "rank": chess.square_rank(square),
                    "color": _color_name(piece.color),
                    "kind": KIND[piece.piece_type],
                }
            )
        return found

    def in_check(self, color: str) -> bool:
        if color != self.side:
            return False
        return self.board.is_check()

    def legal_moves(self) -> list[Move]:
        return [Move(move.uci()) for move in self.board.legal_moves]

    def classify(self, move: Move) -> Verdict:
        try:
            chess_move = chess.Move.from_uci(move.uci)
        except ValueError:
            return Verdict(False, "illegal_move", "That is not a chess move.", move)
        if chess_move in self.board.legal_moves:
            return Verdict(True, move=move)
        # 几何上走得通，但走完自己的王仍被将军，才算 leaves_check。
        if chess_move in self.board.pseudo_legal_moves and self.board.is_into_check(chess_move):
            return Verdict(False, "leaves_check", "That move leaves your king in check.", move)
        return Verdict(False, "illegal_move", "That move is illegal in this position.", move)

    def apply(self, move: Move) -> tuple[Position, str | None, str | None]:
        board = self.board.copy()
        board.push_uci(move.uci)
        nxt = Position(board)
        return (nxt, *nxt._result())

    def _result(self) -> tuple[str | None, str | None]:
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return None, None
        if outcome.winner is None:
            return "draw", DRAW_REASON.get(outcome.termination, "draw")
        winner = _color_name(outcome.winner)
        reason = "checkmate" if outcome.termination == chess.Termination.CHECKMATE else "chess_win"
        return f"{winner}_win", reason


def render_board(pos: Position) -> str:
    lines = ["Uppercase is White. Lowercase is Black. White is at the bottom. A dot is an empty square."]
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            piece = pos.board.piece_at(chess.square(file, rank))
            cells.append(LETTER[(piece.color, piece.piece_type)] if piece else ".")
        lines.append(f"{rank + 1} " + " ".join(cells))
    lines.append("  a b c d e f g h")
    return "\n".join(lines)


def _uci_from_hyphen(text: str) -> str:
    return re.sub(r"[\s\-x=]", "", text).lower()


def _from_san(chunk: str, pos: Position) -> Move | None:
    for san in reversed(SAN_RE.findall(chunk)):
        try:
            chess_move = pos.board.parse_san(san)
        except ValueError:
            continue
        return Move(chess_move.uci())
    return None


def parse_model_move(text: str, pos: Position) -> tuple[Move | None, str | None]:
    """有多个 <move> 时用最后一个标签。优先 UCI，其次 e2-e4，再用当前局面解析 SAN。"""
    chunks = TAG_RE.findall(text or "")
    chunk = chunks[-1] if chunks else (text or "")
    hits = UCI_RE.findall(chunk)
    if hits:
        return Move(hits[-1].lower()), "uci"
    hyphenated = HYPHEN_RE.findall(chunk)
    if hyphenated:
        return Move(_uci_from_hyphen(hyphenated[-1])), "uci"
    move = _from_san(chunk, pos)
    if move is not None:
        return move, "san"
    return None, None


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
        lines.append("No moves have been played.")
    if policy == "legal_list":
        lines.append("Legal moves: " + " ".join(legal))
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


class ChessGame:
    """国际象棋对局入口。规则判断使用 chess 套件。"""

    name = "chess"
    files = 8
    ranks = 8
    seats = (WHITE, BLACK)

    def initial(self) -> Position:
        return Position.initial()

    def opponent(self, seat: str) -> str:
        return BLACK if seat == WHITE else WHITE

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
        return "No playable move was found. Output one tag such as <move>e2e4</move>."

    def illegal_feedback(self, uci: str, detail: str | None) -> str:
        return f"Move {uci} cannot be played: {detail}"

    def request_feedback(self, exc: BaseException) -> str:
        return f"Request failed: {exc}"


CHESS = ChessGame()
