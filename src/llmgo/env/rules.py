"""Shared terminal rules for play, move previews and offline training."""
from __future__ import annotations

import chess


def chess_outcome(board: chess.Board) -> chess.Outcome | None:
    """Automatic fivefold repetition; retain the existing fifty-move claim policy."""
    outcome = board.outcome(claim_draw=False)
    if outcome is None and board.can_claim_fifty_moves():
        return chess.Outcome(chess.Termination.FIFTY_MOVES, None)
    return outcome
