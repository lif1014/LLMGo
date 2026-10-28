"""Shared terminal rules for play, move previews and offline training."""
from __future__ import annotations

import chess


def chess_outcome(board: chess.Board) -> chess.Outcome | None:
    """Ignore threefold claims; retain automatic outcomes and fifty-move draws.

    In particular, a possible or actual third repetition does not end a game.
    Automatic fivefold repetition and the existing fifty-move policy remain.
    """
    outcome = board.outcome(claim_draw=False)
    if outcome is None and board.can_claim_fifty_moves():
        return chess.Outcome(chess.Termination.FIFTY_MOVES, None)
    return outcome
