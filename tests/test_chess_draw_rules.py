"""Draw policy shared by validation and held-out games."""
from types import SimpleNamespace

import chess
import pytest

from llmgo.env.rules import chess_outcome
from llmgo.rsi.contracts import MAX_GAME_PLIES, protocol, validate_protocol
from llmgo.rsi.game import ChessEpisode
from llmgo.rsi.ladder import opponent


def test_fivefold_repetition_ends_game_but_threefold_does_not():
    board=chess.Board()
    for cycle in range(4):
        for move in ['g1f3','g8f6','f3g1','f6g8']:board.push_uci(move)
        result=chess_outcome(board)
        if cycle<3:assert result is None
    assert result.winner is None and result.termination==chess.Termination.FIVEFOLD_REPETITION


@pytest.mark.parametrize('plies,finished',[(200,False),(299,False),(300,True)])
def test_protocol_cap_is_three_hundred_halfmoves(monkeypatch,plies,finished):
    monkeypatch.setattr('llmgo.rsi.game.chess_outcome',lambda board:None)
    episode=SimpleNamespace(board=SimpleNamespace(move_stack=[None]*plies),done=False)
    def finish(score,reason):episode.done=True;episode.reason=reason;assert score==0
    episode.finish=finish
    assert ChessEpisode.boundary(episode)==finished
    if finished:assert episode.reason=='protocol_cap_draw'


def test_checkmate_takes_precedence_at_cap(monkeypatch):
    monkeypatch.setattr('llmgo.rsi.game.chess_outcome',lambda board:chess.Outcome(chess.Termination.CHECKMATE,chess.WHITE))
    episode=SimpleNamespace(board=SimpleNamespace(move_stack=[None]*300),done=False,actor=chess.WHITE)
    def finish(score,reason):episode.done=True;assert score==1 and reason=='checkmate'
    episode.finish=finish
    assert ChessEpisode.boundary(episode)


def test_validation_protocol_records_new_cap():
    value=protocol(opponent(2),'fixture',games=40)
    assert value['max_plies']==MAX_GAME_PLIES==300
    validate_protocol(value)
