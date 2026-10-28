"""Exercise draw boundaries through the public playing and training interfaces."""
from pathlib import Path

import chess
import chess.pgn
import pytest

from llmgo.env.chess_game import Move, Position
from llmgo.env.rules import chess_outcome
from llmgo.harness.tools import preview_move
from llmgo.training.game import ChessEpisode
from llmgo.training.lc0 import terminal_value
from llmgo.training.rollout import board_from_record


def repeated_board(plies):
    board = chess.Board()
    cycle = ('g1f3', 'g8f6', 'f3g1', 'f6g8')
    for ply in range(plies):
        board.push_uci(cycle[ply % 4])
    return board


@pytest.mark.parametrize('plies', [7, 8, 12, 15])
def test_possible_and_actual_threefold_and_fourfold_continue(plies):
    board = repeated_board(plies)
    assert board.can_claim_threefold_repetition()
    assert board.is_repetition(3) is (plies >= 8)
    assert chess_outcome(board) is None
    assert Position(board)._result() == (None, None)
    assert terminal_value(board, chess.WHITE) is None
    record = {'initial_fen': chess.STARTING_FEN,
              'moves': [move.uci() for move in board.move_stack]}
    assert board_from_record(record).fen() == board.fen()


def test_real_training_game_continues_before_and_after_third_repetition(tmp_path):
    fixture = Path(__file__).parent / 'fixtures/threefold-1380000-step4-0001.pgn'
    with fixture.open() as stream:
        game = chess.pgn.read_game(stream)
    board = game.end().board()
    assert len(board.move_stack) == 93
    assert board.fen() == '2r3k1/5p1p/6pq/3pp3/p6b/r4P1P/1K3P1R/6R1 b - - 11 47'
    assert not board.is_repetition(3) and board.can_claim_threefold_repetition()
    assert chess_outcome(board) is None

    before = board.copy()
    last = before.pop()
    assert last.uci() == 'a1b2'
    assert preview_move(before, last)['terminal'] is None
    after, result, reason = Position(before).apply(Move(last.uci()))
    assert after.fen() == board.fen() and result is None and reason is None

    class UnusedEngine:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    episode = ChessEpisode('regression', 'new-policy', 'white', 3, 'unused',
                           tmp_path / 'episode', engine_factory=UnusedEngine)
    try:
        episode.board = board.copy()
        assert not episode.boundary() and not episode.done
        episode.board.push_uci('a3b3')
        assert episode.board.is_repetition(3)
        assert not episode.boundary() and not episode.done
    finally:
        episode.close()


def test_fivefold_still_ends_play_and_training(tmp_path):
    board = repeated_board(16)
    assert chess_outcome(board).termination == chess.Termination.FIVEFOLD_REPETITION
    assert Position(board)._result() == ('draw', 'repetition')
    assert terminal_value(board, chess.WHITE) == .5
    before = board.copy()
    last = before.pop()
    assert preview_move(before, last)['terminal'] == 'fivefold_repetition'
    with pytest.raises(ValueError, match='nonterminal'):
        board_from_record({'initial_fen': chess.STARTING_FEN,
                           'moves': [move.uci() for move in board.move_stack]})


@pytest.mark.parametrize(('fen', 'termination'), [
    ('7k/6Q1/5K2/8/8/8/8/8 b - - 0 1', chess.Termination.CHECKMATE),
    ('7k/5K2/6Q1/8/8/8/8/8 b - - 0 1', chess.Termination.STALEMATE),
    ('7k/8/8/8/8/8/8/K7 w - - 0 1', chess.Termination.INSUFFICIENT_MATERIAL),
    ('7k/8/8/8/8/8/R7/K7 w - - 99 51', chess.Termination.FIFTY_MOVES),
    ('7k/8/8/8/8/8/R7/K7 w - - 100 51', chess.Termination.FIFTY_MOVES),
    ('7k/8/8/8/8/8/R7/K7 w - - 150 76', chess.Termination.SEVENTYFIVE_MOVES),
])
def test_other_terminal_rules_are_preserved(fen, termination):
    board = chess.Board(fen)
    assert board.is_valid()
    assert chess_outcome(board).termination == termination
    assert chess_outcome(board) == board.outcome(claim_draw=True)
