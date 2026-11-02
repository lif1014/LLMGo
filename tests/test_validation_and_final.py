import chess

from llmgo.rsi.deepseek_client import extract_loose_answer
from llmgo.rsi.deepseek_final import DeepSeekPlayer, OPPONENT
from llmgo.rsi.ladder import evaluation_plan, opponent, panel_summary


def test_validation_is_l2_to_l3_and_requires_both_anchors():
    assert evaluation_plan()['levels'] == [2, 3]
    assert evaluation_plan({'levels': [2.5]})['levels'] == [2.5]
    assert opponent(2.5)['skill'] == -3
    for level in (1, 4):
        try:
            opponent(level)
        except ValueError:
            pass
        else:
            raise AssertionError('Non-validation level admitted')
    rows = [{'level': level, 'statistics': {'score': score, 'games': 20}}
            for level, score in ((2, .4), (2.5, .9), (3, .6))]
    result = panel_summary(rows)['player']
    assert result['complete'] and result['score'] == .5
    assert panel_summary(rows[:1])['player']['score'] is None


def test_deepseek_opponent_uses_loose_box_and_records_fallback(monkeypatch):
    board = chess.Board()
    assert OPPONENT['thinking'] == 'disabled'
    assert extract_loose_answer('I choose \\boxed{e4}.', board.fen()).uci == 'e2e4'
    monkeypatch.setattr('llmgo.rsi.deepseek_final.deepseek_request',
        lambda messages, request_id: {'choices': [{'message': {'content': '\\boxed{e4}'},
            'finish_reason': 'stop'}], 'usage': {'completion_tokens': 12}})
    player = DeepSeekPlayer('white', 'example', 7)
    move, detail = player._play_board(board)
    assert move == 'e2e4' and detail['fallback_reason'] is None
    assert detail['raw_output'] == '\\boxed{e4}'
    monkeypatch.setattr('llmgo.rsi.deepseek_final.deepseek_request',
        lambda messages, request_id: {'choices': [{'message': {'content': 'invalid'},
            'finish_reason': 'stop'}]})
    move, detail = player._play_board(board)
    assert chess.Move.from_uci(move) in board.legal_moves
    assert detail['fallback_reason'] == 'missing_box'
