import chess
import pytest
from llmgo.training.chess_r1 import extract_answer
from llmgo.rsi.deepseek_client import extract_loose_answer


@pytest.mark.parametrize('text,san,error',[
    (r'\boxed{e4}','e4',None),
    (r'I choose \boxed{e4}.','e4',None),
    (r'\boxed{e4} because it controls the center.','e4',None),
    (r'\boxed{e4} then \boxed{d4}','d4',None),
    (r'\boxed{e5} then \boxed{e4}','e4',None),
    (r'\boxed{e4} then \boxed{e5}','e5','illegal_or_noncanonical_san'),
    (r'\boxed{e4} then \boxed{e2e4}','e2e4','illegal_or_noncanonical_san'),
    (r'\boxed{e4} then \boxed{d4',None,'unclosed_box'),
    (r'\boxed{e4} then \boxed{}',None,'not_one_san'),
    (r'Final: \boxed{\text{d4}}.','d4',None),
    (r'<answer>\boxed{e4}</answer>','e4',None),
    ('e4',None,'missing_box'),
])
def test_both_models_use_exactly_the_same_last_box_policy(text,san,error):
    qwen=extract_answer(text,chess.STARTING_FEN)
    deepseek=extract_loose_answer(text,chess.STARTING_FEN)
    assert qwen==deepseek
    assert qwen.san==san and qwen.error==error
    assert qwen.legal==(error is None)


@pytest.mark.parametrize('finish',['length','max_tokens'])
def test_truncation_still_rejects_the_last_box(finish):
    for parser in [extract_answer,extract_loose_answer]:
        assert parser(r'I choose \boxed{e4}.',chess.STARTING_FEN,finish_reason=finish).error=='truncated'
