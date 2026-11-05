import json
from pathlib import Path
from llmgo.rsi.contracts import read_json
from llmgo.rsi.random_moves import count_game,write_report


def test_executed_moves_are_counted_once_and_original_evidence_is_preserved(tmp_path):
    directory=tmp_path/'final-test/games/deepseek-final-00';directory.mkdir(parents=True)
    path=directory/'events.jsonl'
    events=[{'type':'fallback','reason':'missing_box'},
        {'type':'student_move','source':'random','fallback_reason':'missing_box'},
        {'type':'student_move','source':'model'}, {'type':'forced_move'},
        {'type':'opponent_move','diagnostics':{'fallback_reason':None}},
        {'type':'opponent_move','diagnostics':{'fallback_reason':'conflicting_boxed_moves','raw_output':r'\boxed{e4} \boxed{d4}'}},
        {'type':'episode_finished'}]
    original=''.join(json.dumps(e)+'\n' for e in events).encode();path.write_bytes(original)
    row=count_game(path)
    assert row['student_random_moves']==row['opponent_random_moves']==1
    assert row['student_moves']==row['opponent_moves']==2
    assert row['student_fallback_reasons']=={'missing_box':1}
    assert row['opponent_fallback_reasons']=={'conflicting_boxed_moves':1}
    assert row['completed'] and row['cursor']==len(original)
    result=write_report(tmp_path/'final-test')
    assert result['student_total']==result['opponent_total']==1
    assert read_json(tmp_path/'final-test/random-moves.json')['games'][directory.name]==row
    assert path.read_bytes()==original


def test_partial_last_record_is_not_counted_until_it_is_written(tmp_path):
    path=tmp_path/'events.jsonl'
    first=b'{"type":"student_move","source":"model"}\n'
    partial=b'{"type":"student_move","source":"random"'
    path.write_bytes(first+partial)
    assert count_game(path)['cursor']==len(first)
    assert count_game(path)['student_random_moves']==0
    path.write_bytes(first+partial+b'}\n')
    assert count_game(path)['student_random_moves']==1
