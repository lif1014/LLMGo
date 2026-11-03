"""Actual CSV-to-Parquet boundaries: solver turns, held-outs and selection."""
import csv
import io
import json

import chess
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zstandard

from llmgo.data.lichess import prepare, report_training_data, select, verify
from llmgo.training.chess_r1 import ANSWER_PROTOCOL, messages

SCHEMA = pa.schema([
    ('data_source', pa.string()),
    ('prompt', pa.list_(pa.struct([('role',pa.string()),('content',pa.string())]))),
    ('ability',pa.string()),
    ('reward_model',pa.struct([('style',pa.string()),('ground_truth',pa.string())])),
    ('extra_info',pa.struct([('split',pa.string()),('puzzle_id',pa.string()),
        ('rating',pa.int64()),('solver_index',pa.int64()),('answer_protocol',pa.string())])),
])


def sample(pid, moves):
    board = chess.Board()
    board.push_uci(moves[0])
    prompt = messages(board.fen(), enable_thinking=False)
    gt = {'id':pid,'board_fen':board.fen(),'next_move_san':board.san(chess.Move.from_uci(moves[1])),
          'rating':1000,'legal_moves_san':prompt[1]['content'].split('\nLegal moves: ')[1]}
    return {'data_source':'fixture','prompt':prompt,'ability':'chess',
        'reward_model':{'style':'rule','ground_truth':json.dumps(gt)},
        'extra_info':{'split':'dev','puzzle_id':pid,'rating':1000,'solver_index':0,
                      'answer_protocol':ANSWER_PROTOCOL}}


@pytest.fixture
def corpus(tmp_path):
    reference = tmp_path/'reference.parquet'
    pq.write_table(pa.Table.from_pylist([],schema=SCHEMA),reference)
    heldouts=[]
    for name,pid,moves in [('dev','held',['e2e4','e7e5']),('test','test',['a2a3','a7a6'])]:
        path=tmp_path/(name+'.parquet')
        pq.write_table(pa.Table.from_pylist([sample(pid,moves)],schema=SCHEMA),path)
        heldouts.append(path)
    records=[]
    for pid,moves in [('held','g1f3 d7d5'),('fen-alias','e2e4 e7e5'),
                      ('train','d2d4 d7d5 c2c4 e7e6'),('duplicate','d2d4 d7d5 c2c4 e7e6'),
                      ('invalid','e2e5 e7e5'),('late-invalid','c2c4 e7e5 a1a8')]:
        fen=chess.STARTING_FEN
        if pid=='fen-alias':
            fen=fen.replace('0 1','8 99')
        records.append({'PuzzleId':pid,'FEN':fen,'Moves':moves,'Rating':'1200',
            'RatingDeviation':'70','Popularity':'90','NbPlays':'100',
            'Themes':'fork middlegame','GameUrl':'https://lichess.org/fixture',
            'OpeningTags':'Queens_Pawn_Game','DailyDate':''})
    text=io.StringIO()
    writer=csv.DictWriter(text,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    raw=tmp_path/'raw.csv.zst'
    raw.write_bytes(zstandard.ZstdCompressor().compress(text.getvalue().encode()))
    return raw,tmp_path/'pool',reference,heldouts


@pytest.mark.parametrize('workers',[1,2])
def test_full_pool_excludes_whole_holdout_puzzles_alias_positions_and_invalid_trajectories(corpus,workers):
    raw,pool,reference,heldouts=corpus
    result=prepare(raw,pool,reference,heldouts,workers=workers)
    assert result['counts']['raw_puzzles']==6
    assert result['counts']['excluded_holdout_id_puzzles']==1
    assert result['counts']['excluded_holdout_position_puzzles']==1
    assert result['counts']['invalid_puzzles']==2
    assert result['counts']['duplicate_positions']==2
    assert result['counts']['positions']==2
    rows=pq.read_table(pool/'train.parquet').to_pylist()
    labels=[json.loads(row['reward_model']['ground_truth']) for row in rows]
    assert [gt['next_move_san'] for gt in labels]==['d5','e6']
    assert {gt['id'] for gt in labels}=={'train'}
    assert all(gt['board_fen'].split()[1]=='b' for gt in labels)
    assert pq.read_schema(pool/'train.parquet').equals(pq.read_schema(reference))
    assert not (pool/'dev.parquet').exists() and not (pool/'test.parquet').exists()
    proof=verify(pool,reference,heldouts)
    assert proof['rows']==2 and proof['holdout_position_overlap']==0


def test_selection_produces_compatible_training_and_four_delivery_artifacts(corpus):
    raw,pool,reference,heldouts=corpus
    prepare(raw,pool,reference,heldouts)
    verify(pool,reference,heldouts)
    output=pool.parent/'selection'
    result=select(pool,output,hypothesis='Target missed forks from losing games',
                  rating_min=1000,rating_max=1400,themes=['fork'],max_positions=1,evidence=['game/events.jsonl'])
    assert result['positions']==1 and result['puzzles']==1
    assert (output/'selected_ids.txt').read_text()=='train\n'
    assert (output/'selected_positions.txt').read_text()=='train:1\n'
    assert json.loads((output/'stats.json').read_text())['themes']['fork']==1
    assert pq.read_schema(output/'train.parquet').equals(pq.read_schema(reference))
    assert 'seed' not in result['query']
    own_query = pool.parent/'chosen-sample-ids.txt'
    own_query.write_text('train:3\n')
    precise = pool.parent/'precise-selection'
    select(pool,precise,hypothesis='Use a custom query ID list',sample_ids=own_query)
    precise_gt=json.loads(pq.read_table(precise/'train.parquet').to_pylist()[0]['reward_model']['ground_truth'])
    assert precise_gt['next_move_san']=='e6'
    retained = pool.parent/'retained-default'
    report = report_training_data([pool/'train.parquet'],retained,hypothesis='Keep the default data')
    assert report['positions']==2 and report['puzzles']==1
    assert (retained/'selected_ids.txt').read_text()=='train\n'
    with pytest.raises(FileExistsError):
        select(pool,output,hypothesis='Do not overwrite evidence')


def test_verification_rejects_changed_heldout_and_pool_bytes(corpus):
    raw,pool,reference,heldouts=corpus
    prepare(raw,pool,reference,heldouts)
    path=pool/'train.parquet'
    with path.open('ab') as stream:stream.write(b'changed')
    with pytest.raises((ValueError,pa.ArrowInvalid)):
        verify(pool,reference,heldouts)
