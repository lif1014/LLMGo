import json
from pathlib import Path

import pytest

from llmgo.rsi.checkpoint_elo import POSITIVE_DP, performance
from llmgo.rsi.checkpoint_benchmark import collect_entry, request, replay_index
from llmgo.rsi.contracts import atomic_json


@pytest.mark.parametrize('wdl,elo',[( (8,10,2),1910),((0,20,0),1800),((40,0,0),2600),((0,0,40),1000),((30,0,10),1993)])
def test_fide_table_and_endpoints(wdl,elo):
    assert performance(*wdl)['performance_rating']==elo


def test_no_games_has_no_rating_and_full_table_is_monotone():
    assert performance(0,0,0) is None
    assert len(POSITIVE_DP)==51
    ratings=[performance(p,0,100-p)['performance_rating'] for p in range(101)]
    assert ratings==sorted(ratings) and all(ratings[p]+ratings[100-p]==3600 for p in range(101))
    assert performance(1,1,38)['table_percent']==4  # 3.75%, closest integer percent.


def test_partial_games_do_not_publish_a_rating(tmp_path):
    entry={'id':'base','run_id':'run','expected_sha256':'abc'}
    atomic_json(tmp_path/'run/final-test/game_results/00.json',{'index':0,'result':1})
    row=collect_entry(entry,tmp_path,{'games':40})
    assert row['completed_games']==1 and row['wins']==1 and row['score_rate']==1
    assert row['elo'] is None and not row['formal']


def test_formal_rating_requires_full_population_and_checkpoint_identity(tmp_path):
    entry={'id':'base','run_id':'run','expected_sha256':'abc'}
    config={'games':40,'parallel_games':40,'teacher_rating':1800,'opponent':{'base_url':'http://fixture'}}
    final=tmp_path/'run/final-test'
    for i in range(40):atomic_json(final/f'game_results/{i:02d}.json',{'index':i,'result':0})
    atomic_json(final/'result.json',{'status':'completed','games':40,'wins':0,'draws':40,'losses':0})
    protocol={'student_model_identity':{'sha256':'abc'},'answer_parser':'last-box-san-v1',
        'opponent_answer_parser':'last-box-san-v1','parallel_games':40,'student_vllm_max_num_seqs':40,
        'opponent':{'model':'deepseek-v4.1-flash','base_url':'http://fixture','thinking':'disabled'},
        'student_protocol':{'enable_thinking':False},'max_plies':300,'fivefold_repetition':True}
    atomic_json(final/'protocol.json',protocol)
    row=collect_entry(entry,tmp_path,config)
    assert row['formal'] and row['elo']==1800
    protocol['student_model_identity']['sha256']='wrong'
    atomic_json(final/'protocol.json',protocol)
    with pytest.raises(ValueError,match='identity'):collect_entry(entry,tmp_path,config)


def test_six_jobs_use_the_requested_pool_priority_and_concurrency():
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/evaluation/checkpoint-elo-20261106.json').read_text())
    assert len(config['entries'])==6 and config['parallel_games']==config['games']==40
    config['repo']='/user/lyh4170/LLMGo!'
    for entry in config['entries']:
        job=request(config,entry)
        assert job['resourcePool']=='model-new' and job['priority']=='PREEMPTABLE'
        assert job['gpuModel']=='H800' and job['resources']['gpuCount']==1
        assert 'worker' in job['entry'] and '--entry '+entry['id'] in job['entry']
        import shlex
        assert shlex.split(job['entry'])[0]==config['worker_python']
        assert '/llmgo-rsi-control-20261022/bin/python' in shlex.split(job['entry'])[0]


def test_replay_index_keeps_all_forty_games_for_older_web_servers(tmp_path):
    for i in range(40):
        path=tmp_path/f'final-test/games/deepseek-final-{i:02d}/events.jsonl'
        path.parent.mkdir(parents=True);path.write_text('{}\n')
    replay_index(tmp_path)
    index=json.loads((tmp_path/'final-test/replay-index.json').read_text())
    assert len(index['files'])==40
    assert index['files'][0]['path'].endswith('deepseek-final-00/events.jsonl')
    assert index['files'][-1]['path'].endswith('deepseek-final-39/events.jsonl')
