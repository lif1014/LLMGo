import copy
from pathlib import Path

import pytest

from llmgo.rsi.contracts import atomic_json, digest, file_hash
from llmgo.training.baseline_import import import_baseline
from llmgo.training.verl_ppo import training_config


def fixture(root):
    source=root/'source';new=root/'new';old=root/'old/training'
    source.mkdir();new.mkdir();old.mkdir(parents=True)
    code={}
    for name in ('game.py','rollout.py','trajectories.py','quality.py','vllm_compat.py','verl_agent.py','verl_games.py'):
        p=source/'src/llmgo/training'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('def complete_games():\n    pass\n' if name=='verl_games.py' else '# fixture\n')
        q=new/p.relative_to(source);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
        code[name]=file_hash(p)
    h=source/'src/llmgo/harness/prompt.py';h.parent.mkdir(parents=True);h.write_text('# harness')
    identity={'model':'base','objective':'full_game_ppo_lc0','cards':'cards','reward':{},'runtime':{},
        'allocated_gpus':8,'verl':'.8','lc0_binary':'lc0','training':training_config({'internal_validation':True}),
        'code':code,'harness':{'prompt.py':file_hash(h)},
        'fairy':{'binary_sha256':'engine','nnue_sha256':'nnue'}}
    prior=copy.deepcopy(identity);prior.pop('harness');prior.pop('fairy')
    prior['training'].update(quality_gate_min_valid=.9,validation_min_valid=.9,validation_max_drop=.05)
    atomic_json(old/'identity.json',prior)
    payload={'candidate_source':{'root':str(source),'files':{
        p.relative_to(source).as_posix():file_hash(p) for p in source.rglob('*') if p.is_file()}}}
    manifest=root/'manifest.json';atomic_json(manifest,{'payload':payload,'sha256':digest(payload)})
    policy=digest({'identity':digest(prior),'completed_updates':0})
    for i,color in enumerate(('white','black')):
        folder=old/'episodes'/f'validation-00000000-{i:04d}'
        summary={'completed':True,'aborted':False,'student_color':color,'requests':1,'result':-1,'reason':'protocol_forfeit'}
        atomic_json(folder/'summary.json',summary)
        atomic_json(folder/'trajectory.json',{'complete':True,'training':False,'advantages_used_for_training':False,
            'transitions':[{'policy_version':policy,'student_color':color,'turn_index':0,'done':True,'reward':-1,
                'requests':[{'failure_reason':'' if i==0 else 'invalid_final_answer','response_tokens':10}]}]})
        (folder/'events.jsonl').write_text('{}\n')
    atomic_json(old/'quality-latest.json',{'step':0,'stage':'validation','samples':2,'valid':1,'valid_rate':.5,'response_tokens_total':20})
    atomic_json(old/'update-diagnostics-latest.json',{'step':0,'reference_kl_move_mean':.001})
    atomic_json(old/'sync-check-latest.json',{'step':0,'mean_abs_logprob_delta':.01})
    return old,identity,{'previous_output':old.parent,'destination':root/'out','identity':identity,
        'previous_manifest':manifest,'current_source':new,'engine_evidence':identity['fairy']}


def test_reuse_preserves_raw_observations_and_cannot_overwrite(tmp_path):
    old,identity,args=fixture(tmp_path)
    before={str(p):file_hash(p) for p in old.rglob('*') if p.is_file()}
    row=import_baseline(**args)
    assert row['imported'] and row['stage']=='baseline' and row['valid_rate']==.5
    assert row['losses']==2 and row['score']==0
    assert before=={str(p):file_hash(p) for p in old.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='empty'):import_baseline(**args)


def test_older_validation_schema_without_redundant_training_flag(tmp_path):
    import json
    old,identity,args=fixture(tmp_path)
    for p in (old/'episodes').glob('*/trajectory.json'):
        row=json.loads(p.read_text());row.pop('advantages_used_for_training');atomic_json(p,row)
    assert import_baseline(**args)['imported']


@pytest.mark.parametrize('change',['model','harness','sampling','engine','environment','partial','updated','quality','agent','collector'])
def test_mismatched_or_post_update_evidence_is_rejected(tmp_path,change):
    old,identity,args=fixture(tmp_path)
    if change=='model':identity['model']='another'
    elif change=='harness':identity['harness']={}
    elif change=='sampling':identity['training']['max_response_tokens']=8192
    elif change=='engine':args['engine_evidence']={**identity['fairy'],'binary_sha256':'other'}
    elif change=='environment':identity['environment']={'invalid_move_policy':'random_legal'}
    elif change=='partial':atomic_json(old/'episodes/validation-00000000-0000/summary.json',{'completed':False})
    elif change=='updated':(old/'metrics.jsonl').write_text('{"stage":"update_completed"}\n')
    elif change=='quality':atomic_json(old/'quality-latest.json',{'step':0,'stage':'validation','samples':2,'valid':2,'valid_rate':1.})
    elif change=='agent':identity['code']['verl_agent.py']='another'
    elif change=='collector':(args['current_source']/'src/llmgo/training/verl_games.py').write_text('def complete_games():\n    return 1\n')
    with pytest.raises(ValueError):import_baseline(**args)
