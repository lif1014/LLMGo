"""CPU contracts for the verl adapter and concurrent evaluation/recovery."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
import json
import pytest

from llmgo.training.verl_ppo import training_config, hydra_overrides, complete_checkpoint, dataset_rows
from llmgo.training.rollout import seed_curriculum
from llmgo.rsi import evaluator
from llmgo.rsi.contracts import atomic_json, digest, file_hash


def test_verl_resource_partition_and_configuration():
    cfg = training_config({},8)
    assert cfg['batch_size']==56 and cfg['minibatch_size']==14
    overrides = hydra_overrides('/weights','/data',Path('/output/training'),cfg,8,'/teacher')
    assert 'trainer.n_gpus_per_node=7' in overrides
    assert 'critic.enable=false' in overrides
    assert 'algorithm.adv_estimator="lc0_td0"' in overrides
    assert 'algorithm.rollout_correction.bypass_mode=true' in overrides
    assert 'actor_rollout_ref.rollout.n=1' in overrides
    assert '+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=true' in overrides
    for gpus in range(1,9):
        cfg = training_config({},gpus)
        assert cfg['minibatch_size']%(gpus-1 or 1)==0
    for gpus in (0,9,True):
        with pytest.raises(ValueError):training_config({},gpus)
    for config in ({'sft':True},{'backend':'hf_ppo'},{'group_size':8},{'batch_size':8},{'learning_rate':float('nan')}):
        with pytest.raises(ValueError):training_config(config)


def test_checkpoint_rejects_partial_rank_and_wrong_world(tmp_path):
    root=tmp_path/'global_step_3'
    (root/'actor/huggingface').mkdir(parents=True)
    (root/'actor/huggingface/config.json').write_text('{}')
    (root/'data.pt').write_text('state')
    for kind in ('model','optim','extra_state'):
        for i in range(2):
            (root/'actor'/f'{kind}_world_size_2_rank_{i}.pt').write_text('state')
    assert complete_checkpoint(tmp_path,2)==root
    assert complete_checkpoint(tmp_path,7) is None
    (root/'actor/optim_world_size_2_rank_1.pt').unlink()
    assert complete_checkpoint(tmp_path,2) is None


def test_dataset_is_positions_without_teacher_answers():
    rows=dataset_rows(seed_curriculum(3),[],14)
    assert len(rows)==14
    assert all(row['reward_model']['ground_truth']=='' for row in rows)
    assert all('moves' in json.loads(row['extra_info']['record_json']) for row in rows)
    assert all(row['prompt'][0]['role']=='user' for row in rows)


def test_parallel_slots_retry_only_incomplete_games_and_merge_in_order(tmp_path,monkeypatch):
    active=set();peak=0;calls={};servers=[]
    class Server:
        def __init__(self,model,p,log,port,gpu,**kw):
            self.base_url=f'gpu-{gpu}';servers.append(self)
        def start(self):pass
        def stop(self):pass
    helpers=SimpleNamespace(_pick_ports=lambda p,n:list(range(n)),VllmServer=Server,_wait_servers=lambda *a:None)
    async def play(payload,url,cards,output,*,indices):
        nonlocal peak
        index=next(iter(indices));calls[index]=calls.get(index,0)+1
        active.add(url);peak=max(peak,len(active))
        try:
            await asyncio.sleep(.01)
            if index==1 and calls[index]==1:raise RuntimeError('server failed')
            return [{'index':index}]
        finally:active.remove(url)
    monkeypatch.setattr(evaluator,'play_batch',play)
    payload={'protocol':{'games':10},'site':{'user_root':str(tmp_path)},'model_path':'model'}
    rows=asyncio.run(evaluator.parallel_batch(payload,None,tmp_path,helpers,[str(i) for i in range(8)]))
    assert [r['index'] for r in rows]==list(range(10))
    assert peak==8 and calls[1]==2 and all(n==1 for i,n in calls.items() if i!=1)
    assert len(servers)==9
    assert json.loads((tmp_path/'progress.json').read_text())['completed_games']==10
    assert (tmp_path/'attempts/game-01-1.json').is_file()


def test_resume_rejects_different_identity_or_tampered_raw_log(tmp_path):
    payload={'protocol':{},'seed_schedule':[123],'site':{'user_root':str(tmp_path)}}
    events=tmp_path/'events.jsonl';events.write_text('original')
    row={'evaluation_key':evaluator.evaluation_key(payload),'index':0,'pair_id':0,'seed':123,'model_seat':'white',
         'raw_events':str(events),'short_events':str(events),
         'event_hashes':{'raw_events':file_hash(events),'short_events':file_hash(events)}}
    row['integrity']=digest(row);path=tmp_path/'game.json';atomic_json(path,row)
    assert evaluator.checked_game(path,payload,0)==row
    with pytest.raises(ValueError):evaluator.checked_game(path,{**payload,'track':'model'},0)
    events.write_text('changed')
    with pytest.raises(ValueError):evaluator.checked_game(path,payload,0)


def test_parallel_does_not_publish_partial_batch(tmp_path,monkeypatch):
    class Server:
        def __init__(self,*a,**kw):self.base_url='url'
        def start(self):pass
        def stop(self):pass
    helpers=SimpleNamespace(_pick_ports=lambda p,n:list(range(n)),VllmServer=Server,_wait_servers=lambda *a:None)
    async def play(*args,**kwargs):raise RuntimeError('failure')
    monkeypatch.setattr(evaluator,'play_batch',play)
    with pytest.raises(RuntimeError,match='incomplete'):
        asyncio.run(evaluator.parallel_batch({'protocol':{'games':10},'model_path':'m','site':{}},None,tmp_path,helpers,['0']))
    assert not (tmp_path/'metrics.json').exists()


@pytest.mark.parametrize("transformers_version",["5.2.0","5.4.0"])
def test_rope_compat_is_scoped_idempotent_and_does_not_mutate_callers(monkeypatch,transformers_version):
    import sys
    import importlib.metadata
    from types import ModuleType
    from llmgo.training.vllm_compat import install_rope_compat
    monkeypatch.setattr(importlib.metadata,'version',lambda name: {'transformers':transformers_version,'vllm':'0.17.0'}[name])
    class Config:
        def convert_rope_params_to_dict(self,ignore_keys_at_rope_validation=None,**kwargs):
            return ignore_keys_at_rope_validation,kwargs
    module=ModuleType('transformers.configuration_utils');module.PretrainedConfig=Config
    monkeypatch.setitem(sys.modules,'transformers.configuration_utils',module)
    Legacy=type('Qwen3_5TextConfig',(Config,),{'__module__':'vllm.transformers_utils.configs.qwen3_5'})
    Other=type('OtherConfig',(Config,),{'__module__':'transformers.models.other'})
    keys=['mrope_section','mrope_interleaved']
    assert install_rope_compat()
    wrapped=Config.convert_rope_params_to_dict
    assert install_rope_compat() and Config.convert_rope_params_to_dict is wrapped
    assert Legacy().convert_rope_params_to_dict(keys,foo=1)==(set(keys),{'foo':1})
    assert Other().convert_rope_params_to_dict(keys)[0] is keys
    assert keys==['mrope_section','mrope_interleaved']


@pytest.mark.parametrize("gpus",range(1,9))
@pytest.mark.parametrize("multiplier",[1,2,8])
def test_rollout_worker_count_always_divides_training_batch(gpus,multiplier):
    actors=max(gpus-1,1)
    cfg=training_config({"batch_size":actors*multiplier,"minibatch_size":actors},gpus)
    args=hydra_overrides('/weights','/data',Path('/out'),cfg,gpus,'/teacher')
    workers=int(next(x.split('=',1)[1] for x in args if x.startswith('actor_rollout_ref.rollout.agent.num_workers=')))
    assert 1<=workers<=4 and cfg['batch_size']%workers==0



def test_parallel_upgrade_keeps_legacy_evidence_but_reruns_unverified_games(tmp_path,monkeypatch):
    old=tmp_path/'old';output=tmp_path/'new'
    atomic_json(old/'game_results/00.json',{'game_id':'legacy-00','result':'draw'})
    original=(old/'game_results/00.json').read_bytes()
    class Server:
        def __init__(self,*a,**kw):self.base_url='url'
        def start(self):pass
        def stop(self):pass
    helpers=SimpleNamespace(_pick_ports=lambda p,n:list(range(n)),VllmServer=Server,_wait_servers=lambda *a:None)
    called=[]
    async def play(*args,indices,**kwargs):
        index=next(iter(indices));called.append(index);return [{'index':index}]
    monkeypatch.setattr(evaluator,'play_batch',play)
    payload={'protocol':{'games':10},'site':{'user_root':str(tmp_path)},'model_path':'model','resume_from':str(old)}
    rows=asyncio.run(evaluator.parallel_batch(payload,None,output,helpers,['0']))
    assert sorted(called)==list(range(10)) and len(rows)==10
    assert (old/'game_results/00.json').read_bytes()==original
    assert json.loads((output/'resume-skipped.json').read_text())['games'][0]['reason']=='legacy_unverified_receipt'


def test_upgrade_skips_failed_receipts_and_changed_evaluator_without_touching_history(tmp_path,monkeypatch):
    old=tmp_path/'old';output=tmp_path/'new'
    failed={'integrity':'old','evaluation_key':'old','result':'black_win','result_reason':'harness_error'}
    changed={'integrity':'old','evaluation_key':'old','result':'white_win','result_reason':'checkmate'}
    atomic_json(old/'game_results/00.json',failed);atomic_json(old/'game_results/01.json',changed)
    class Server:
        def __init__(self,*a,**kw):self.base_url='url'
        def start(self):pass
        def stop(self):pass
    helpers=SimpleNamespace(_pick_ports=lambda p,n:list(range(n)),VllmServer=Server,_wait_servers=lambda *a:None)
    called=[]
    async def play(*args,indices,**kwargs):
        index=next(iter(indices));called.append(index);return [{'index':index}]
    monkeypatch.setattr(evaluator,'play_batch',play)
    payload={'protocol':{'games':10},'site':{'user_root':str(tmp_path)},'model_path':'model','resume_from':str(old)}
    asyncio.run(evaluator.parallel_batch(payload,None,output,helpers,['0']))
    assert called==list(range(10))
    skips=json.loads((output/'resume-skipped.json').read_text())['games']
    assert [x['reason'] for x in skips]==['infrastructure_failure','evaluation_identity_changed']
    assert json.loads((old/'game_results/00.json').read_text())==failed
    with pytest.raises(ValueError,match='Infrastructure'):
        evaluator.checked_game(old/'game_results/00.json',payload,0)
