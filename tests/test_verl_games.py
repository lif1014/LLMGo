"""Real verl DataProto tests, including retry/padding/reward identities."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('verl')
import numpy as np
from verl import DataProto
from llmgo.training.verl_games import game_batch, object_array
from llmgo.training.verl_ppo import training_config
from llmgo.training.quality import rollout_quality


def sample(n=2):
    mask=torch.tensor([[1]*n+[0]*(4-n)])
    return DataProto.from_dict(tensors={'responses':torch.tensor([[10,11,12,13]]),
        'prompts':torch.tensor([[1,2]]),'input_ids':torch.tensor([[1,2,10,11,12,13]]),
        'attention_mask':torch.tensor([[1,1]+[1]*n+[0]*(4-n)]),
        'position_ids':torch.arange(6)[None,:],'response_mask':mask,
        'rollout_log_probs':torch.zeros((1,4))},
        non_tensors={'multi_modal_inputs':object_array([{}])})


def episode(identifier,turns,result):
    rows=[]
    for i,requests in enumerate(turns):
        rows.append({'episode_id':identifier,'policy_version':'p1','student_color':'white',
            'turn_index':i,'value':0.,'reward':float(result) if i==len(turns)-1 else 0.,
            'done':i==len(turns)-1,'requests':[{'request_key':key,'response_tokens':n,
                'failure_reason':'invalid_final_answer' if j==0 and len(requests)>1 else ''}
                for j,(key,n) in enumerate(requests)]})
    return SimpleNamespace(done=True,aborted=False,transitions=rows,id=identifier)


def test_padding_never_corrupts_real_requests_or_counts_as_invalid():
    episodes=[episode('a',[[('a0',2)],[('a1',3)]],1),episode('b',[[('b0',4)]],-1)]
    tensors={key:sample(n) for key,n in [('a0',2),('a1',3),('b0',4)]}
    batch=game_batch(episodes,tensors,training_config({'minibatch_size':7}),scorer=None)
    assert len(batch)==7 and batch.meta_info['real_requests']==3
    assert batch.batch['response_mask'].sum(-1).tolist()==[2,3,4,0,0,0,0]
    assert batch.batch['token_level_rewards'].sum(-1).tolist()==[0,1,-1,0,0,0,0]
    assert batch.batch['advantages'].count_nonzero()==0  # validation cannot optimize
    quality=rollout_quality(batch,max_response_length=4)
    assert quality['samples']==quality['valid']==3 and quality['invalid']==0
    assert quality['score_mean']==0


def test_game_gae_is_mapped_to_requests_and_retry_reward_only_once(monkeypatch,tmp_path):
    import ray
    class Score:
        @staticmethod
        def remote(metadata):return [{'value':.2} for row in metadata]
    monkeypatch.setattr(ray,'get',lambda x:x)
    e=episode('game',[[('first',2),('retry',4)],[('last',3)]],1)
    for row in e.transitions:row['record']={'initial_fen':'unused','moves':[]}
    tensors={key:sample(n) for key,n in [('first',2),('retry',4),('last',3)]}
    batch=game_batch([e],tensors,training_config({'minibatch_size':7,'gae_lambda':1.}),
        scorer=SimpleNamespace(score_game=Score()),output=tmp_path)
    advantages=batch.batch['advantages']
    per=(advantages*batch.batch['response_mask']).sum(-1)/batch.batch['response_mask'].sum(-1).clamp(min=1)
    # Overall average request loss = average of two equally weighted moves.
    assert float(per.mean())==pytest.approx(.8)
    assert batch.batch['token_level_rewards'].sum(-1).tolist()==[0,0,1,0,0,0,0]
    assert batch.non_tensor_batch['raw_advantage'][:3].tolist()==pytest.approx([.8,.8,.8])
    assert batch.batch['response_mask'].sum(-1).tolist()==[2,4,3,0,0,0,0]


def test_random_student_action_keeps_original_tokens_probabilities_and_loss(monkeypatch,tmp_path):
    import ray
    class Score:
        @staticmethod
        def remote(metadata):return [{'value':.2} for row in metadata]
    monkeypatch.setattr(ray,'get',lambda x:x)
    e=episode('fallback',[[('failed-output',2)]],1)
    turn=e.transitions[0]
    turn.update(record={'initial_fen':'unused','moves':[]},action_source='random',uci='e2e4',
                model_uci=None,fallback_reason='attempts_exhausted')
    turn['requests'][0]['failure_reason']='invalid_final_answer'
    original=sample(2)
    original.batch['rollout_log_probs'][0,:2]=torch.tensor([-.2,-.3])
    batch=game_batch([e],{'failed-output':original},training_config({'minibatch_size':7}),
        scorer=SimpleNamespace(score_game=Score()),output=tmp_path)
    assert batch.batch['responses'][0,:2].tolist()==[10,11]
    assert batch.batch['rollout_log_probs'][0,:2].tolist()==pytest.approx([-.2,-.3])
    assert batch.batch['response_mask'][0].sum()==2
    assert float(batch.batch['advantages'].sum()/batch.batch['response_mask'].sum()/7)==pytest.approx(.8)
    assert batch.non_tensor_batch['action_source'][0]=='random'
    assert batch.non_tensor_batch['executed_uci'][0]=='e2e4'
    assert rollout_quality(batch,max_response_length=4)['valid']==0


def test_collector_random_fallback_continues_standard_game_until_real_checkmate(monkeypatch,tmp_path):
    import chess
    from llmgo.training.game import ChessEpisode
    from llmgo.training.verl_games import complete_games
    class Fairy:
        def __init__(self,*a,**kw):pass
        def _play_board(self,board):return ('e7e5' if board.fullmove_number==1 else 'd8h4'),{}
        def close(self):pass
    class Picker:
        def choice(self,legal):
            return next(m for m in legal if m.uci()==('f2f3' if any(x.uci()=='f2f3' for x in legal) else 'g2g4'))
    def game(*args,**kwargs):
        e=ChessEpisode(*args,**kwargs,engine_factory=Fairy);e.rng=Picker();return e
    monkeypatch.setattr('llmgo.training.verl_games.ChessEpisode',game)
    class Generator:
        def generate_sequences(self,inputs):
            outputs=[]
            for extra in inputs.non_tensor_batch['extra_info']:
                row=sample(2)
                row.non_tensor_batch['request_receipt']=object_array([{'finish_reason':'stop','seconds':.01}])
                outputs.append(row)
            return DataProto.concat(outputs)
    trainer=SimpleNamespace(config=SimpleNamespace(research=training_config({'attempts_per_move':1,'concurrent_games':1})),
        async_rollout_manager=Generator(),tokenizer=SimpleNamespace(decode=lambda ids,**kw:'</think> still no answer'))
    episodes,tensors,_=complete_games(trainer,{'output':str(tmp_path),'identity':'test','cards':[],
        'site':{'engine':'fake'}},count=1,step=1)
    e=episodes[0]
    assert e.reason=='checkmate' and e.result==-1 and len(e.transitions)==2
    assert [r['uci'] for r in e.transitions]==['f2f3','g2g4']
    assert [r['reward'] for r in e.transitions]==[0.,-1.]
    assert all(r['action_source']=='random' and r['model_uci'] is None for r in e.transitions)
    assert all(r['requests'][0]['response']=='</think> still no answer' for r in e.transitions)
    assert len(tensors)==2 and e.board.is_checkmate()


@pytest.mark.parametrize('step,colors',[
    (1,['white','black','white','black','white','black','white']),
    (2,['black','white','black','white','black','white','black']),
])
def test_seven_game_collector_completes_one_cohort_with_rotating_colors(monkeypatch,tmp_path,step,colors):
    import chess
    from llmgo.training.game import ChessEpisode
    from llmgo.training.verl_games import complete_games
    class Fairy:
        def __init__(self,*a,**kw):pass
        def _play_board(self,board):
            moves=('f2f3','g2g4') if board.turn==chess.WHITE else ('e7e5','d8h4')
            uci=moves[board.fullmove_number-1]
            assert chess.Move.from_uci(uci) in board.legal_moves
            return uci,{}
        def close(self):pass
    monkeypatch.setattr('llmgo.training.verl_games.ChessEpisode',
        lambda *a,**kw:ChessEpisode(*a,**kw,engine_factory=Fairy))
    answers={17:'</think> f2f3',18:'</think> g2g4',19:'</think> e7e5',20:'</think> d8h4'}
    cohort_sizes=[]
    class Generator:
        def generate_sequences(self,inputs):
            cohort_sizes.append(len(inputs));rows=[]
            for extra in inputs.non_tensor_batch['extra_info']:
                index=int(extra['episode_id'].rsplit('-',1)[1])
                code=(17 if colors[index]=='white' else 19)+extra['turn_index']
                row=sample(2);row.batch['responses'][0,:2]=torch.tensor([code,99])
                row.non_tensor_batch['request_receipt']=object_array([{'finish_reason':'stop','seconds':.01}])
                rows.append(row)
            return DataProto.concat(rows)
    trainer=SimpleNamespace(config=SimpleNamespace(research=training_config({})),
        async_rollout_manager=Generator(),tokenizer=SimpleNamespace(decode=lambda ids,**kw:answers[ids[0]]))
    episodes,tensors,_=complete_games(trainer,{'output':str(tmp_path),'identity':'test','cards':[],
        'site':{'engine':'fake'}},count=7,step=step)
    assert [e.color for e in episodes]==colors
    assert len(episodes)==7 and len(tensors)==14 and cohort_sizes==[7,7]
    assert all(e.done and e.reason=='checkmate' and e.board.is_checkmate() for e in episodes)
    assert all(e.initial_fen==chess.STARTING_FEN and len(e.transitions)==2 for e in episodes)
    assert [e.result for e in episodes]==[-1 if color=='white' else 1 for color in colors]
    assert len({e.policy for e in episodes})==1


def test_hydra_config_and_actual_verl_worker_contract(tmp_path):
    import verl
    from hydra import compose,initialize_config_dir
    from llmgo.training.verl_ppo import hydra_overrides
    from verl.trainer.ppo.utils import need_critic,need_reference_policy
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    cfg=training_config({'games_per_batch':2,'concurrent_games':2,'steps':1})
    with initialize_config_dir(config_dir=str(Path(verl.__file__).parent/'trainer/config'),version_base=None):
        actual=compose(config_name='ppo_trainer',overrides=hydra_overrides(
            '/user/liyaohu4170/models/Qwen3.5-4B','/data',tmp_path,cfg,8,tmp_path/'teacher.json'))
    assert not need_critic(actual) and need_reference_policy(actual)
    assert actual.algorithm.lam==.95 and actual.algorithm.gamma==1
    for method in ('_update_actor','_compute_old_log_prob','_compute_ref_log_prob','_save_checkpoint'):
        assert callable(getattr(RayPPOTrainer,method))


def test_request_agent_keeps_actual_behavior_probabilities(monkeypatch):
    import asyncio
    from llmgo.training.verl_agent import ChessTurnAgentLoop
    captured=[]
    class Server:
        async def generate(self,**kw):
            captured.append(kw)
            return SimpleNamespace(token_ids=[7,8],log_probs=[-.2,-.3],stop_reason='completed',num_preempted=0)
    agent=object.__new__(ChessTurnAgentLoop)
    agent.server_manager=Server()
    agent.tokenizer=SimpleNamespace(eos_token_id=8,decode=lambda ids,**kw:"raw </think> e2e4")
    agent.rollout_config=SimpleNamespace(prompt_length=20)
    async def template(messages):return [1,2,3]
    agent.apply_chat_template=template
    result=asyncio.run(agent.run({'temperature':1.,'top_p':1.,'top_k':-1,'logprobs':True},
        raw_prompt=[{'role':'user','content':'board'}],extra_info={'seed':42,'max_tokens':4,'request_timeout_sec':10}))
    assert result.response_logprobs==[-.2,-.3] and result.response_ids==[7,8]
    assert captured[0]['sampling_params']['seed']==42
    assert result.extra_fields['request_receipt']['finish_reason']=='stop'


def test_collector_plays_from_start_to_opponent_checkmate_and_keeps_all_requests(monkeypatch,tmp_path):
    import json
    import chess
    from llmgo.training.game import ChessEpisode
    from llmgo.training.verl_games import complete_games
    class Fairy:
        def __init__(self,*a,**kw):pass
        def _play_board(self,board):
            uci='e7e5' if board.fullmove_number==1 else 'd8h4'
            assert chess.Move.from_uci(uci) in board.legal_moves
            return uci,{}
        def close(self):pass
    def game(*a,**kw):return ChessEpisode(*a,**kw,engine_factory=Fairy)
    monkeypatch.setattr('llmgo.training.verl_games.ChessEpisode',game)
    answers={17:'</think>\nf2f3',18:'</think>\ng2g4'}
    class Generator:
        def generate_sequences(self,inputs):
            rows=[]
            for extra in inputs.non_tensor_batch['extra_info']:
                code=17 if extra['turn_index']==0 else 18
                row=sample(2)
                row.batch['responses'][0,:2]=torch.tensor([code,99])
                row.non_tensor_batch['request_receipt']=object_array([{
                    'request_id':str(code),'finish_reason':'stop','sampling':{'temperature':1},
                    'seconds':.01,'completion_tokens':2}])
                rows.append(row)
            return DataProto.concat(rows)
    trainer=SimpleNamespace(config=SimpleNamespace(research=training_config({'games_per_batch':2,'concurrent_games':1})),
        async_rollout_manager=Generator(),tokenizer=SimpleNamespace(decode=lambda ids,**kw:answers[ids[0]]))
    # Exercise one complete standard-start game, with real production collector/board.
    context={'output':str(tmp_path),'identity':'test','cards':[],'site':{'engine':'fake'}}
    episodes,tensors,_=complete_games(trainer,context,count=1,step=1)
    e=episodes[0]
    assert e.initial_fen==chess.STARTING_FEN and e.done and e.result==-1
    from llmgo.rsi.contracts import digest
    assert e.policy==digest({'identity':'test','completed_updates':0})
    assert [r['reward'] for r in e.transitions]==[0.,-1.]
    assert len(tensors)==2 and e.board.is_checkmate()
    events=[json.loads(line) for line in (e.output/'events.jsonl').read_text().splitlines()]
    assert sum(x['type']=='model_request_finished' for x in events)==2
    assert any(x['type']=='opponent_move' and x['uci']=='d8h4' for x in events)


def test_zero_reward_manager_accepts_actual_verl_loader_kwargs():
    import asyncio
    from llmgo.training.verl_reward import ChessRewardManager
    manager=ChessRewardManager(config=SimpleNamespace(),tokenizer=object(),compute_score=None,
        reward_router_address='unused',num_examine=0)
    assert asyncio.run(manager.run_single(None))=={'reward_score':0.,'reward_extra_info':{}}


def test_native_verl_actor_accepts_expanded_game_batch_and_whole_batch_metadata(tmp_path):
    import verl
    from hydra import compose,initialize_config_dir
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.utils import tensordict_utils as tu
    from tensordict import TensorDict
    from llmgo.training.verl_ppo import hydra_overrides
    cfg=training_config({'games_per_batch':2,'minibatch_size':7,'steps':1})
    with initialize_config_dir(config_dir=str(Path(verl.__file__).parent/'trainer/config'),version_base=None):
        actual=compose(config_name='ppo_trainer',overrides=hydra_overrides(
            '/weights','/data',tmp_path,cfg,8,tmp_path/'teacher.json'))
    ep=episode('one',[[('a',2)],[('b',4)]],1)
    batch=game_batch([ep],{'a':sample(2),'b':sample(4)},cfg)
    batch.batch['old_log_probs']=batch.batch['rollout_log_probs'].clone()
    seen={}
    class Worker:
        def update_actor(self,data):
            for key in ('global_batch_size','mini_batch_size','epochs'):
                seen[key]=tu.get(data,key)
            result=TensorDict({},batch_size=[])
            tu.assign_non_tensor(result,metrics={'mfu':0.,'grad_norm':1.})
            return result
    trainer=object.__new__(RayPPOTrainer)
    trainer.config=actual;trainer.distillation_config=None;trainer.actor_rollout_wg=Worker()
    trainer.config.actor_rollout_ref.actor.ppo_mini_batch_size=len(batch)
    result=trainer._update_actor(batch)
    assert seen['global_batch_size']==seen['mini_batch_size']==len(batch)
    assert seen['epochs']==1 and result.meta_info['metrics']['actor/grad_norm']==1.


def test_agent_completed_output_survives_a_failed_sibling(monkeypatch,tmp_path):
    import asyncio
    import json
    from llmgo.training.verl_agent import ChessTurnAgentLoop
    monkeypatch.setenv('LLMGO_VERL_OUTPUT',str(tmp_path))
    def make(fail=False):
        class Server:
            async def generate(self,**kw):
                if fail:raise ConnectionError('server disconnected')
                return SimpleNamespace(token_ids=[7,8],log_probs=[-.2,-.3],stop_reason='completed',num_preempted=0)
        agent=object.__new__(ChessTurnAgentLoop);agent.server_manager=Server()
        agent.tokenizer=SimpleNamespace(eos_token_id=8,decode=lambda ids,**kw:'raw </think> e2e4')
        agent.rollout_config=SimpleNamespace(prompt_length=20)
        async def template(messages):return [1,2,3]
        agent.apply_chat_template=template
        return agent
    async def run():
        return await asyncio.gather(*[make(fail).run({'temperature':1.,'logprobs':True},
            raw_prompt=[{'role':'user','content':'board'}],extra_info={
                'seed':42,'max_tokens':4,'request_timeout_sec':10,'episode_id':'same-cohort',
                'turn_index':0,'attempt':0}) for fail in [False,True]],return_exceptions=True)
    outcomes=asyncio.run(run())
    assert isinstance(outcomes[1],ConnectionError)
    logs=[json.loads(p.read_text()) for p in (tmp_path/'agent_requests').glob('*.json')]
    assert {row['status'] for row in logs}=={'generation_completed','failed_before_token_output'}
    good=next(row for row in logs if row['status']=='generation_completed')
    assert good['response_ids']==[7,8] and good['behavior_logprobs']==[-.2,-.3]
    assert good['response']=='raw </think> e2e4' and good['receipt']['finish_reason']=='stop'


@pytest.mark.parametrize('validation_enabled',[False,True])
@pytest.mark.parametrize('resume_step',[0,1])
@pytest.mark.parametrize('games_per_batch',[2,7])
def test_fit_updates_checkpoints_and_resumes_with_optional_internal_validation(
        monkeypatch,tmp_path,validation_enabled,resume_step,games_per_batch):
    import json
    import ray
    from llmgo.rsi.contracts import atomic_json,read_json
    from llmgo.training import verl_games
    from llmgo.training.validation import validation_identity
    from llmgo.training.stability import select_export_checkpoint
    settings=training_config({'steps':2,'games_per_batch':games_per_batch,'internal_validation':validation_enabled})
    settings['game_context']=str(tmp_path/'context.json')
    atomic_json(tmp_path/'context.json',{'output':str(tmp_path),'identity':'experiment','cards':[]})
    atomic_json(tmp_path/'lc0-context.json',{'namespace':'test','name':'lc0'})
    if resume_step and validation_enabled:
        atomic_json(tmp_path/'validation-baseline.json',{'passed':True,'stage':'baseline','valid_rate':1.,
            'validation_identity':validation_identity(settings,'experiment')})
    class Score:
        @staticmethod
        def remote(metadata):return [{'value':.2} for row in metadata]
    monkeypatch.setattr(ray,'get_actor',lambda *a,**kw:SimpleNamespace(score_game=Score()))
    monkeypatch.setattr(ray,'get',lambda x:x)
    calls=[];updates=[]
    def collect(trainer,context,*,count,step,validation=False):
        assert count==(settings['validation_games'] if validation else games_per_batch)
        calls.append(('validation' if validation else 'train',step))
        episodes=[];tensors={}
        for i in range(count):
            key=f'{validation}-{step}-{i}'
            e=episode(key,[[(key,2)]],1)
            e.result=1.;e.reason='checkmate';e.output=tmp_path/'episodes'/key
            e.transitions[0]['record']={'initial_fen':'unused','moves':[]}
            atomic_json(e.output/'summary.json',{'decisions':1,'random_moves':0})
            episodes.append(e);tensors[key]=sample(2)
        return episodes,tensors,.01
    monkeypatch.setattr(verl_games,'complete_games',collect)
    trainer=SimpleNamespace(config=SimpleNamespace(research=settings,
        algorithm=SimpleNamespace(rollout_correction=None),
        actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(ppo_mini_batch_size=14,policy_loss=None))),
        checkpoint_manager=SimpleNamespace(update_weights=lambda step:None,sleep_replicas=lambda:None),
        use_reference_policy=True,tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda t:11,eos_token_id=13))
    def load():trainer.global_steps=resume_step
    trainer._load_checkpoint=load
    trainer._compute_ref_log_prob=lambda batch:DataProto.from_dict(tensors={
        'ref_log_prob':torch.zeros_like(batch.batch['rollout_log_probs'])})
    trainer._compute_old_log_prob=lambda batch:(DataProto.from_dict(tensors={
        'old_log_probs':torch.zeros_like(batch.batch['rollout_log_probs'])}),None)
    def update(batch):
        assert trainer.config.actor_rollout_ref.actor.ppo_mini_batch_size==len(batch)
        updates.append(trainer.global_steps)
        return SimpleNamespace(meta_info={'metrics':{'actor/grad_norm':.2}})
    trainer._update_actor=update
    def save():
        path=tmp_path/'checkpoints'/f'global_step_{trainer.global_steps}'
        atomic_json(path/'actor/huggingface/config.json',{})
        (path/'data.pt').write_text('state')
        for kind in ('model','optim','extra_state'):
            for rank in range(7):
                (path/'actor'/f'{kind}_world_size_7_rank_{rank}.pt').write_text('shard')
    trainer._save_checkpoint=save
    monkeypatch.setattr('verl.trainer.ppo.rollout_corr_helper.apply_bypass_mode',lambda **kw:None)
    verl_games.fit_games(trainer)
    assert updates==list(range(resume_step+1,3))
    assert [step for split,step in calls if split=='train']==updates
    validations=[step for split,step in calls if split=='validation']
    assert validations==(([0] if not resume_step else [])+updates if validation_enabled else [])
    state=read_json(tmp_path/'checkpoints/global_step_2/games.json')
    assert state['internal_validation'] is validation_enabled and state['update']['step']==2
    assert (state['validation_baseline'] is not None) is validation_enabled
    assert (tmp_path/'validation-baseline.json').exists() is validation_enabled
    assert read_json(tmp_path/'internal-validation.json')['enabled'] is validation_enabled
    chosen,receipt=select_export_checkpoint(tmp_path,7,
        internal_validation=validation_enabled,experiment_digest='experiment')
    assert chosen.name=='global_step_2'
    assert receipt['passed'] is (True if validation_enabled else None)
