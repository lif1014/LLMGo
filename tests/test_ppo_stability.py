import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmgo.rsi.contracts import atomic_json
from llmgo.training.stability import (TrainingSafetyStop, enforce_update_safety,
    select_validated_checkpoint, select_export_checkpoint)
from llmgo.training.verl_ppo import training_config, hydra_overrides


def checkpoint(root, step, actors=2):
    path = root/'checkpoints'/f'global_step_{step}'
    (path/'actor/huggingface').mkdir(parents=True)
    (path/'actor/huggingface/config.json').write_text('{}')
    (path/'data.pt').write_text('data')
    for kind in ('model','optim','extra_state'):
        for rank in range(actors):
            (path/'actor'/f'{kind}_world_size_{actors}_rank_{rank}.pt').write_text('shard')
    return path




def test_actual_reference_kl_stop_blocks_before_update(tmp_path):
    cfg=training_config({})
    quality={'step':3,'valid':56,'valid_rate':1.}
    sync={'mean_abs_logprob_delta':.01}
    enforce_update_safety(quality,{'reference_kl_move_mean':.02},sync,cfg,tmp_path)
    with pytest.raises(TrainingSafetyStop,match='reference_kl_gate'):
        enforce_update_safety(quality,{'reference_kl_move_mean':.11},sync,cfg,tmp_path)
    row=json.loads((tmp_path/'stability-stop.json').read_text())
    assert row['evidence']['stage']=='before_update' and row['evidence']['step']==3
    for value in (None,float('nan')):
        with pytest.raises(TrainingSafetyStop):
            enforce_update_safety(quality,{'reference_kl_move_mean':value},sync,cfg,tmp_path)


def test_missing_sync_evidence_is_not_silently_accepted(tmp_path):
    with pytest.raises(TrainingSafetyStop,match='actor_rollout_sync_gate'):
        enforce_update_safety({'step':2,'valid':56,'valid_rate':1.},
            {'reference_kl_move_mean':0.}, {'error':'failed'},training_config({}),tmp_path)


def test_all_failed_batch_is_refused_even_when_the_rate_gate_is_disabled(tmp_path):
    with pytest.raises(TrainingSafetyStop,match='rollout_quality_gate'):
        enforce_update_safety({'step':2,'valid':0,'valid_rate':0.},{},None,
            training_config({'quality_gate_min_valid':0.}),tmp_path)










def test_signed_sync_ratio_reproduces_the_positive_side_miss(tmp_path):
    torch=pytest.importorskip('torch')
    from llmgo.training.quality import sync_check
    class Trainer:
        global_steps=1
        def _compute_old_log_prob(self,batch):
            return SimpleNamespace(batch={'old_log_probs':torch.tensor([[.21,-.21]])}),None
    b=SimpleNamespace(batch={'rollout_log_probs':torch.zeros(1,2),'response_mask':torch.ones(1,2)})
    row=sync_check(Trainer(),b,samples=1,output=tmp_path)
    assert row['fraction_ratio_outside_0.8_1.2']==.5
    assert row['mean_signed_logprob_delta']==pytest.approx(0.)


def test_diagnostics_measure_length_weighting_without_calling_it_gradients():
    torch=pytest.importorskip('torch')
    from llmgo.training.stability import batch_diagnostics
    b=SimpleNamespace(batch={'advantages':torch.tensor([[.1,.1,.1],[-1.,0.,0.]]),
        'response_mask':torch.tensor([[1.,1.,1.],[1.,0.,0.]]),
        'old_log_probs':torch.zeros(2,3),'ref_log_prob':torch.zeros(2,3)},
        non_tensor_batch={'failure_reason':['','length']})
    d=batch_diagnostics(b)
    assert d['valid']['positive_token_coefficient_mass']==pytest.approx(.3)
    assert d['invalid']['negative_token_coefficient_mass']==1.
    assert d['reference_kl_move_mean']==0.


def test_only_complete_post_update_game_validated_checkpoint_can_export(tmp_path):
    atomic_json(tmp_path/'validation-baseline.json',{'validation_identity':'fixed','valid_rate':1.})
    first=checkpoint(tmp_path,1);second=checkpoint(tmp_path,2)
    for step,path in [(1,first),(2,second)]:
        atomic_json(path/'games.json',{'step':step})
        atomic_json(tmp_path/'validation'/f'step-{step:08d}.json',{
            'step':step,'validation_identity':'fixed','stage':'after_update',
            'passed':True,'valid_rate':1.,'score':.5 if step==1 else 1.})
    chosen,row=select_validated_checkpoint(tmp_path,2)
    assert chosen==second and row['step']==2
    (second/'actor/model_world_size_2_rank_1.pt').unlink()
    assert select_validated_checkpoint(tmp_path,2)[0]==first
    atomic_json(tmp_path/'validation/step-00000001.json',{'step':1,'stage':'baseline','passed':True,
        'validation_identity':'fixed','valid_rate':1.,'score':1.})
    with pytest.raises(RuntimeError,match='post-update'):select_validated_checkpoint(tmp_path,2)


def test_resume_skips_weights_without_committed_game_state(tmp_path):
    from llmgo.training.verl_ppo import complete_checkpoint
    first=checkpoint(tmp_path,1);second=checkpoint(tmp_path,2)
    atomic_json(first/'games.json',{'step':1,'identity':'policy'})
    assert complete_checkpoint(tmp_path/'checkpoints',2)==second
    assert complete_checkpoint(tmp_path/'checkpoints',2,require_game_state=True)==first


def test_validation_disabled_exports_latest_complete_updated_checkpoint_without_baseline(tmp_path):
    for step in (1,2):
        path=checkpoint(tmp_path,step)
        atomic_json(path/'games.json',{'step':step,'identity':'experiment','internal_validation':False,
            'validation_baseline':None,'update':{'step':step,'stage':'update_completed',
                'metrics':{'training/global_step':step,'actor/grad_norm':.2}}})
    chosen,row=select_export_checkpoint(tmp_path,2,internal_validation=False,experiment_digest='experiment')
    assert chosen.name=='global_step_2' and row['passed'] is None
    assert row['enabled'] is False and row['formal_evaluation_required'] is True
    (chosen/'actor/optim_world_size_2_rank_1.pt').unlink()
    assert select_export_checkpoint(tmp_path,2,internal_validation=False,
        experiment_digest='experiment')[0].name=='global_step_1'
    assert not (tmp_path/'validation-baseline.json').exists()


@pytest.mark.parametrize('change',['identity','missing_update','step','nonfinite','validation_mode','baseline'])
def test_validation_disabled_refuses_uncommitted_or_incompatible_update(tmp_path,change):
    path=checkpoint(tmp_path,1)
    state={'step':1,'identity':'experiment','internal_validation':False,'validation_baseline':None,
        'update':{'step':1,'stage':'update_completed','metrics':{'training/global_step':1,'actor/loss':.1}}}
    if change=='identity':state['identity']='other'
    elif change=='missing_update':state.pop('update')
    elif change=='step':state['update']['step']=2
    elif change=='nonfinite':state['update']['metrics']['actor/loss']=float('inf')
    elif change=='validation_mode':state['internal_validation']=True
    elif change=='baseline':state['validation_baseline']={'passed':True}
    # Preserve a nonfinite historical receipt to exercise the reader's rejection.
    (path/'games.json').write_text(json.dumps(state))
    with pytest.raises(RuntimeError,match='finite optimizer update'):
        select_export_checkpoint(tmp_path,2,internal_validation=False,experiment_digest='experiment')
