"""Online complete-game sampling and variable-length move batching for verl 0.8.

The actor, reference, optimizer and distributed checkpoints remain native verl.
Only environment orchestration and move-axis advantage construction live here.
"""
from __future__ import annotations

import concurrent.futures
import json
import math
from pathlib import Path
import time

from llmgo.harness.choices import parse_move
from llmgo.harness.config import HarnessConfig
from llmgo.harness.tools import TOOL_LIBRARY
from llmgo.rsi.contracts import atomic_json, digest, read_json, utc
from .game import ChessEpisode
from .rollout import prompt_for_record
from .trajectories import game_gae, request_weights
from .validation import validation_identity, minimum_valid_rate


def object_array(items):
    import numpy as np
    result = np.empty(len(items),dtype=object)
    result[:] = items
    return result


def complete_games(trainer, context, *, count, step, validation=False):
    """No policy updates until ALL games from this version have finished."""
    import numpy as np
    import torch
    from verl import DataProto
    settings = dict(trainer.config.research)
    out = Path(context['output'])
    split = 'validation' if validation else 'train'
    policy = digest({'identity':context['identity'],
                     'completed_updates':step if validation else step-1})
    episodes, active, pending, tensors = [], [], {}, {}
    seed_base = settings['validation_seed'] if validation else settings['seed']+step*100000
    started = time.monotonic()
    cfg = HarnessConfig.from_dict()

    def progress(stage='game_rollout'):
        summaries = [{'episode_id':e.id,'student_color':e.color,'plies':len(e.board.move_stack),
            'decisions':len(e.transitions),'completed':e.done,'result':e.result,'reason':e.reason}
            for e in episodes]
        for row,e in zip(summaries,episodes,strict=True):
            row['random_moves'] = sum(t.get('action_source')=='random' for t in e.transitions)
        atomic_json(out/'game-progress.json',{'stage':stage,'step':step,'split':split,
            'completed_games':sum(e.done for e in episodes),'total_games':count,'games':summaries[-32:],
            'omitted_games':max(0,len(summaries)-32),'policy_version':policy,
            'completed_updates':step if validation else step-1,
            'elapsed_seconds':time.monotonic()-started,'updated':utc()})
        atomic_json(out.parent/'progress.json',{'stage':stage,'step':step,'split':split,
            'completed_games':sum(e.done for e in episodes),'total_games':count,
            'message':f'{split}完整对局 {sum(e.done for e in episodes)}/{count}',
            'progress_path':str(out/'game-progress.json'),'updated':utc()})

    def make_request(episode, state):
        cap = min(settings['max_response_tokens'],settings['move_token_budget']-state['spent'])
        remaining = settings['turn_timeout_sec']-(time.monotonic()-state['started'])
        if cap < 1 or remaining <= 0:
            episode.act(None,state['requests'],fallback_reason='turn_budget_exhausted' if cap<1 else 'turn_timeout')
            pending.pop(episode.id)
            return None
        prompt = prompt_for_record(episode.record(),context['cards'],cfg,correction=state['correction'])
        attempt = len(state['requests'])
        extra = {'seed':int(digest({'seed':episode.seed,'turn':len(episode.transitions),'attempt':attempt})[:8],16)%(2**31),
            'max_tokens':cap,'request_timeout_sec':min(remaining,settings['request_timeout_sec']),
            'episode_id':episode.id,'turn_index':len(episode.transitions),'attempt':attempt}
        episode.event('model_request_started',prompt=prompt['prompt'],skills=prompt['skills'],
            candidates=prompt['candidates'],observations=prompt['observations'],tool_library=TOOL_LIBRARY,
            request=extra,policy_version=policy)
        return episode,state,prompt,extra

    try:
        while sum(e.done for e in episodes) < count:
            while len(active) < settings['concurrent_games'] and len(episodes)<count:
                i = len(episodes)
                episode_id = f'{split}-{step:08d}-{i:04d}'
                episode = ChessEpisode(episode_id,policy,'white' if i%2==0 else 'black',
                    settings['opponent_level'],context['site']['engine'],out/'episodes'/episode_id,
                    seed=seed_base+i//2)
                episodes.append(episode);episode.advance()
                if not episode.done:active.append(episode)
            active = [e for e in active if not e.done]
            if not active:continue
            progress()
            requests = []
            for episode in active:
                state = pending.setdefault(episode.id,{'requests':[],'spent':0,'started':time.monotonic(),'correction':None})
                request = make_request(episode,state)
                if request:requests.append(request)
            if not requests:continue
            inputs = DataProto.from_dict(non_tensors={
                'raw_prompt':object_array([p['messages'] for _,_,p,_ in requests]),
                'extra_info':object_array([extra for *_,extra in requests]),
                'index':np.arange(len(requests))},meta_info={'global_steps':step,'validate':validation})
            generated = trainer.async_rollout_manager.generate_sequences(inputs)
            if len(generated) != len(requests):raise ValueError('Rollout count mismatch')
            for i,(episode,state,prompt,extra) in enumerate(requests):
                sample = generated[i:i+1]
                receipt = sample.non_tensor_batch['request_receipt'][0]
                length = int(sample.batch['response_mask'].sum())
                ids = sample.batch['responses'][0,:length].tolist()
                text = trainer.tokenizer.decode(ids,skip_special_tokens=False)
                uci = None if receipt['finish_reason']=='length' or '</think>' not in text else parse_move(text,set(prompt['legal_moves']))
                failure = ('length' if receipt['finish_reason']=='length' else 'incomplete_thinking'
                    if '</think>' not in text else 'invalid_final_answer' if uci is None else '')
                key = f'{episode.id}-{len(episode.transitions):04d}-{len(state["requests"]):02d}'
                path = out/'requests'/f'{key}.pt';path.parent.mkdir(exist_ok=True)
                if path.exists():raise FileExistsError('Request trace already exists')
                # Store actual tensors before flattening; never retokenize the response.
                sample.meta_info = {}
                sample.non_tensor_batch = {'multi_modal_inputs':object_array([{}])}
                tensors[key] = sample
                torch.save({'tensors':dict(sample.batch)},path)
                row = {**receipt,'request_key':key,'tensor_path':str(path),'response_tokens':length,
                    'prompt':prompt['prompt'],'prompt_ids':sample.batch['prompts'][0][sample.batch['attention_mask'][0,:sample.batch['prompts'].shape[1]].bool()].tolist(),
                    'response':text,'response_ids':ids,'behavior_logprobs':sample.batch['rollout_log_probs'][0,:length].tolist(),
                    'failure_reason':failure,'uci':uci,'skills':prompt['skills']}
                state['requests'].append(row);state['spent']+=length
                episode.event('model_request_finished',**row)
                exhausted = (len(state['requests']) >= settings['attempts_per_move']
                    or state['spent']>=settings['move_token_budget'])
                if uci is not None or exhausted:
                    episode.act(uci,state['requests'],fallback_reason='turn_budget_exhausted'
                        if state['spent']>=settings['move_token_budget'] else 'attempts_exhausted')
                    pending.pop(episode.id)
                else:
                    state['correction']='Previous response did not yield a legal final move. Finish thinking and output one listed UCI move.'
                episode.write_summary()
            progress()
        if any(e.aborted or not e.transitions for e in episodes):
            raise ValueError('Complete-game batch has an aborted or empty episode')
        progress('game_rollout_completed')
        return episodes,tensors,time.monotonic()-started
    except BaseException as error:
        for episode in episodes:
            if not episode.done:episode.abort(f'{type(error).__name__}: {error}')
        raise
    finally:
        for episode in episodes:episode.close()


def game_batch(episodes, tensors, settings, *, scorer=None, output=None):
    """Assign terminal r once; compute GAE over moves before flattening requests."""
    import numpy as np
    import torch
    from verl import DataProto
    transitions = []
    if len({row['policy_version'] for e in episodes for row in e.transitions}) != 1:
        raise ValueError('Different behavior policy versions in one PPO batch')
    for episode in episodes:
        rows = episode.transitions
        if not episode.done or episode.aborted:raise ValueError('Unfinished game cannot train')
        if scorer is not None:
            import ray
            metadata = [{k:row[k] for k in ('episode_id','turn_index','policy_version','student_color','record')} for row in rows]
            values = ray.get(scorer.score_game.remote(metadata))
            if len(values) != len(rows):raise ValueError('Missing teacher states')
            rows = [{**row,**v,'value_source':'lc0_native_wdl'} for row,v in zip(rows,values,strict=True)]
        else:
            # Validation receives zero advantages; its games are NEVER optimized.
            rows = [{**row,'value':0.,'value_source':'not_evaluated_validation_placeholder'} for row in rows]
        rows = game_gae(rows,gamma=settings['gamma'],lam=settings['gae_lambda'])
        if output:
            atomic_json(Path(output)/'episodes'/episode.id/'trajectory.json',{
                'episode_id':episode.id,'complete':True,'training':scorer is not None,
                'advantages_used_for_training':scorer is not None,
                'gamma':settings['gamma'],'gae_lambda':settings['gae_lambda'],'transitions':rows})
        transitions.extend(rows)
    weights = request_weights(transitions)
    samples, extra = [], []
    for turn,factors in zip(transitions,weights,strict=True):
        for j,(request,factor) in enumerate(zip(turn['requests'],factors,strict=True)):
            sample = tensors[request['request_key']]
            mask = sample.batch['response_mask']
            reward = turn['reward'] if j==len(turn['requests'])-1 else 0.
            scores = torch.zeros_like(mask,dtype=torch.float32)
            scores[0,int(mask.sum())-1] = reward
            sample.batch['token_level_scores']=scores
            sample.batch['token_level_rewards']=scores.clone()
            sample.batch['advantages']=mask.float()*(turn['advantage']*factor if scorer is not None else 0.)
            sample.batch['returns']=mask.float()*turn['return']
            sample.batch['values']=mask.float()*turn['value']
            extra.append({'episode_id':turn['episode_id'],'turn_index':turn['turn_index'],
                'failure_reason':request['failure_reason'],'value_before':turn['value'],
                'action_source':turn.get('action_source','model'),'executed_uci':turn.get('uci'),
                'fallback_reason':turn.get('fallback_reason'),
                'environment_reward':reward,'raw_advantage':turn['advantage'],'delta':turn['delta'],
                'move_weight':factor,'request_key':request['request_key'],'padded':False})
            samples.append(sample)
    batch = DataProto.concat(samples)
    for key in extra[0]:batch.non_tensor_batch[key] = object_array([row[key] for row in extra])
    n = len(batch);target = math.ceil(n/settings['minibatch_size'])*settings['minibatch_size']
    if target>n:
        padding = batch[:1].repeat(repeat_times=target-n,interleave=True)
        for key in ('response_mask','token_level_scores','token_level_rewards','advantages','returns','values'):
            padding.batch[key].zero_()
        padding.non_tensor_batch['padded'] = object_array([True]*(target-n))
        batch = DataProto.concat([batch,padding])
    # Equal mean over real moves, independent of how many zero-weight pads are needed.
    batch.batch['advantages'] *= target/n
    batch.meta_info = {'temperature':1.,'global_token_num':batch.batch['attention_mask'].sum(-1).tolist(),
                       'images_seqlens':[],'real_requests':n,'real_moves':len(transitions)}
    return batch


def fit_games(trainer):
    """Full-game PPO driver; native verl methods perform all neural operations."""
    import torch
    import ray
    from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode
    from verl.trainer.ppo.ray_trainer import reduce_metrics
    from .quality import rollout_quality, sync_check
    from .stability import batch_diagnostics, enforce_update_safety, safety_stop
    settings = dict(trainer.config.research)
    context = read_json(settings['game_context'])
    out = Path(context['output'])
    teacher = read_json(out/'lc0-context.json')
    scorer = ray.get_actor(teacher['name'],namespace=teacher['namespace'])
    trainer.global_steps = 0
    trainer._load_checkpoint()
    trainer.checkpoint_manager.update_weights(trainer.global_steps)
    trainer.max_steps_duration = 0.
    baseline_path = out/'validation-baseline.json'

    def persist_metrics(metrics, stage, **extra):
        row = {'step':int(trainer.global_steps),'utc':utc(),'backend':'verl_ppo',
            'mode':'full_game_ppo_lc0','stage':stage,'metrics':metrics,**extra}
        row = json.loads(json.dumps(row,default=lambda x:x.item()))
        with (out/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        atomic_json(out/'latest.json',row)
        atomic_json(out/'health.json',{'updated':utc(),**row})
        print(json.dumps(row,ensure_ascii=False),flush=True)

    def inspect(batch, stage):
        # Both validation and training use untruncated temperature=1 sampling, so
        # native actor/ref logprobs and behavior probabilities share a distribution.
        if trainer.use_reference_policy:
            batch = batch.union(trainer._compute_ref_log_prob(batch))
        quality = rollout_quality(batch,max_response_length=settings['max_response_tokens'])
        quality.update(step=int(trainer.global_steps),stage=stage)
        diag = batch_diagnostics(batch,think_end_id=trainer.tokenizer.convert_tokens_to_ids('</think>'),
            eos_ids=trainer.tokenizer.eos_token_id if isinstance(trainer.tokenizer.eos_token_id,list)
                    else [trainer.tokenizer.eos_token_id])
        diag.update(step=int(trainer.global_steps),stage=stage)
        sync = sync_check(trainer,batch,samples=settings['preupdate_consistency_samples'],output=out) \
            if settings['preupdate_consistency_samples'] else None
        atomic_json(out/'quality-latest.json',quality)
        with (out/'quality.jsonl').open('a') as stream:stream.write(json.dumps(quality)+'\n')
        atomic_json(out/'update-diagnostics-latest.json',diag)
        atomic_json(out/'update-diagnostics'/f'{stage}-{trainer.global_steps:08d}.json',diag)
        enforce_update_safety(quality,diag,sync,settings,out)
        return batch,quality,diag,sync

    def validate():
        episodes,tensors,seconds = complete_games(trainer,context,count=settings['validation_games'],
            step=int(trainer.global_steps),validation=True)
        trainer.checkpoint_manager.sleep_replicas()
        batch = game_batch(episodes,tensors,settings,output=out)
        _,quality,diag,sync = inspect(batch,'validation')
        base = read_json(baseline_path) if baseline_path.exists() else None
        identity = validation_identity(settings,context['identity'])
        if base and base['validation_identity'] != identity:raise ValueError('Validation baseline identity changed')
        minimum = minimum_valid_rate(settings,base)
        summaries = [read_json(e.output/'summary.json') for e in episodes]
        row = {'step':int(trainer.global_steps),'stage':'after_update' if base else 'baseline',
            'validation_identity':identity,'valid_rate':quality['valid_rate'],
            'passed':quality['valid_rate']>=minimum,'minimum_valid_rate':minimum,
            'score':sum((e.result+1)/2 for e in episodes)/len(episodes),
            'wins':sum(e.result==1 for e in episodes),'losses':sum(e.result==-1 for e in episodes),
            'draws':sum(e.result==0 for e in episodes),'protocol_forfeits':sum(e.reason=='protocol_forfeit' for e in episodes),
            'random_moves':sum(s.get('random_moves',0) for s in summaries),
            'random_rate':sum(s.get('random_moves',0) for s in summaries)/sum(s['decisions'] for s in summaries),
            'games':summaries,'reference_kl_move_mean':diag.get('reference_kl_move_mean'),
            'sync':sync,'seconds':seconds,'utc':utc()}
        atomic_json(out/'validation'/f'step-{trainer.global_steps:08d}.json',row)
        if base is None:atomic_json(baseline_path,row)
        atomic_json(out/'validation-latest.json',row)
        persist_metrics({'validation/valid_rate':row['valid_rate'],'validation/score':row['score'],
            'validation/reference_kl':row['reference_kl_move_mean']},'validation',
            validation=row,quality=quality,diagnostics=diag)
        if not row['passed']:safety_stop(out,'fixed_game_validation_gate',row)
        return row

    if baseline_path.exists():
        baseline = read_json(baseline_path)
        if (not baseline.get('passed') or baseline.get('stage')!='baseline'
                or baseline['validation_identity'] != validation_identity(settings,context['identity'])):
            raise ValueError('Preseeded/resumed baseline does not match this experiment')
        minimum_valid_rate(settings,baseline)
    if not baseline_path.exists():
        if trainer.global_steps:raise ValueError('Resume lost original validation baseline')
        validate()
        trainer.checkpoint_manager.update_weights(trainer.global_steps)
    for step in range(int(trainer.global_steps)+1,settings['steps']+1):
        trainer.global_steps = step
        started = time.monotonic()
        episodes,tensors,generation_seconds = complete_games(trainer,context,
            count=settings['games_per_batch'],step=step)
        trainer.checkpoint_manager.sleep_replicas()
        atomic_json(out.parent/'progress.json',{'stage':'lc0_values','step':step,
            'message':'完整局已结束，标注学生状态价值并计算GAE','updated':utc()})
        batch = game_batch(episodes,tensors,settings,scorer=scorer,output=out)
        apply_bypass_mode(batch=batch,rollout_corr_config=trainer.config.algorithm.rollout_correction,
            policy_loss_config=trainer.config.actor_rollout_ref.actor.policy_loss)
        batch,quality,diag,sync = inspect(batch,'before_update')
        folder = out/'batches';folder.mkdir(exist_ok=True)
        torch.save({'step':step,'tensors':dict(batch.batch),'non_tensor':batch.non_tensor_batch,
            'meta':batch.meta_info},folder/f'step-{step:08d}.pt')
        atomic_json(out/'batch-latest.json',{'step':step,'path':str(folder/f'step-{step:08d}.pt'),'utc':utc()})
        atomic_json(out.parent/'progress.json',{'stage':'ppo_update','step':step,'updated':utc(),
            'message':f'完整局PPO更新 {step}/{settings["steps"]}'})
        update_started = time.monotonic()
        # Complete games contain many moves. A "2-step" smoke test should not
        # silently become dozens of optimizer steps on one old-policy cohort.
        # verl's native worker takes mini/global batch size through the metadata
        # built by _update_actor; microbatch=1 still bounds activation memory.
        old_minibatch = trainer.config.actor_rollout_ref.actor.ppo_mini_batch_size
        if settings['whole_batch_update']:
            trainer.config.actor_rollout_ref.actor.ppo_mini_batch_size = len(batch)
        try:
            actor_output = trainer._update_actor(batch)
        finally:
            trainer.config.actor_rollout_ref.actor.ppo_mini_batch_size = old_minibatch
        update_seconds = time.monotonic()-update_started
        metrics = reduce_metrics(actor_output.meta_info['metrics'])
        metrics.update({'training/global_step':step,'timing_s/gen':generation_seconds,
            'timing_s/update':update_seconds,'game/count':len(episodes),
            'game/wins':sum(e.result==1 for e in episodes),'game/draws':sum(e.result==0 for e in episodes),
            'game/losses':sum(e.result==-1 for e in episodes),
            'game/protocol_forfeits':sum(e.reason=='protocol_forfeit' for e in episodes),
            'game/random_moves':sum(t.get('action_source')=='random' for e in episodes for t in e.transitions),
            'game/random_rate':sum(t.get('action_source')=='random' for e in episodes for t in e.transitions)
                /batch.meta_info['real_moves'],
            'game/real_requests':batch.meta_info['real_requests'],'game/real_moves':batch.meta_info['real_moves'],
            'research/valid_rate':quality['valid_rate'],'research/reference_kl_move_mean':diag.get('reference_kl_move_mean'),
            'research/generation_tokens_per_s':quality['response_tokens_total']/generation_seconds})
        if any(not math.isfinite(float(v)) for v in metrics.values() if isinstance(v,(int,float))):
            safety_stop(out,'nonfinite_update_metrics',{'step':step,'metrics':metrics})
        persist_metrics(metrics,'update_completed',quality=quality,diagnostics=diag)
        trainer.checkpoint_manager.update_weights(step)
        final = step==settings['steps']
        if step%settings['validation_every']==0 or final:
            validate()
        else:
            trainer.checkpoint_manager.sleep_replicas()
        if step%settings['checkpoint_every']==0 or final:
            trainer._save_checkpoint()
            atomic_json(out/'checkpoints'/f'global_step_{step}'/'games.json',{
                'step':step,'identity':context['identity'],'episodes':[e.id for e in episodes],
                'validation_baseline':read_json(baseline_path),'next_seed':settings['seed']+(step+1)*100000})
        trainer.max_steps_duration = max(trainer.max_steps_duration,time.monotonic()-started)
        if not final:trainer.checkpoint_manager.update_weights(step)
    if hasattr(trainer,'_shutdown_dump_executor'):trainer._shutdown_dump_executor()
