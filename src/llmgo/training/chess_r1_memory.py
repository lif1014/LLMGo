"""Disposable 16K backward/remap acceptance using native verl actor updates."""
import math
import json
import time
import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.metric import reduce_metrics


def run(trainer):
    from .chess_r1_verl import atomic, emit, progress, root
    trainer.global_steps = 0
    trainer.checkpoint_manager.update_weights(0)
    trainer.checkpoint_manager.sleep_replicas()
    # The CUDA comparison runs in the allocated GPU launcher, before Ray;
    # this coordinator deliberately has no GPU assigned.
    attention_proof = json.loads((root()/'attention-gpu-acceptance.json').read_text())
    assert attention_proof['status'] == 'passed'
    gpus = int(trainer.config.trainer.n_gpus_per_node)
    groups = max(1, math.ceil(gpus*2/8))
    requests = groups*8
    batch = DataProto.from_single_dict(next(iter(trainer.train_dataloader)))[:groups].repeat(8, interleave=True)
    # Modern verl datasets carry raw_prompt and a dummy tensor; tokenization is
    # normally done by AgentLoopWorker, rather than the dataloader.
    chats = batch.non_tensor_batch['raw_prompt'].tolist()
    sequences = trainer.tokenizer.apply_chat_template(chats, tokenize=True, add_generation_prompt=True,
                                                      enable_thinking=True)
    if hasattr(sequences, 'keys'):
        sequences = sequences['input_ids']
    prompts = torch.full((requests,1024), trainer.tokenizer.pad_token_id, dtype=torch.long)
    prompt_mask = torch.zeros_like(prompts)
    for i, ids in enumerate(sequences):
        if len(ids)>1024:
            raise ValueError('Memory fixture prompt exceeds the production cap')
        prompts[i,-len(ids):] = torch.tensor(ids)
        prompt_mask[i,-len(ids):] = 1
    words = trainer.tokenizer.encode(' I compare the legal candidate moves and calculate the likely reply.',
                                     add_special_tokens=False)
    response = torch.tensor((words*(16384//len(words)+1))[:16384], dtype=prompts.dtype)
    responses = response.unsqueeze(0).repeat(requests, 1)
    attention = torch.cat((prompt_mask, torch.ones_like(responses)), dim=-1)
    positions = attention.long().cumsum(-1)-1
    positions.masked_fill_(attention==0, 0)
    # HF expands these 2D text positions to four identical MRoPE axes itself.
    # Keep the cross-process representation free of VLM jagged dimensions.
    if 'dummy_tensor' in batch.batch:
        batch.batch.pop('dummy_tensor')
    batch.batch['prompts'] = prompts
    batch.batch['responses'] = responses
    batch.batch['input_ids'] = torch.cat((prompts, responses), dim=-1)
    batch.batch['attention_mask'] = attention
    batch.batch['position_ids'] = positions
    batch.batch['response_mask'] = torch.ones_like(responses)
    batch.non_tensor_batch['uid'] = np.array([f'memory-{i//8}' for i in range(requests)], dtype=object)
    batch.meta_info.update(temperature=1., global_token_num=attention.sum(-1).tolist())
    scores = torch.zeros(requests, 16384)
    scores[:, -1] = torch.arange(8).repeat(groups)/7
    advantages, returns = compute_grpo_outcome_advantage(
        scores, batch.batch['response_mask'], batch.non_tensor_batch['uid'], norm_adv_by_std_in_grpo=True)
    batch.batch['advantages'], batch.batch['returns'] = advantages, returns
    batch.batch['token_level_scores'], batch.batch['token_level_rewards'] = scores, scores
    trainer.config.actor_rollout_ref.actor.ppo_mini_batch_size = groups
    progress('long_context_reference_forward', response_tokens=16384,
             forward_micro_batch_size=int(trainer.config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu))
    ref = RayPPOTrainer._compute_ref_log_prob(trainer, batch)
    batch = batch.union(ref)
    progress('long_context_actor_forward', response_tokens=16384,
             forward_micro_batch_size=int(trainer.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu))
    old, _ = RayPPOTrainer._compute_old_log_prob(trainer, batch)
    entropy = old.batch.pop('entropys')
    assert torch.isfinite(entropy).all()
    batch = batch.union(old)
    evidence = []
    for step in range(1, 3):
        trainer.global_steps = step
        progress('long_context_backward', step=step, response_tokens=16384)
        started = time.monotonic()
        result = RayPPOTrainer._update_actor(trainer, batch)
        metrics = reduce_metrics(result.meta_info['metrics'])
        assert all(math.isfinite(float(x)) for x in metrics.values() if isinstance(x, (int, float)))
        if metrics.get('actor/grad_norm', 0) <= 0:
            raise RuntimeError('Long-context acceptance has no finite nonzero gradient')
        trainer.checkpoint_manager.update_weights(step)
        trainer.checkpoint_manager.sleep_replicas()
        row = {'step': step, 'seconds': time.monotonic()-started, 'metrics': metrics}
        evidence.append(row)
        atomic(root()/'long-context-updates.json', evidence)
        emit(step, {**metrics, 'acceptance/response_tokens': 16384,
                    'acceptance/synthetic_fixture': 1, 'entropy/token_mean_nats': float(entropy.mean())})
    result = {'status': 'passed', 'response_tokens': 16384, 'requests': requests, 'actor_gpus': gpus,
              'microbatches_per_rank': requests//gpus, 'native_verl_actor_update': True,
              'forward_micro_batch_size': int(trainer.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu),
              'backward_micro_batch_size': int(trainer.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu),
              'native_grpo_advantages': True, 'entropy_coefficient': .001,
              'entropy_checkpointing': True, 'weight_remaps': 2,
              'synthetic_fixture': True, 'exported_model': False, 'updates': evidence}
    result['attention_parity_and_backward'] = attention_proof
    atomic(root()/'long-context-acceptance.json', result)
    trainer._shutdown_dump_executor()
    return result
