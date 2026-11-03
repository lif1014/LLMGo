"""Independent native-verl GRPO task; no complete-game PPO hooks are imported.

The native fit(), GRPO estimator, actor update and checkpoint manager stay in
verl. This subclass adds bounded chess evidence and original group length retries.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np
import torch
from verl.trainer.main_ppo import TaskRunner
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from .vllm_compat import install_rope_compat
from .verl_memory import install_wake_cache_flush
from .chess_r1_attention import install_3d_position_compat

install_rope_compat()
install_wake_cache_flush()
install_3d_position_compat()


class SegmentComplete(Exception):
    """A committed optimizer boundary, not a failed experiment."""


def root():
    return Path(os.environ['CHESS_R1_OUTPUT'])


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def progress(stage, **extra):
    path = root()/'progress.json'
    if 'step' not in extra and path.exists():
        previous = json.loads(path.read_text())
        if 'step' in previous:
            extra['step'] = previous['step']
        if stage == 'failed':
            extra['last_stage'] = previous['stage']
    atomic(path, {'stage': stage, 'time': time.time(), **extra})


def emit(step, metrics, tables=None):
    scalars = {k: float(v) for k, v in metrics.items()
               if isinstance(v, (float, int, np.number)) and math.isfinite(float(v))}
    scalars.update({'resume/attempt': int(os.environ.get('CHESS_R1_RESUME_ATTEMPT', 0)),
                    'resume/checkpoint_step': int(os.environ.get('CHESS_R1_RESUME_STEP', 0))})
    row = {'step': int(step), 'time': time.time(), 'metrics': scalars}
    with (root()/'metrics.jsonl').open('a') as f:
        f.write(json.dumps(row, allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())
    atomic(root()/'latest.json', row)
    packet = {'run_id': os.environ['WANDB_RUN_ID'], 'step': int(step),
              'scalars': {'training/global_step': int(step), **scalars}, 'tables': tables or {}}
    event = hashlib.sha256(json.dumps(packet, sort_keys=True).encode()).hexdigest()
    atomic(root()/'wandb/relay'/(event+'.json'), {**packet, 'event': event})


CURRENT = None


class ReplacementDataLoader:
    """Share one native sampler cursor between fit() and replacement batches.

    The fixed-step run may cross a source epoch while filling a retained batch.
    Native dataloader, sampler and RNG states are preserved in state_dict().
    """
    def __init__(self, native):
        self.native, self.cursor, self.source_epoch = native, None, 0

    def __len__(self):
        return len(self.native)

    def __iter__(self):
        if self.cursor is None:
            self.cursor = iter(self.native)
        return self

    def __next__(self):
        if self.cursor is None:
            self.cursor = iter(self.native)
        try:
            return next(self.cursor)
        except StopIteration:
            self.source_epoch += 1
            self.cursor = iter(self.native)
            return next(self.cursor)

    def state_dict(self):
        return {'native': self.native.state_dict(), 'chess_r1_source_epoch': self.source_epoch}

    def load_state_dict(self, state):
        if 'chess_r1_source_epoch' in state:
            self.source_epoch = state['chess_r1_source_epoch']
            state = state['native']
        self.native.load_state_dict(state)
        self.cursor = None

    def __getattr__(self, name):
        return getattr(self.native, name)


class ChessR1Trainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        global CURRENT
        super().__init__(*args, **kwargs)
        CURRENT = self
        self.chess_tables, self.chess_metrics = {}, {}
        self.train_dataloader = ReplacementDataLoader(self.train_dataloader)

    def _get_gen_batch(self, batch):
        if getattr(self, '_validation_active', False):
            progress('validating', step=self.global_steps, positions_completed=self._validation_seen,
                     positions_total=len(self.val_dataset))
            self._validation_seen += len(batch)
        else:
            progress('sampling', step=self.global_steps)
            self._training_batch = batch
        return super()._get_gen_batch(batch)

    def _validate(self, merged=False):
        self._validation_active, self._validation_seen = True, 0
        try:
            return super()._validate(merged=merged)
        finally:
            self._validation_active = False

    def _compute_ref_log_prob(self, batch):
        from .chess_r1_attention import use_native_text_positions
        use_native_text_positions(batch)
        return super()._compute_ref_log_prob(batch)

    def _native_actor_log_prob(self, batch):
        """Align inference batches, then remove padding before any statistics.

        Whole-group cap rejection and refill oversupply produce arbitrary
        multiples of rollout.n, which need not fit the actor DP/micro-batch
        divisor. Padding is used only for inference, never actor updates.
        """
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        if not len(batch):
            raise ValueError('Actor inference requires a nonempty batch')
        dp_size = int(self._get_dp_size(self.actor_rollout_wg, 'actor'))
        micro_batch = int(self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu)
        padded, padding = pad_dataproto_to_divisor(batch, dp_size*micro_batch)
        if padding:
            padded.meta_info = dict(batch.meta_info)
            padded.meta_info['global_token_num'] = padded.batch['attention_mask'].sum(-1).tolist()
        result, mfu = super()._compute_old_log_prob(padded)
        return unpad_dataproto(result, padding), mfu

    def _compute_old_log_prob(self, batch):
        from .chess_r1_attention import use_native_text_positions
        use_native_text_positions(batch)
        result, mfu = self._native_actor_log_prob(batch)
        entropy = result.batch['entropys']
        mask = batch.batch['response_mask'].bool()
        if not torch.isfinite(entropy[mask]).all():
            raise ValueError('Missing or nonfinite full-vocabulary entropy')
        lengths = mask.sum(-1)
        per_request = entropy.masked_fill(~mask, 0).sum(-1)/lengths.clamp_min(1)
        total_entropy, total_tokens = float(entropy[mask].sum()), int(mask.sum())
        all_requests = per_request.tolist()
        rejected_evidence = []
        for attempt, rejected in enumerate(getattr(self, '_rejected_batches', [])):
            use_native_text_positions(rejected)
            retry, _ = self._native_actor_log_prob(rejected)
            retry_mask = rejected.batch['response_mask'].bool()
            retry_entropy = retry.batch['entropys']
            if not torch.isfinite(retry_entropy[retry_mask]).all():
                raise ValueError('Nonfinite rejected-request entropy')
            retry_lengths = retry_mask.sum(-1)
            retry_per_request = retry_entropy.masked_fill(~retry_mask, 0).sum(-1)/retry_lengths.clamp_min(1)
            total_entropy += float(retry_entropy[retry_mask].sum())
            total_tokens += int(retry_mask.sum())
            all_requests.extend(retry_per_request.tolist())
            retry_file = root()/'entropy'/f'step-{self.global_steps:08d}-retry-{attempt:02d}.pt'
            retry_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'policy_entropy': retry_entropy.cpu(), 'response_mask': retry_mask.cpu()}, retry_file)
            for i in range(len(rejected)):
                extra = rejected.non_tensor_batch['extra_info'][i]
                rejected_evidence.append({'uid': str(rejected.non_tensor_batch['uid'][i]),
                    'puzzle_id': str(extra['puzzle_id']), 'response_tokens': int(retry_lengths[i]),
                    'entropy_nats': float(retry_per_request[i]), 'retained_for_update': 0})
        self._rejected_batches = []
        self._rejected_requests = rejected_evidence
        self.chess_metrics.update({'entropy/token_mean_nats': total_entropy/max(total_tokens,1),
                                  'entropy/request_mean_nats': float(np.mean(all_requests)),
                                  'entropy/retained_token_mean_nats': float(entropy[mask].mean()),
                                  'entropy/retained_request_mean_nats': float(per_request.mean())})
        self._request_entropies = per_request.tolist()
        # Local tensor evidence is kept on shared disk, outside metric packets.
        folder = root()/'entropy'
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({'policy_entropy': entropy.detach().cpu(), 'response_mask': mask.cpu()},
                   folder/f'step-{self.global_steps:08d}.pt')
        atomic(folder/f'step-{self.global_steps:08d}.json',
               {'step': self.global_steps, 'response_tokens': lengths.tolist(),
                'request_entropy_nats': self._request_entropies, 'all_response_tokens': total_tokens,
                'denominator': 'all generated requests including rejected retries; padding excluded',
                'rejected_requests': rejected_evidence, 'rejected_retry_entropy_available': True})
        return result, mfu

    def _update_actor(self, batch):
        progress('updating', step=self.global_steps)
        dp_size = self._get_dp_size(self.actor_rollout_wg, 'actor')
        forward_micro_batch = int(self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu)
        probe_divisor = dp_size*forward_micro_batch
        selected = min(len(batch), math.ceil(8/probe_divisor)*probe_divisor)
        if selected == 0 or selected % probe_divisor:
            raise ValueError('Actor verification batch must fit the DP and forward micro-batch sizes')
        mask = batch.batch['response_mask'].bool()
        lengths = mask.sum(-1)
        rewards = batch.batch['token_level_scores'].sum(-1)
        advantages = (batch.batch['advantages'] * mask).sum(-1)/lengths.clamp_min(1)
        rows, groups, group_indices = [], {}, {}
        for i in range(len(batch)):
            uid = str(batch.non_tensor_batch['uid'][i])
            extra = batch.non_tensor_batch['extra_info'][i]
            row = {'uid': uid, 'puzzle_id': str(extra['puzzle_id']), 'rating': int(extra['rating']),
                   'reward': float(rewards[i]), 'advantage': float(advantages[i]),
                   'response_tokens': int(lengths[i]), 'entropy_nats': float(self._request_entropies[i])}
            for key in ['q', 'format', 'english', 'legal', 'solution_correct', 'oracle_top1', 'q_gap', 'truncated']:
                if key in batch.non_tensor_batch:
                    row[key] = float(batch.non_tensor_batch[key][i])
            rows.append(row)
            groups.setdefault(uid, []).append(row)
            group_indices.setdefault(uid, []).append(i)
        group_rows = []
        for uid, items in groups.items():
            if len(items) != int(self.config.actor_rollout_ref.rollout.n):
                raise ValueError('GRPO group lost repeated responses')
            indices = group_indices[uid]
            # Preserve native torch dtype, device and reduction semantics.
            # Recomputing in NumPy float64 falsely rejects constant-reward
            # groups when float32 mean/std rounding is amplified by epsilon.
            scores = rewards[indices]
            expected = (scores-scores.mean())/(scores.std()+1e-6)
            actual_tokens = batch.batch['advantages'][indices]
            expected_tokens = expected.unsqueeze(-1)*batch.batch['response_mask'][indices]
            if not torch.allclose(expected_tokens, actual_tokens, atol=2e-4):
                atomic(root()/'diagnostics'/f'advantage-mismatch-step-{self.global_steps:08d}.json',
                       {'step': self.global_steps, 'uid': uid, 'dtype': str(scores.dtype),
                        'rewards': scores.tolist(), 'expected': expected.tolist(),
                        'observed': advantages[indices].tolist(),
                        'max_token_error': float((actual_tokens-expected_tokens).abs().max())})
                raise ValueError('Native GRPO advantages disagree with per-uid reward normalization')
            group_rows.append({'uid': uid, 'puzzle_id': items[0]['puzzle_id'], 'responses': len(items),
                               'reward_mean': float(scores.mean()), 'reward_std': float(scores.std()),
                               'constant_reward': int(bool((scores == scores[0]).all())),
                               'zero_advantage': int(bool((expected == 0).all()))})
        self.chess_tables = {'requests': rows, 'groups': group_rows,
                             'rejected_requests': getattr(self, '_rejected_requests', [])}
        self.chess_metrics.update({'quality/reward_mean': float(rewards.mean()),
                                  'quality/response_tokens_mean': float(lengths.float().mean()),
                                  'quality/response_tokens_max': int(lengths.max()),
                                  'quality/response_tokens_total': int(lengths.sum()),
                                  'groups/zero_advantage_fraction': float(np.mean([x['zero_advantage'] for x in group_rows])),
                                  'groups/constant_reward_fraction': float(np.mean([x['constant_reward'] for x in group_rows]))})
        all_lengths = [x['response_tokens'] for x in rows+getattr(self, '_rejected_requests', [])]
        self.chess_metrics.update({'quality/retained_response_tokens_mean': float(lengths.float().mean()),
                                  'quality/response_tokens_mean': float(np.mean(all_lengths)),
                                  'quality/response_tokens_max': max(all_lengths),
                                  'quality/response_tokens_total': sum(all_lengths),
                                  'quality/request_count': len(all_lengths)})
        for key in ['q', 'format', 'english', 'legal', 'solution_correct', 'oracle_top1', 'truncated']:
            if key in rows[0]:
                self.chess_metrics['quality/'+key] = float(np.mean([x[key] for x in rows]))
        if 'ref_log_prob' in batch.batch:
            delta = batch.batch['ref_log_prob']-batch.batch['old_log_probs']
            self.chess_metrics['reference/kl_before_update'] = float((torch.exp(delta)-delta-1)[mask].mean())
        if 'rollout_log_probs' in batch.batch:
            sync = float((batch.batch['old_log_probs']-batch.batch['rollout_log_probs']).abs()[mask].mean())
            self.chess_metrics['sync/mean_abs_logprob_delta'] = sync
            if not math.isfinite(sync) or sync > .1:
                raise RuntimeError('Actor/rollout logprob mismatch exceeds0.1')
        # Preserve exact real response tensors locally, never in W&B packets.
        probe = batch[:selected]
        before = probe.batch['old_log_probs'].clone()
        result = super()._update_actor(batch)
        for value in result.meta_info.get('metrics', {}).values():
            vals = value if isinstance(value, list) else [value]
            if any(isinstance(x, (float, int, np.number)) and not math.isfinite(x) for x in vals):
                raise RuntimeError('Nonfinite optimizer metrics')
        atomic(root()/'updates'/f'step-{self.global_steps:08d}-optimizer-returned.json',
               {'step': self.global_steps, 'optimizer_completed': True, 'verification_completed': False,
                'probe_responses': selected, 'actor_dp_size': dp_size, 'forward_micro_batch_size': forward_micro_batch,
                'time': time.time()})
        progress('verifying_update', step=self.global_steps, probe_responses=selected,
                 actor_dp_size=dp_size, forward_micro_batch_size=forward_micro_batch)
        after, _ = self._native_actor_log_prob(probe)
        change = float((after.batch['old_log_probs']-before).abs()[probe.batch['response_mask'].bool()].mean())
        self.chess_metrics['update/mean_abs_logprob_change'] = change
        per_move = {}
        for item in rows+getattr(self, '_rejected_requests', []):
            per_move.setdefault(item['uid'], []).append(item['entropy_nats'])
        self.chess_metrics['entropy/move_mean_nats'] = float(np.mean([np.mean(x) for x in per_move.values()]))
        self.chess_metrics['update/optimizer_steps_per_rollout'] = len(batch)/(int(
            self.config.actor_rollout_ref.actor.ppo_mini_batch_size)*int(self.config.actor_rollout_ref.rollout.n))
        if not math.isfinite(change) or change <= 1e-9:
            raise RuntimeError('Optimizer did not measurably change actor probabilities')
        if os.environ.get('CHESS_R1_VERIFY_CHECKPOINT_RESTORE') == '1':
            self._checkpoint_probe = probe
            self._checkpoint_probe_expected = after.batch['old_log_probs'].clone()
        atomic(root()/'updates'/f'step-{self.global_steps:08d}.json',
               {'step': self.global_steps, 'rows': len(batch), 'group_count': len(groups),
                'optimizer_completed': True, 'logprob_change': change, 'metrics': self.chess_metrics})
        atomic(root()/'tables'/f'step-{self.global_steps:08d}.json', self.chess_tables)
        return result

    def _save_checkpoint(self):
        progress('saving_checkpoint', step=self.global_steps)
        super()._save_checkpoint()
        folder = Path(self.config.trainer.default_local_dir)/f'global_step_{self.global_steps}'
        if not (folder/'actor').is_dir() or not (folder/'data.pt').is_file():
            raise RuntimeError('Native checkpoint missing actor or dataloader state')
        shards = list((folder/'actor').glob('*.pt'))
        for kind in ['model_world_size_', 'optim_world_size_', 'extra_state_world_size_']:
            if len([p for p in shards if p.name.startswith(kind)]) != int(self.config.trainer.n_gpus_per_node):
                raise RuntimeError('Native checkpoint is missing '+kind+' shards')
        if os.environ.get('CHESS_R1_VERIFY_CHECKPOINT_RESTORE') == '1':
            progress('restoring_native_checkpoint', step=self.global_steps)
            previous_mode = self.config.trainer.resume_mode
            previous_path = self.config.trainer.resume_from_path
            try:
                self.config.trainer.resume_mode = 'resume_path'
                self.config.trainer.resume_from_path = str(folder)
                # This calls the real distributed actor/optimizer/RNG loader
                # and the native trainer's dataloader restore path.
                super()._load_checkpoint()
                restored, _ = self._native_actor_log_prob(self._checkpoint_probe)
                mask = self._checkpoint_probe.batch['response_mask'].bool()
                difference = (restored.batch['old_log_probs']-
                              self._checkpoint_probe_expected).abs()[mask]
                if not torch.isfinite(difference).all() or float(difference.max()) > 1e-5:
                    raise RuntimeError('Restored actor probabilities differ from saved actor')
                atomic(root()/'checkpoint-gpu-acceptance.json', {
                    'status': 'passed', 'step': self.global_steps,
                    'distributed_resume_executed': True,
                    'native_actor_optimizer_rng_and_dataloader_loader': True,
                    'restored_response_tokens_compared': int(mask.sum()),
                    'restored_logprob_max_abs_delta': float(difference.max()),
                    'restored_logprob_mean_abs_delta': float(difference.mean()),
                    'actor_gpus': int(self.config.trainer.n_gpus_per_node)})
            finally:
                self.config.trainer.resume_mode = previous_mode
                self.config.trainer.resume_from_path = previous_path
        from .chess_r1_resume import checkpoint_manifest
        atomic(root()/'checkpoints'/f'receipt-{self.global_steps:08d}.json',
               {'step': self.global_steps, 'path': str(folder), 'native_verl': True,
                'file_sizes': checkpoint_manifest(folder, int(self.config.trainer.n_gpus_per_node)),
                'resume_contains_optimizer_rng_and_data': True})

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        metrics = super()._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)
        correct = reward_extra_infos_dict.get('solution_correct', [])
        if len(correct):
            metrics['chess_validation/step_accuracy'] = float(np.mean(correct))
            puzzles = {}
            for puzzle, value in zip(reward_extra_infos_dict['puzzle_id'], correct, strict=True):
                puzzles.setdefault(puzzle, []).append(value)
            metrics['chess_validation/puzzle_all_positions_correct'] = float(np.mean([
                all(items) for items in puzzles.values()]))
            if 'rating' in reward_extra_infos_dict:
                by_bin = {}
                ratings = dict(zip(reward_extra_infos_dict['puzzle_id'], reward_extra_infos_dict['rating'], strict=True))
                for puzzle, items in puzzles.items():
                    by_bin.setdefault(int(ratings[puzzle])//200*200, []).append(all(items))
                for lower, values in by_bin.items():
                    metrics[f'chess_validation/rating_{lower}_{lower+200}'] = float(np.mean(values))
        return metrics

    def fit(self):
        if os.environ.get('CHESS_R1_MEMORY_ONLY') == '1':
            from .chess_r1_memory import run
            return run(self)
        import verl.utils.tracking as tracking
        original = tracking.Tracking
        trainer = self
        class ObserveTracking(original):
            def log(self, data, step, *a, **kw):
                original.log(self, data, step, *a, **kw)
                emit(step, {**data, **trainer.chess_metrics}, trainer.chess_tables)
                progress('update_completed' if trainer.chess_tables else 'validation', step=step)
                target = int(os.environ.get('CHESS_R1_SEGMENT_TARGET', '0'))
                if target and step >= target and trainer.chess_tables:
                    if not (root()/'checkpoints'/f'receipt-{step:08d}.json').is_file():
                        trainer._save_checkpoint()
                    raise SegmentComplete()
        tracking.Tracking = ObserveTracking
        original_generate = self.async_rollout_manager.generate_sequences
        def complete_groups(data):
            # Validation n=1 preserves the native Lichess evaluation population.
            if data.meta_info.get('validate', False):
                return original_generate(data)
            from verl import DataProto
            self._rejected_batches = []
            primary = self._training_batch
            target_groups = len(primary)
            accepted = []
            rejected_lengths = []
            last_meta = {}
            inputs = data
            group_size = int(self.config.actor_rollout_ref.rollout.n)
            for attempt in range(10):
                generated = original_generate(inputs)
                last_meta = dict(generated.meta_info)
                lengths = generated.batch['response_mask'].sum(-1).tolist()
                # The native generation input can have batch=None. Start from
                # the generated tensors and add the input metadata explicitly.
                combined = DataProto(batch=generated.batch,
                                     non_tensor_batch=inputs.non_tensor_batch.copy(),
                                     meta_info=inputs.meta_info.copy()).union(generated)
                groups = {}
                for j, uid in enumerate(inputs.non_tensor_batch['uid']):
                    groups.setdefault(str(uid), []).append(j)
                discarded = []
                for indices in groups.values():
                    if len(indices) != group_size:
                        raise ValueError('Incomplete repeated prompt group before filtering')
                    if any(lengths[j] >= 16384 for j in indices) or len(accepted) >= target_groups:
                        discarded.extend(indices)
                        rejected_lengths.extend(lengths[j] for j in indices)
                    else:
                        group = combined[indices]
                        group.meta_info = {}
                        accepted.append(group)
                if discarded:
                    rejected = combined[discarded]
                    rejected.meta_info = {'temperature': 1.,
                        'global_token_num': rejected.batch['attention_mask'].sum(-1).tolist()}
                    self._rejected_batches.append(rejected)
                pending_groups = target_groups-len(accepted)
                attempt_metrics = {
                    'step': self.global_steps, 'attempt': attempt+1,
                    'generated_responses': len(lengths),
                    'capped_responses': sum(length >= 16384 for length in lengths),
                    'response_tokens_mean': float(np.mean(lengths)),
                    'response_tokens_max': max(lengths),
                    'retained_groups_so_far': len(accepted),
                    'target_groups': target_groups,
                    'pending_responses': pending_groups*group_size}
                atomic(root()/'sampling'/f'step-{self.global_steps:08d}-attempt-{attempt+1:02d}.json',
                       attempt_metrics)
                progress('resampling_overlong_groups' if pending_groups else 'sampling_completed',
                         attempts=attempt+1, **attempt_metrics)
                if not pending_groups:
                    result = DataProto.concat(accepted)
                    # Native fit repeats its original prompt metadata after this
                    # callback. Keep it aligned with the accepted fresh groups.
                    for key in list(primary.non_tensor_batch):
                        primary.non_tensor_batch[key] = result.non_tensor_batch[key][::group_size].copy()
                    result.meta_info = last_meta
                    self.chess_metrics.update({'sampling/generation_batches': attempt+1,
                                               'sampling/rejected_responses': len(rejected_lengths),
                                               'sampling/rejected_response_tokens': sum(rejected_lengths),
                                               'sampling/source_epoch': self.train_dataloader.source_epoch})
                    return result
                replacement = DataProto.from_single_dict(next(self.train_dataloader))
                replacement.meta_info.update(temperature=1., global_steps=self.global_steps)
                replacement.non_tensor_batch['uid'] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(replacement))], dtype=object)
                inputs = RayPPOTrainer._get_gen_batch(self, replacement).repeat(group_size, interleave=True)
            raise RuntimeError('10 generation attempts exceeded: overlong groups cannot enter training')
        self.async_rollout_manager.generate_sequences = complete_groups
        try:
            # Native fit computes GRPO advantages and calls native actor/checkpoint.
            return super().fit()
        except SegmentComplete:
            if hasattr(self.actor_rollout_wg, 'async_calls_finalize_fn_exec'):
                self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
            self._shutdown_dump_executor()
            return
        finally:
            tracking.Tracking = original
            self.async_rollout_manager.generate_sequences = original_generate


class ChessR1TaskRunner(TaskRunner):
    def add_actor_rollout_worker(self, config):
        import ray
        from verl.trainer.ppo.ray_trainer import Role
        from .chess_r1_worker import ChessR1Worker
        _, worker_group_cls = super().add_actor_rollout_worker(config)
        for role in (Role.ActorRollout, Role.ActorRolloutRef):
            if role in self.role_worker_mapping:
                self.role_worker_mapping[role] = ray.remote(ChessR1Worker)
        return ChessR1Worker, worker_group_cls

    def run(self, config):
        import verl.trainer.main_ppo as native
        original = native.RayPPOTrainer
        native.RayPPOTrainer = ChessR1Trainer
        try:
            return super().run(config)
        finally:
            native.RayPPOTrainer = original
