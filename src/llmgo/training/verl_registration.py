"""Registration executes inside the Ray TaskRunner, not just the launcher process."""
from pathlib import Path
import json
import os

from llmgo.rsi.contracts import atomic_json, utc
from llmgo.training.quality import (RolloutQualityError, enforce_quality_gate,
    generation_throughput, rollout_quality, sync_check)
from llmgo.training.verl_ppo import bootstrap_advantage, centered_advantage
from llmgo.training.vllm_compat import install_rope_compat
from llmgo.training.verl_memory import install_wake_cache_flush
from llmgo.training.advantages import MODE as GUARDED_MODE, apply_protocol_guard

install_rope_compat()
# Round-9 default: release the trainer's cached blocks before the rollout engine re-maps its
# weights (j000023 OOM). Set LLMGO_WAKE_CACHE_FLUSH=0 to run the pre-fix behaviour.
if os.environ.get("LLMGO_WAKE_CACHE_FLUSH", "1") == "1":
    install_wake_cache_flush()
from verl.trainer.ppo.core_algos import register_adv_est
from verl.trainer.main_ppo import TaskRunner

register_adv_est("lc0_td0")(bootstrap_advantage)
register_adv_est("lc0_td0_centered")(centered_advantage)


class ResearchTaskRunner(TaskRunner):
    def run(self, config):
        from verl.utils.tracking import Tracking
        output = Path(os.environ["LLMGO_VERL_OUTPUT"])
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        import torch
        max_response_length = int(config.data.max_response_length)
        gate = float(config.research.quality_gate_min_valid)
        consistency_samples = int(config.research.preupdate_consistency_samples)
        sync_steps = int(config.research.sync_check_steps)
        original = Tracking.log
        original_update = RayPPOTrainer._update_actor
        state = {"quality":None,"sync":None,"protocol_guard":None}
        def update(trainer, batch):
            if str(config.research.advantage_estimator) == GUARDED_MODE:
                if config.algorithm.use_kl_in_reward:
                    raise ValueError("Guarded LC0 TD mode requires separate reference KL loss")
                state["protocol_guard"] = apply_protocol_guard(batch, output, int(trainer.global_steps),
                    penalty=float(config.research.protocol_failure_penalty))
            # Capture the exact on-policy tensors BEFORE optimizer updates.
            folder = output/"batches"
            folder.mkdir(exist_ok=True)
            target = folder/f"step-{trainer.global_steps:08d}.pt"
            temporary = target.with_suffix(".tmp")
            tensors = {key:value.detach().cpu() for key,value in batch.batch.items()}
            torch.save({"step":trainer.global_steps,"tensors":tensors,
                        "non_tensor":batch.non_tensor_batch},temporary)
            temporary.replace(target)
            atomic_json(output/"batch-latest.json", {"step":trainer.global_steps,
                "path":str(target),"tensors":list(tensors),"utc":utc()})
            quality = rollout_quality(batch,max_response_length=max_response_length)
            quality.update(step=int(trainer.global_steps),utc=utc())
            with (output/"quality.jsonl").open("a") as stream:
                stream.write(json.dumps(quality)+"\n")
            atomic_json(output/"quality-latest.json",quality)
            state["quality"] = quality
            if state["protocol_guard"] is not None and state["protocol_guard"]["valid"] == 0:
                raise RolloutQualityError("All responses failed protocol; saved batch and refused optimizer update")
            # 0 means every step. Comparing the trainer's own recomputation with the
            # log-probs vLLM returned for the *same* batch is the only available check
            # that the rollout engine is serving the weights the actor currently holds
            # (bypass_mode assigns old_log_probs from the rollout, so the saved batches
            # cannot show this).
            if consistency_samples and (sync_steps == 0 or trainer.global_steps <= sync_steps):
                state["sync"] = sync_check(trainer,batch,samples=consistency_samples,output=output)
            enforce_quality_gate(quality,gate,output=output)
            return original_update(trainer,batch)
        def log(instance, data, step, *args, **kwargs):
            row = {"step":step,"utc":utc(),"backend":"verl_ppo","metrics":data}
            # verl metric values may be NumPy scalars.
            row = json.loads(json.dumps(row,default=lambda x:x.item()))
            quality = state["quality"]
            guard = state["protocol_guard"]
            if guard is not None and guard.get("step") == step:
                row["metrics"].update({f"protocol_guard/{key}":value for key,value in guard.items()
                                      if key not in {"mode", "step"}})
                row["protocol_guard"] = guard
            if quality is not None and quality.get("step") == step:
                row["metrics"].update({f"research/{key}":value for key,value in quality.items()
                    if key not in ("utc","failure_counts")})
                row["metrics"]["research/failure_counts"] = quality["failure_counts"]
                row["metrics"]["research/generation_tokens_per_s"] = generation_throughput(row["metrics"])
            sync = state["sync"]
            state["sync"] = None
            if sync is not None and sync.get("step") == step:
                row["metrics"]["research/sync_mean_abs_logprob_delta"] = sync.get("mean_abs_logprob_delta")
                row["metrics"]["research/sync_fraction_ratio_outside"] = sync.get("fraction_ratio_outside_0.8_1.2")
                row["metrics"]["research/sync_error"] = sync.get("error")
            # Console and persisted metrics expose the same post-guard diagnostics.
            result = original(instance,row["metrics"],step,*args,**kwargs)
            with (output/"metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row)+"\n")
            atomic_json(output/"latest.json",row)
            atomic_json(output.parent/"progress.json", {"stage":"ppo_update", "updated":utc(),
                "message":f"verl PPO训练步骤 {step}/{config.trainer.total_training_steps}",
                "step":step,"total_steps":config.trainer.total_training_steps,
                "actor_gpus":config.trainer.n_gpus_per_node,
                "metrics_path":str(output/"metrics.jsonl")})
            return result
        Tracking.log = log
        RayPPOTrainer._update_actor = update
        try:
            return super().run(config)
        finally:
            Tracking.log = original
            RayPPOTrainer._update_actor = original_update
