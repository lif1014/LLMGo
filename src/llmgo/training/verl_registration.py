"""Registration executes inside the Ray TaskRunner, not just the launcher process."""
from pathlib import Path
import json
import os

from llmgo.rsi.contracts import atomic_json, utc
from llmgo.training.verl_ppo import bootstrap_advantage
from llmgo.training.vllm_compat import install_rope_compat

install_rope_compat()
from verl.trainer.ppo.core_algos import register_adv_est
from verl.trainer.main_ppo import TaskRunner

register_adv_est("lc0_td0")(bootstrap_advantage)


class ResearchTaskRunner(TaskRunner):
    def run(self, config):
        from verl.utils.tracking import Tracking
        output = Path(os.environ["LLMGO_VERL_OUTPUT"])
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        import torch
        original = Tracking.log
        original_update = RayPPOTrainer._update_actor
        def update(trainer, batch):
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
            return original_update(trainer,batch)
        def log(instance, data, step, *args, **kwargs):
            result = original(instance,data,step,*args,**kwargs)
            row = {"step":step,"utc":utc(),"backend":"verl_ppo","metrics":data}
            # verl metric values may be NumPy scalars.
            row = json.loads(json.dumps(row,default=lambda x:x.item()))
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
