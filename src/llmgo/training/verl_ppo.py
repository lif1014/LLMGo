"""verl 0.8 PPO with a frozen external LC0 TD(0) critic; no learned value head."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from llmgo.rsi.contracts import atomic_json, digest, file_hash, model_identity, read_json, utc
from .rollout import prompt_for_record

VERL_VERSION = "0.8.0"


def training_config(config, allocated_gpus=8):
    if type(allocated_gpus) is not int or not 1 <= allocated_gpus <= 8:
        raise ValueError("verl PPO requires 1-8 allocated GPUs")
    actors = allocated_gpus-1 if allocated_gpus > 1 else 1
    defaults = {"backend":"verl_ppo", "batch_size":actors*8, "minibatch_size":actors*2,
        "steps":16, "ppo_epochs":4, "learning_rate":1e-5, "clip_ratio":.2,
        "kl_beta":.02, "max_prompt_tokens":8192, "max_response_tokens":2048,
        "checkpoint_every":4, "seed":4170, "rollout_memory_utilization":.5}
    if not isinstance(config, dict) or set(config)-set(defaults):
        raise ValueError("Unknown verl PPO configuration; no SFT/GRPO/legacy LoRA fields")
    cfg = {**defaults, **config}
    if cfg["backend"] != "verl_ppo":
        raise ValueError("Use backend=verl_ppo; legacy HF PPO checkpoints are not verl checkpoints")
    for key in ("batch_size","minibatch_size","steps","ppo_epochs","max_prompt_tokens","max_response_tokens","checkpoint_every","seed"):
        if type(cfg[key]) is not int or cfg[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid {key}")
    if cfg["batch_size"] % cfg["minibatch_size"] or cfg["minibatch_size"] % actors:
        raise ValueError(f"Batch must divide into minibatches, and minibatch_size must be divisible by {actors} actor GPUs")
    for key,low,high in (("learning_rate",0,1),("clip_ratio",0,1),("kl_beta",0,10),("rollout_memory_utilization",0,.95)):
        from llmgo.rsi.contracts import finite
        finite(cfg[key],key,low,high)
        if key != "kl_beta" and cfg[key] == 0:
            raise ValueError(f"{key} must be positive")
    if cfg["max_response_tokens"] > 32768 or cfg["max_prompt_tokens"]+cfg["max_response_tokens"] > 40960:
        raise ValueError("Training sequence exceeds the existing 40960-token context ceiling")
    return cfg


def bootstrap_advantage(token_level_rewards, response_mask, **kwargs):
    """Already-computed chess TD error, broadcast over this move's generated tokens.

    No group normalization and no token-wise discounting: one response is one action.
    Returns are unused because critic.enable=False; LC0 values remain in sample logs.
    """
    scores = (token_level_rewards*response_mask).sum(-1, keepdim=True)
    advantages = scores*response_mask
    return advantages, advantages.clone()


def dataset_rows(records, cards, batch_size):
    rows = []
    for index, record in enumerate(records):
        prompt = prompt_for_record(record, cards)
        rows.append({"data_source":"chess_lc0", "prompt":prompt["messages"],
            # Required by verl's dataset schema; deliberately contains no teacher answer.
            "reward_model":{"style":"rule", "ground_truth":""},
            "extra_info":{"index":index, "position_id":prompt["position_id"],
                          "record_json":json.dumps(record,sort_keys=True)}})
    if not rows:
        raise ValueError("Empty PPO curriculum")
    return (rows*math.ceil(batch_size/len(rows)))[:max(batch_size,len(rows))]


def hydra_overrides(model, dataset, output, config, allocated_gpus, teacher_context, resume=None):
    actors = allocated_gpus-1 if allocated_gpus > 1 else 1
    reward_module = str(Path(__file__).with_name("verl_reward.py"))
    # Strings use JSON quoting so paths containing !/spaces cannot become Hydra syntax.
    values = {
        "actor_rollout_ref.model.path":str(model),
        "actor_rollout_ref.model.use_remove_padding":False,
        "actor_rollout_ref.model.enable_gradient_checkpointing":True,
        "actor_rollout_ref.model.external_lib":"llmgo.training.verl_registration",
        "actor_rollout_ref.actor.strategy":"fsdp2",
        "actor_rollout_ref.actor.fsdp_config.strategy":"fsdp2",
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile":False,
        "actor_rollout_ref.actor.use_torch_compile":False,
        "actor_rollout_ref.actor.use_remove_padding":False,
        "actor_rollout_ref.actor.ppo_mini_batch_size":config["minibatch_size"],
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu":1,
        "actor_rollout_ref.actor.ppo_epochs":config["ppo_epochs"],
        "actor_rollout_ref.actor.optim.lr":config["learning_rate"],
        "actor_rollout_ref.actor.clip_ratio":config["clip_ratio"],
        "actor_rollout_ref.actor.clip_ratio_low":config["clip_ratio"],
        "actor_rollout_ref.actor.clip_ratio_high":config["clip_ratio"],
        "actor_rollout_ref.actor.use_kl_loss":config["kl_beta"] > 0,
        "actor_rollout_ref.actor.kl_loss_coef":config["kl_beta"],
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu":1,
        "actor_rollout_ref.ref.fsdp_config.use_torch_compile":False,
        "actor_rollout_ref.rollout.name":"vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size":1,
        "actor_rollout_ref.rollout.temperature":1.,
        "actor_rollout_ref.rollout.top_p":1.,
        "actor_rollout_ref.rollout.top_k":-1,
        "actor_rollout_ref.rollout.n":1,
        "actor_rollout_ref.rollout.calculate_log_probs":True,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu":1,
        "actor_rollout_ref.rollout.enforce_eager":True,
        "actor_rollout_ref.rollout.max_model_len":config["max_prompt_tokens"]+config["max_response_tokens"],
        "actor_rollout_ref.rollout.gpu_memory_utilization":config["rollout_memory_utilization"],
        "actor_rollout_ref.rollout.agent.num_workers":math.gcd(config["batch_size"],min(actors,4)),
        "critic.enable":False,
        "algorithm.adv_estimator":"lc0_td0",
        "algorithm.gamma":1., "algorithm.lam":0.,
        "algorithm.use_kl_in_reward":False,
        "algorithm.rollout_correction.bypass_mode":True,
        "algorithm.rollout_correction.loss_type":"ppo_clip",
        "data.train_files":[str(dataset)],"data.val_files":[str(dataset)],
        "data.train_batch_size":config["batch_size"],
        "data.max_prompt_length":config["max_prompt_tokens"],
        "data.max_response_length":config["max_response_tokens"],
        "data.filter_overlong_prompts":False,
        "data.truncation":"error",
        "data.filter_overlong_prompts_workers":1,
        "data.dataloader_num_workers":0,
        "data.seed":config["seed"],
        "reward.num_workers":1,
        "reward.reward_manager.source":"importlib",
        "reward.reward_manager.name":"ChessRewardManager",
        "reward.reward_manager.module.path":reward_module,
        "trainer.nnodes":1,"trainer.n_gpus_per_node":actors,
        "trainer.total_training_steps":config["steps"],
        "trainer.total_epochs":config["steps"]+1,
        "trainer.project_name":"rsi", "trainer.experiment_name":output.parent.name,
        "trainer.logger":["console"],"trainer.val_before_train":False,"trainer.test_freq":-1,
        "trainer.save_freq":config["checkpoint_every"],
        "trainer.default_local_dir":str(output/"checkpoints"),
        "trainer.rollout_data_dir":str(output/"rollouts"),
        "trainer.resume_mode":"resume_path" if resume else "disable",
        "trainer.resume_from_path":str(resume) if resume else None,
    }
    result = [f"{k}={json.dumps(v)}" for k,v in values.items()]
    result.append("+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=true")
    result.append("+data.apply_chat_template_kwargs.enable_thinking=true")
    result.append("+reward.lc0_context="+json.dumps(str(teacher_context)))
    return result


def complete_checkpoint(root, actor_gpus):
    """Never resume a half-written checkpoint or a different distributed world size."""
    candidates = []
    for path in Path(root).glob("global_step_*"):
        try:
            step = int(path.name.removeprefix("global_step_"))
        except ValueError:
            continue
        actor = path/"actor"
        if ((path/"data.pt").is_file() and (actor/"huggingface/config.json").is_file()
                and all((actor/f"{kind}_world_size_{actor_gpus}_rank_{i}.pt").is_file()
                        for kind in ("model","optim","extra_state") for i in range(actor_gpus))):
            candidates.append((step,path))
    return max(candidates, default=(0,None))[1]


def train(model, records, cards, site, reward_config, config, output, *, allocated_gpus=8, resume_from=None):
    if os.environ.get("LLMGO_RSI_WORKER") != "1":
        raise RuntimeError("Submit PPO through a GPU research job")
    import importlib.metadata
    if importlib.metadata.version("verl") != VERL_VERSION:
        raise RuntimeError(f"Expected verl=={VERL_VERSION}; run scripts/bootstrap_verl.py in the designated image")
    from .rewards import Lc0Reward, validate_reward
    reward_config = validate_reward(reward_config)
    config = training_config(config, allocated_gpus)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    actors = allocated_gpus-1 if allocated_gpus > 1 else 1
    if allocated_gpus == 1 and reward_config["backend"].startswith("cuda"):
        raise ValueError("One-GPU PPO needs a CPU LC0 backend (eigen/blas) or request at least two GPUs")
    if allocated_gpus > 1 and not reward_config["backend"].startswith("cuda"):
        raise ValueError("Multi-GPU default reserves one GPU for LC0; select a CUDA backend")
    identity = {"model":model_identity(model)["sha256"], "records":digest(records), "cards":digest(cards),
        "training":config,"reward":reward_config,"allocated_gpus":allocated_gpus,"verl":VERL_VERSION,
        "runtime":{name:importlib.metadata.version(name) for name in ("verl","transformers","torch","vllm","tokenizers","ray")},
        "code":{name:file_hash(Path(__file__).with_name(name)) for name in
            ("verl_ppo.py","verl_registration.py","verl_reward.py","vllm_compat.py","rewards.py","rollout.py")},
        "lc0_binary":file_hash(site["lc0_bin"])}
    dataset = output/"curriculum.jsonl"
    dataset.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in dataset_rows(records,cards,config["batch_size"])))
    resume = None
    if resume_from:
        old = Path(resume_from)/"training"
        if not (old/"identity.json").is_file() or read_json(old/"identity.json") != identity:
            raise ValueError("Resume identity changed; submit a new experiment rather than mixing optimizer state")
        resume = complete_checkpoint(old/"checkpoints", actors)
        if resume is None:
            raise ValueError("No complete verl checkpoint is available in the previous task")
    atomic_json(output/"identity.json",identity)
    import ray
    from hydra import compose, initialize_config_dir
    import verl
    import torch
    if torch.cuda.device_count() < allocated_gpus:
        raise RuntimeError("Fewer visible GPUs than the registered PPO resource request")
    # Fail before allocating distributed workers if the pinned HF/vLLM model
    # configuration cannot be parsed. The patch also runs in every verl model
    # worker through model.external_lib, including rollout server processes.
    from .vllm_compat import install_rope_compat
    install_rope_compat()
    from vllm.transformers_utils.config import get_config
    parsed_model = get_config(str(model), trust_remote_code=False)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(model), trust_remote_code=False)
    # Both training and export must retain the real Qwen processor. Do not rely
    # on verl's optional-processor helper silently swallowing dependency errors.
    from vllm.engine.arg_utils import AsyncEngineArgs
    if "language_model_only" not in AsyncEngineArgs.__dataclass_fields__:
        raise RuntimeError("This vLLM build does not support text-only Qwen rollout")
    atomic_json(output/"model-preflight.json", {"model_type":parsed_model.model_type,
        "processor":type(processor).__name__, "language_model_only":True})
    del processor
    from .verl_reward import Lc0Scorer
    namespace = "llmgo-"+uuid.uuid4().hex
    source = str(Path(__file__).resolve().parents[2])
    ray.init(address="local", num_gpus=allocated_gpus, namespace=namespace,
        _temp_dir=str(Path(site["user_root"])/"r"/namespace[-8:]), runtime_env={"env_vars":{
            "PYTHONPATH":source, "LLMGO_RSI_WORKER":"1", "LLMGO_VERL_OUTPUT":str(output),
            "TOKENIZERS_PARALLELISM":"false"}})
    scorer = None
    try:
        scorer = ray.remote(num_gpus=1 if allocated_gpus>1 else 0, num_cpus=1)(Lc0Scorer).options(name="lc0").remote(site,reward_config,str(output))
        teacher_identity = ray.get(scorer.ready.remote())
        atomic_json(output/"lc0-context.json", {"namespace":namespace,"name":"lc0","output":str(output)})
        atomic_json(output/"teacher.json", teacher_identity)
        with initialize_config_dir(config_dir=str(Path(verl.__file__).parent/"trainer/config"), version_base=None):
            cfg = compose(config_name="ppo_trainer", overrides=hydra_overrides(
                model,dataset,output,config,allocated_gpus,output/"lc0-context.json",resume))
        from omegaconf import OmegaConf
        atomic_json(output/"resolved-config.json",OmegaConf.to_container(cfg,resolve=True))
        from .verl_registration import ResearchTaskRunner
        from verl.trainer.main_ppo import run_ppo
        run_ppo(cfg,task_runner_class=ray.remote(num_cpus=1)(ResearchTaskRunner))
    finally:
        if scorer is not None:
            try:
                ray.get(scorer.close.remote(),timeout=30)
            finally:
                ray.kill(scorer)
        ray.shutdown()
    checkpoint = complete_checkpoint(output/"checkpoints", actors) or resume
    if checkpoint is None or int(checkpoint.name.removeprefix("global_step_")) < config["steps"]:
        raise RuntimeError("verl returned without a complete final checkpoint")
    target = output/"model"
    temporary = output/"model.exporting"
    if temporary.exists():
        shutil.rmtree(temporary)
    subprocess.run([sys.executable,"-m","verl.model_merger","merge","--backend","fsdp",
        "--local_dir",str(checkpoint/"actor"),"--target_dir",str(temporary)],check=True)
    exported = model_identity(temporary)
    if target.exists():
        raise FileExistsError("Export already exists; verify the prior submission before retrying")
    temporary.rename(target)
    atomic_json(output/"training-result.json", {"backend":"verl_ppo","sft_enabled":False,
        "allocated_gpu_count":allocated_gpus,"actor_gpus":actors,"lc0_gpus":int(allocated_gpus>1),
        "runtime":identity["runtime"],
        "checkpoint":str(checkpoint),"model_identity":{**exported,"path":str(target)},"utc":utc()})
    return target
