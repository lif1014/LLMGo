"""verl 0.8 full-game PPO, terminal-only reward and frozen LC0 move-axis GAE."""
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
from .export_weights import normalize_export_weights
from .rollout import prompt_for_record

VERL_VERSION = "0.8.0"
CUDAGRAPH_MODE = "PIECEWISE"


def training_config(config, allocated_gpus=8):
    """Versioned full-game objective. Reject old single-position recipes explicitly."""
    if type(allocated_gpus) is not int or not 1 <= allocated_gpus <= 8:
        raise ValueError("verl PPO requires 1-8 allocated GPUs")
    actors = allocated_gpus-1 if allocated_gpus > 1 else 1
    defaults = {"backend":"verl_ppo", "mode":"full_game_ppo_lc0",
        "games_per_batch":actors*2,"concurrent_games":actors,"opponent_level":2,
        "minibatch_size":actors*2,"whole_batch_update":True,"steps":2,"ppo_epochs":1,"learning_rate":1e-6,
        "gamma":1.,"gae_lambda":.95,"clip_ratio":.2,"kl_beta":.05,
        "max_prompt_tokens":6144,"max_response_tokens":16384,"move_token_budget":32768,
        "attempts_per_move":2,"request_timeout_sec":540.,"turn_timeout_sec":1800.,
        "checkpoint_every":1,"seed":4170,"rollout_memory_utilization":.45,
        "rollout_enforce_eager":False,"cudagraph_capture_sizes":None,
        "lr_warmup_steps_ratio":0.,"loss_agg_mode":"seq-mean-token-mean",
        "quality_gate_min_valid":.5,"preupdate_consistency_samples":16,
        "reference_kl_warning":.05,"reference_kl_stop":.1,
        "sync_max_mean_abs_logprob_delta":.1,"sync_check_steps":0,
        "internal_validation":False,"validation_games":2,"validation_every":1,"validation_min_valid":.5,
        "validation_max_drop":.1,"validation_temperature":1.,"validation_top_p":1.,
        "validation_top_k":-1,"validation_seed":914170,"export_validated_on_stop":True,
        "optimizer_offload":"auto","rollout_wake_cache_flush":True}
    if not isinstance(config,dict) or set(config)-set(defaults):
        unknown = sorted(set(config)-set(defaults)) if isinstance(config,dict) else []
        raise ValueError(f"Unknown full-game PPO configuration {unknown}; migrate single-step/SFT/GRPO recipes")
    cfg = {**defaults,**config}
    if cfg["backend"] != "verl_ppo" or cfg["mode"] != "full_game_ppo_lc0":
        raise ValueError("Use backend=verl_ppo, mode=full_game_ppo_lc0")
    positive = ("games_per_batch","concurrent_games","minibatch_size","steps","ppo_epochs",
        "max_prompt_tokens","max_response_tokens","move_token_budget","attempts_per_move",
        "checkpoint_every","validation_every")
    for key in positive:
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"Invalid {key}")
    if cfg["minibatch_size"] % actors:
        raise ValueError("minibatch_size must divide across actor GPUs")
    for key in ("games_per_batch","validation_games"):
        if type(cfg[key]) is not int or cfg[key] < 2 or cfg[key]%2:
            raise ValueError(f"{key} must be an even count >=2 for balanced colors")
    for key in ("seed","validation_seed","sync_check_steps","preupdate_consistency_samples"):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f"Invalid {key}")
    if cfg["preupdate_consistency_samples"] > 16:
        raise ValueError("preupdate_consistency_samples must be <=16")
    from llmgo.rsi.ladder import opponent
    opponent(cfg["opponent_level"])
    from llmgo.rsi.contracts import finite
    for key,low,high in (("learning_rate",0,1),("clip_ratio",0,1),("kl_beta",0,10),
            ("gamma",0,1),("gae_lambda",0,1),("rollout_memory_utilization",0,.95),
            ("lr_warmup_steps_ratio",0,1),("quality_gate_min_valid",0,1),
            ("validation_min_valid",0,1),("validation_max_drop",0,1),
            ("reference_kl_warning",0,100),("reference_kl_stop",0,100),
            ("sync_max_mean_abs_logprob_delta",0,100),("validation_temperature",0,2),
            ("validation_top_p",0,1),("request_timeout_sec",.001,86400),
            ("turn_timeout_sec",.001,86400)):
        finite(cfg[key],key,low,high)
    if cfg["clip_ratio"] == 0 or cfg["rollout_memory_utilization"] == 0 or cfg["validation_top_p"] == 0:
        raise ValueError("clip/memory/top_p must be positive")
    if cfg["max_response_tokens"] > 32768 or cfg["max_prompt_tokens"]+cfg["max_response_tokens"] > 40960:
        raise ValueError("Training sequence exceeds the 40960-token ceiling")
    if cfg["validation_seed"] == cfg["seed"]:
        raise ValueError("Training and fixed validation need separate seed schedules")
    if cfg["internal_validation"] and cfg["validation_every"] % cfg["checkpoint_every"]:
        raise ValueError("Validation interval must be a multiple of checkpoint_every")
    if cfg["reference_kl_stop"] and cfg["reference_kl_warning"] > cfg["reference_kl_stop"]:
        raise ValueError("KL warning exceeds stop threshold")
    if cfg["sync_max_mean_abs_logprob_delta"] and (not cfg["preupdate_consistency_samples"] or cfg["sync_check_steps"]):
        raise ValueError("Sync safety requires checking every update")
    if cfg["loss_agg_mode"] != "seq-mean-token-mean":
        raise ValueError("Full-game PPO uses equal move weighting")
    for key in ("internal_validation","rollout_enforce_eager","rollout_wake_cache_flush","export_validated_on_stop","whole_batch_update"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if cfg["optimizer_offload"] not in ("auto",True,False):
        raise ValueError("optimizer_offload must be auto/boolean")
    if type(cfg["validation_top_k"]) is not int or not (cfg["validation_top_k"] == -1 or cfg["validation_top_k"] >= 1):
        raise ValueError("Invalid validation_top_k")
    if (cfg["validation_temperature"],cfg["validation_top_p"],cfg["validation_top_k"]) != (1.,1.,-1):
        raise ValueError("Full-game sync/KL checks require validation sampling temperature1/top_p1/top_k-1")
    sizes = cfg["cudagraph_capture_sizes"]
    if sizes is not None and (not isinstance(sizes,list) or not sizes
            or any(type(x) is not int or x<1 for x in sizes) or sizes != sorted(set(sizes))
            or cfg["rollout_enforce_eager"]):
        raise ValueError("Invalid CUDA graph capture sizes")
    return cfg


def actor_memory_preflight(config, allocated_gpus, *, device_bytes=79*2**30, budget_fraction=0.6):
    """Refuse an actor layout that cannot hold FSDP2 optimizer state on one GPU.

    Round-4 job j000014 requested 2 GPUs, which leaves a single actor GPU for the whole
    model, and died in the first Adam step with 76.8 GiB allocated (`_multi_tensor_adam`,
    `exp_avg_sq_sqrt`). With one actor the fp32 master weights plus the two fp32 Adam
    moments are 12 bytes per parameter before activations, the logits buffer and the
    colocated vLLM engine. This guard is arithmetic, not a measurement: it fails the
    configuration before Ray, LC0 and the first rollout are paid for, and it records the
    estimate so a later run can be compared against the real peak.
    """
    actors = allocated_gpus-1 if allocated_gpus > 1 else 1
    parameters = 4.0e9     # Qwen3.5-4B total parameters
    vocabulary = 151936    # logits buffer scales with the response token budget
    estimate = {
        "actor_gpus":actors,
        "master_and_optimizer_bytes":int(12*parameters/actors),
        "compute_copy_bytes":int(2*parameters/actors),
        "logits_bytes":int(2*vocabulary*config["max_response_tokens"]*2),
        "activation_reserve_bytes":int(4*2**30),
        "device_bytes":int(device_bytes),
        "budget_bytes":int(device_bytes*budget_fraction),
    }
    estimate["total_bytes"] = (estimate["master_and_optimizer_bytes"]
        + estimate["compute_copy_bytes"] + estimate["logits_bytes"]
        + estimate["activation_reserve_bytes"])
    estimate["without_optimizer_state_bytes"] = (estimate["total_bytes"]
        - estimate["master_and_optimizer_bytes"])
    wanted = config["optimizer_offload"]
    offload = (estimate["total_bytes"] > estimate["budget_bytes"]) if wanted == "auto" else wanted
    fitting = estimate["without_optimizer_state_bytes"] if offload else estimate["total_bytes"]
    estimate["optimizer_offload"] = offload
    estimate["fits"] = fitting <= estimate["budget_bytes"]
    if not estimate["fits"]:
        raise ValueError(
            f"{actors} actor GPU(s) cannot hold the actor for max_response_tokens="
            f"{config['max_response_tokens']}: estimated {fitting/2**30:.1f} GiB against a "
            f"{estimate['budget_bytes']/2**30:.1f} GiB budget. Request more GPUs (7 actors is "
            "the validated 8-GPU layout) or enable optimizer_offload=true.")
    return estimate


# j000016/j000023 both reported actor/perf/max_memory_reserved_gb 47.8-50.6 with the
# validated 16384-token response budget on 7 actors. That measurement, not the analytic
# estimate above (which has no way to see the cached blocks or the FSDP gather), is the
# one that decided whether the colocated rollout could re-map its weights.
MEASURED_ACTOR_RESERVED_BYTES_AT_16K = int(50*2**30)
MEASURED_ACTOR_RESERVED_TOKENS = 16384


def coexistence_preflight(config, allocated_gpus, *, device_bytes=79*2**30, reserve_bytes=3*2**30):
    """Can the actor and the colocated rollout engine share one device across an update?

    Round 8's j000023 died inside vLLM's ``CuMemAllocator.wake_up`` on the third weight
    remap, with the training process still holding everything its caching allocator had
    reserved. The two processes share the device but not an allocator, so the engine
    cannot reuse those blocks: the *live* memory plus the rollout pool has to fit, and the
    *reserved* memory plus the pool is what actually breaks.

    ``fits_without_flush`` therefore uses the measured reserved peak, and
    ``fits_with_flush`` uses the analytic live estimate, i.e. what is left once
    ``torch.cuda.empty_cache()`` (rollout_wake_cache_flush) hands the unused blocks back.
    This is arithmetic over recorded evidence, not a measurement.
    """
    actor = actor_memory_preflight(config, allocated_gpus)
    pool = int(device_bytes*config["rollout_memory_utilization"])
    scale = config["max_response_tokens"]/MEASURED_ACTOR_RESERVED_TOKENS
    reserved = int(MEASURED_ACTOR_RESERVED_BYTES_AT_16K*scale)
    live = max(actor["without_optimizer_state_bytes"], actor["total_bytes"]
        if not actor["optimizer_offload"] else actor["without_optimizer_state_bytes"])
    estimate = {
        "vllm_pool_bytes":pool,
        "actor_reserved_hint_bytes":reserved,
        "actor_live_bytes":live,
        "reserve_bytes":int(reserve_bytes),
        "device_bytes":int(device_bytes),
        "rollout_memory_utilization":config["rollout_memory_utilization"],
        "rollout_wake_cache_flush":config["rollout_wake_cache_flush"],
        "needs_cache_flush":pool + reserved + reserve_bytes > device_bytes,
    }
    estimate["without_flush_bytes"] = pool + reserved + int(reserve_bytes)
    estimate["with_flush_bytes"] = pool + live + int(reserve_bytes)
    estimate["margin_with_flush_bytes"] = int(device_bytes) - estimate["with_flush_bytes"]
    estimate["fits_without_flush"] = estimate["without_flush_bytes"] <= device_bytes
    estimate["fits_with_flush"] = estimate["with_flush_bytes"] <= device_bytes
    estimate["fits"] = estimate["fits_with_flush"] if config["rollout_wake_cache_flush"] \
        else estimate["fits_without_flush"]
    return estimate


def recipe_override(default_path, argv):
    """Resolve `--recipe PATH` and `--control` for research jobs.

    `--recipe` lets one proposal register several recipes (for example the no-op
    control at learning_rate=0) without touching ppo.json. `--control` keeps a
    diagnostic export out of submission.json so it can never be offered as the
    candidate model for the following evaluation.
    """
    path, control = Path(default_path), False
    rest = list(argv)
    while rest:
        item = rest.pop(0)
        if item == "--recipe":
            if not rest:
                raise ValueError("--recipe needs a path")
            path = Path(rest.pop(0))
        elif item == "--control":
            control = True
        else:
            raise ValueError(f"Unknown training entry argument: {item}")
    return path, control


def dataset_rows(cards, batch_size):
    """Bootstrap verl worker initialization; real states come from online games."""
    import chess
    record = {"initial_fen":chess.STARTING_FEN,"moves":[],"split":"train"}
    prompt = prompt_for_record(record,cards)
    return [{"data_source":"chess_full_game", "prompt":prompt["messages"],
        "reward_model":{"style":"rule","ground_truth":""},
        "extra_info":{"index":i,"bootstrap_only":True}} for i in range(batch_size)]


def hydra_overrides(model, dataset, output, config, allocated_gpus, teacher_context, resume=None):
    actors = allocated_gpus-1 if allocated_gpus > 1 else 1
    offload = config["optimizer_offload"]
    if offload == "auto":  # train() resolves this before the run; keep one code path
        offload = actor_memory_preflight(config, allocated_gpus)["optimizer_offload"]
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
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload":bool(offload),
        "actor_rollout_ref.actor.use_torch_compile":False,
        "actor_rollout_ref.actor.use_remove_padding":False,
        "actor_rollout_ref.actor.ppo_mini_batch_size":config["minibatch_size"],
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu":1,
        "actor_rollout_ref.actor.ppo_epochs":config["ppo_epochs"],
        "actor_rollout_ref.actor.loss_agg_mode":config["loss_agg_mode"],
        "actor_rollout_ref.actor.optim.lr":config["learning_rate"],
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio":config["lr_warmup_steps_ratio"],
        "actor_rollout_ref.actor.clip_ratio":config["clip_ratio"],
        "actor_rollout_ref.actor.clip_ratio_low":config["clip_ratio"],
        "actor_rollout_ref.actor.clip_ratio_high":config["clip_ratio"],
        "actor_rollout_ref.actor.use_kl_loss":config["kl_beta"] > 0 or config["reference_kl_stop"] > 0 or config["reference_kl_warning"] > 0,
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
        "actor_rollout_ref.rollout.enforce_eager":config["rollout_enforce_eager"],
        "actor_rollout_ref.rollout.max_model_len":config["max_prompt_tokens"]+config["max_response_tokens"],
        "actor_rollout_ref.rollout.gpu_memory_utilization":config["rollout_memory_utilization"],
        "actor_rollout_ref.rollout.agent.num_workers":1,
        "actor_rollout_ref.rollout.agent.default_agent_loop":"chess_turn_agent",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path":str(output/"agent-loops.json"),
        "actor_rollout_ref.rollout.val_kwargs.do_sample":config["validation_temperature"] > 0,
        "actor_rollout_ref.rollout.val_kwargs.temperature":config["validation_temperature"],
        "actor_rollout_ref.rollout.val_kwargs.top_p":config["validation_top_p"],
        "actor_rollout_ref.rollout.val_kwargs.top_k":config["validation_top_k"],
        "critic.enable":False,
        "algorithm.adv_estimator":"full_game_lc0_gae",
        "algorithm.gamma":config["gamma"],"algorithm.lam":config["gae_lambda"],
        "algorithm.use_kl_in_reward":False,
        "algorithm.rollout_correction.bypass_mode":True,
        "algorithm.rollout_correction.loss_type":"ppo_clip",
        "data.train_files":[str(dataset)],"data.val_files":[str(dataset)],
        "data.val_batch_size":config["minibatch_size"],
        "data.train_batch_size":config["minibatch_size"],
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
        "trainer.logger":["console"],"trainer.val_before_train":False,
        "trainer.test_freq":-1,
        "trainer.validation_data_dir":None,
        "trainer.save_freq":config["checkpoint_every"],
        "trainer.default_local_dir":str(output/"checkpoints"),
        "trainer.rollout_data_dir":str(output/"rollouts"),
        "trainer.resume_mode":"resume_path" if resume else "disable",
        "trainer.resume_from_path":str(resume) if resume else None,
    }
    result = [f"{k}={json.dumps(v)}" for k,v in values.items()]
    result.append("+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=true")
    if config["cudagraph_capture_sizes"]:
        # Bounded capture work: the vLLM default captures decode graphs up to
        # max_num_seqs (1024), while one actor GPU only ever decodes ~8 sequences.
        # This must go through engine_kwargs.compilation_config: verl 0.8.0 maps
        # RolloutConfig.cudagraph_capture_sizes to the CLI flag --cuda-graph-sizes,
        # which vLLM 0.17.0 does not define (verified in vllm/engine/arg_utils.py at
        # tag v0.17.0), so the top-level field would abort the rollout server.
        # PIECEWISE avoids asking for FULL decode graphs on this hybrid-attention
        # model, where vLLM raises instead of downgrading when the attention backend
        # never supports cudagraph.
        result.append("+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config="
            + json.dumps(json.dumps({"cudagraph_capture_sizes":config["cudagraph_capture_sizes"],
                                     "cudagraph_mode":CUDAGRAPH_MODE})))
    result.append("+data.apply_chat_template_kwargs.enable_thinking=true")
    result.append("+reward.lc0_context="+json.dumps(str(teacher_context)))
    for key,value in config.items():
        result.append("+research."+key+"="+json.dumps(value))
    result.append("+research.game_context="+json.dumps(str(output/"game-context.json")))
    return result


INFERENCE_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json",
    "merges.txt", "chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json")
# Files the evaluation stack may load besides the weights. The export must expose the
# same set the base model does: the merger's transformers-5.x `processor_config.json` has
# no counterpart in the base directory, and the base directory is the one configuration
# the pinned evaluator is known to serve.
AUXILIARY_EXPORT_FILES = INFERENCE_TOKENIZER_FILES + ("processor_config.json",)
SAFE_TOKENIZER_CLASSES = {"Qwen2Tokenizer", "Qwen2TokenizerFast"}


def _tokenizer_vocab(path):
    """Vocabulary of a model directory as seen by this process, for a compatibility check."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=False)
    return tokenizer.get_vocab()


def normalize_export_for_inference(export, base, interpreters=None):
    """Make an exported checkpoint loadable by the pinned *inference* stack.

    Round-5 job j000018 lost all 20 L2 games before the first move:
    `ValueError: Tokenizer class TokenizersBackend does not exist or is not currently
    imported.` The training stack (transformers 5.x) writes a tokenizer_config.json whose
    `tokenizer_class` is a 5.x-only backend name, while the evaluation image pins
    transformers 4.57.6 for vLLM. Weights are unaffected, and the vocabulary does not
    change during RL, so the base model's own tokenizer files - which the evaluation
    stack is known to load - are grafted onto the export, after proving that both
    vocabularies are identical. Returns a report; raises if the vocabulary differs.
    """
    export, base = Path(export), Path(base)
    if not (export/"config.json").is_file():
        raise ValueError(f"Missing HF export: {export}")
    base_vocab = _tokenizer_vocab(base)
    export_vocab = _tokenizer_vocab(export)
    if base_vocab != export_vocab:
        raise ValueError(f"Export vocabulary differs from {base}; refusing to graft a tokenizer")
    copied = {}
    removed = []
    for name in AUXILIARY_EXPORT_FILES:
        target = export/name
        if target.is_file() and not (base/name).is_file():
            target.unlink()
            removed.append(name)
    for name in INFERENCE_TOKENIZER_FILES:
        source = base/name
        if not source.is_file():
            continue
        target = export/name
        shutil.copyfile(source, target)
        copied[name] = file_hash(target)
    return {"base": str(base), "export": str(export), "vocabulary_size": len(base_vocab),
        "copied": copied, "removed": removed,
        "source_hashes": {name: file_hash(base/name) for name in copied},
        "dynamic_verification": verify_export_tokenizer(export, interpreters)}


def verify_export_tokenizer(export, interpreters=None):
    """Load the exported tokenizer with the pinned inference stack, if it is reachable.

    The training venv ships transformers 5.x, which accepts tokenizer classes that the
    evaluation image (transformers 4.57.6) rejects, so importing locally cannot answer
    the only question that matters: does the *evaluator* load this directory? Each
    candidate interpreter is asked for its transformers version first and only a 4.x
    interpreter is used. When none is reachable the check reports itself as skipped
    instead of pretending to have passed.
    """
    import subprocess
    program = ("import json,sys,transformers\n"
               "from transformers import AutoTokenizer\n"
               "out={'transformers':transformers.__version__}\n"
               "try:\n"
               "    t=AutoTokenizer.from_pretrained(sys.argv[1],trust_remote_code=False)\n"
               "    out.update(ok=True,klass=type(t).__name__,tokens=len(t))\n"
               "except Exception as error:\n"
               "    out.update(ok=False,error=f'{type(error).__name__}: {error}')\n"
               "print(json.dumps(out))\n")
    candidates = list(interpreters or []) + [sys.executable, "/usr/bin/python3",
        shutil.which("python3") or "/usr/bin/python3"]
    tried = []
    for interpreter in candidates:
        if not interpreter or not Path(interpreter).exists():
            continue
        try:
            result = subprocess.run([interpreter, "-c", program, str(export)],
                capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as error:
            tried.append({"interpreter": interpreter, "error": f"{type(error).__name__}: {error}"})
            continue
        row = None
        for line in reversed(result.stdout.splitlines()):
            try:
                row = json.loads(line)
                break
            except ValueError:
                continue
        if row is None:
            tried.append({"interpreter": interpreter,
                "error": (result.stderr or result.stdout)[-300:] or "no output"})
            continue
        row["interpreter"] = interpreter
        if str(row.get("transformers", "")).startswith("4."):
            row["skipped"] = False
            return row
        tried.append({"interpreter": interpreter, "transformers": row.get("transformers"),
            "note": "not the pinned inference stack; ignored"})
    return {"skipped": True, "reason": "no transformers-4 interpreter found",
        "tried": tried,
        "static_ok": _static_tokenizer_ok(export)}


def _static_tokenizer_ok(export):
    """Fallback check that needs no second interpreter: no 5.x-only tokenizer class."""
    config = Path(export)/"tokenizer_config.json"
    if not config.is_file():
        return False
    declared = read_json(config).get("tokenizer_class")
    return declared is None or declared in SAFE_TOKENIZER_CLASSES


def complete_checkpoint(root, actor_gpus, *, allowed_steps=None, require_game_state=False):
    """Never resume a half-written checkpoint or a different distributed world size."""
    candidates = []
    for path in Path(root).glob("global_step_*"):
        try:
            step = int(path.name.removeprefix("global_step_"))
        except ValueError:
            continue
        if allowed_steps is not None and step not in allowed_steps:
            continue
        if require_game_state and not (path/"games.json").is_file():
            continue
        actor = path/"actor"
        if ((path/"data.pt").is_file() and (actor/"huggingface/config.json").is_file()
                and all((actor/f"{kind}_world_size_{actor_gpus}_rank_{i}.pt").is_file()
                        for kind in ("model","optim","extra_state") for i in range(actor_gpus))):
            candidates.append((step,path))
    return max(candidates, default=(0,None))[1]


def experiment_identity(model, cards, site, reward_config, config, allocated_gpus):
    import importlib.metadata
    from llmgo.rsi.evaluator import engine_identity
    from llmgo.rsi.ladder import opponent
    harness = Path(__file__).parents[1]/"harness"
    harness_files = {p.relative_to(harness).as_posix():file_hash(p) for p in harness.rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    return {"model":model_identity(model)["sha256"], "objective":"full_game_ppo_lc0", "cards":digest(cards),
        "environment":{"invalid_move_policy":"random_legal","action_attribution":"student_player",
            "training_samples":"original_model_responses","max_plies":200},
        "training":config,"reward":reward_config,"allocated_gpus":allocated_gpus,"verl":VERL_VERSION,
        "runtime":{name:importlib.metadata.version(name) for name in ("verl","transformers","torch","vllm","tokenizers","ray")},
        "code":{name:file_hash(Path(__file__).with_name(name)) for name in
            ("verl_ppo.py","verl_registration.py","verl_reward.py","vllm_compat.py","lc0.py",
             "rollout.py","export_weights.py","verl_memory.py","trajectories.py","quality.py","stability.py",
             "game.py","verl_games.py","verl_agent.py","validation.py","baseline_import.py")},
        "lc0_binary":file_hash(site["lc0_bin"]), "harness":harness_files,
        "fairy":engine_identity(site["engine"],opponent(config["opponent_level"]))}



def train(model, cards, site, reward_config, config, output, *, allocated_gpus=8, resume_from=None):
    if os.environ.get("LLMGO_RSI_WORKER") != "1":
        raise RuntimeError("Submit PPO through a GPU research job")
    import importlib.metadata
    if importlib.metadata.version("verl") != VERL_VERSION:
        raise RuntimeError(f"Expected verl=={VERL_VERSION}; run scripts/bootstrap_verl.py in the designated image")
    from .lc0 import teacher_config
    reward_config = teacher_config(reward_config)
    config = training_config(config, allocated_gpus)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    actors = allocated_gpus-1 if allocated_gpus > 1 else 1
    if allocated_gpus == 1 and reward_config["backend"].startswith("cuda"):
        raise ValueError("One-GPU PPO needs a CPU LC0 backend (eigen/blas) or request at least two GPUs")
    if allocated_gpus > 1 and not reward_config["backend"].startswith("cuda"):
        raise ValueError("Multi-GPU default reserves one GPU for LC0; select a CUDA backend")
    memory = actor_memory_preflight(config, allocated_gpus)
    config["optimizer_offload"] = memory["optimizer_offload"]
    coexistence = coexistence_preflight(config, allocated_gpus)
    if not coexistence["fits_with_flush"]:
        raise ValueError(
            "The colocated rollout pool and the actor cannot share one device even with the "
            f"wake-up cache flush: {coexistence['with_flush_bytes']/2**30:.1f} GiB against "
            f"{coexistence['device_bytes']/2**30:.1f} GiB. Lower rollout_memory_utilization "
            f"(now {config['rollout_memory_utilization']}) or shorten max_response_tokens.")
    if coexistence["needs_cache_flush"] and not config["rollout_wake_cache_flush"]:
        raise ValueError(
            "This layout needs rollout_wake_cache_flush=true: the measured actor peak "
            f"({coexistence['actor_reserved_hint_bytes']/2**30:.1f} GiB reserved) plus the "
            f"rollout pool ({coexistence['vllm_pool_bytes']/2**30:.1f} GiB) exceeds the device, "
            "which is how j000023 died in vLLM's wake_up.")
    atomic_json(output/"memory-preflight.json", {**memory, "coexistence":coexistence})
    identity = experiment_identity(model,cards,site,reward_config,config,allocated_gpus)
    from .stability import select_export_checkpoint
    atomic_json(output/"agent-loops.json", [{"name":"chess_turn_agent",
        "_target_":"llmgo.training.verl_agent.ChessTurnAgentLoop"}])
    dataset = output/"worker-bootstrap.jsonl"
    dataset.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n"
        for row in dataset_rows(cards,config["minibatch_size"])))
    atomic_json(output/"game-context.json",{"site":site,"cards":cards,"output":str(output),
        "identity":digest(identity),"initial_fen":"standard","max_plies":200,
        "environment":identity['environment']})
    resume = None
    if resume_from:
        old = Path(resume_from)/"training"
        if not (old/"identity.json").is_file() or read_json(old/"identity.json") != identity:
            raise ValueError("Resume identity changed; submit a new experiment rather than mixing optimizer state")
        resume = complete_checkpoint(old/"checkpoints", actors, require_game_state=True)
        if resume is None:
            raise ValueError("No complete verl checkpoint is available in the previous task")
        state = read_json(resume/"games.json")
        if state["identity"] != digest(identity):
            raise ValueError("Resume full-game state identity changed")
        if state.get("internal_validation") is not config["internal_validation"]:
            raise ValueError("Resume internal validation mode changed")
        if config["internal_validation"]:
            atomic_json(output/"validation-baseline.json",state["validation_baseline"])
        elif state.get("validation_baseline") is not None:
            raise ValueError("Validation-disabled resume has an unexpected baseline")
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
            "TOKENIZERS_PARALLELISM":"false",
            # Read by llmgo.training.verl_registration in every worker process.
            "LLMGO_WAKE_CACHE_FLUSH":"1" if config["rollout_wake_cache_flush"] else "0"}})
    scorer = None
    stopped_early = False
    stopping_error = None
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
        try:
            run_ppo(cfg,task_runner_class=ray.remote(num_cpus=1)(ResearchTaskRunner))
        except Exception as error:
            # Ray wraps the typed safety stop. An unrelated OOM/runtime error must
            # never be converted into an apparently successful model export.
            if not (config["internal_validation"] and config["export_validated_on_stop"]
                    and (output/"stability-stop.json").is_file() and "TrainingSafetyStop" in str(error)):
                raise
            stopped_early = True
            stopping_error = error
    finally:
        if scorer is not None:
            try:
                ray.get(scorer.close.remote(),timeout=30)
            finally:
                ray.kill(scorer)
        ray.shutdown()
    try:
        checkpoint, selection = select_export_checkpoint(output,actors,
            internal_validation=config["internal_validation"],experiment_digest=digest(identity))
    except RuntimeError:
        atomic_json(output/"checkpoint-selection.json", {"status":"no_eligible_checkpoint",
            "stopped_early":stopped_early,"action":"no model exported"})
        if stopping_error is not None:
            raise stopping_error
        raise
    atomic_json(output/"checkpoint-selection.json", {"checkpoint":str(checkpoint),
        "stopped_early":stopped_early,"internal_validation":config["internal_validation"],"validation":selection,
        "rule":("complete checkpoint and passed post-update full games, KL and sync; formal chess required"
            if config["internal_validation"] else
            "latest complete checkpoint with a recorded finite optimizer update; internal validation disabled; formal chess required")})
    target = output/"model"
    temporary = output/"model.exporting"
    if temporary.exists():
        shutil.rmtree(temporary)
    subprocess.run([sys.executable,"-m","verl.model_merger","merge","--backend","fsdp",
        "--local_dir",str(checkpoint/"actor"),"--target_dir",str(temporary)],check=True)
    # The merger writes both the tokenizer side and the parameter names with the
    # training stack's transformers 5.x conventions. The 5.x parameter naming is not
    # something vLLM can resolve: round-7 jobs j000019/j000020 died at engine start with
    # `Following weights were not initialized from checkpoint` because the merged names
    # carry an extra `language_model` level. Remap the names (streaming, payload
    # untouched) and refuse the run if the result does not match the base checkpoint.
    weights = normalize_export_weights(temporary, model)
    # The tokenizer side is normalized for the same reason: round-5 job j000018 lost
    # every L2 game to a 5.x-only tokenizer class. Normalize before hashing the identity
    # so the recorded identity is the directory that will actually be evaluated.
    # The evaluation image pins its inference interpreter in the site manifest; asking it
    # first is what makes this a check of the evaluator rather than of the training venv.
    interpreters = [site.get("inference_python")] if site.get("inference_python") else []
    normalization = normalize_export_for_inference(temporary, model, interpreters)
    exported = model_identity(temporary)
    if target.exists():
        raise FileExistsError("Export already exists; verify the prior submission before retrying")
    temporary.rename(target)
    normalization["export"] = str(target)
    verdict = normalization["dynamic_verification"]
    loadable = verdict.get("ok") if "ok" in verdict else bool(verdict.get("static_ok"))
    atomic_json(output/"export-preflight.json", {"ok":loadable,"normalization":normalization,
        "weights":weights,
        "note":"Tensor payloads are untouched: the parameter names are remapped to the base "
               "checkpoint naming and the tokenizer/processor side is replaced with the base "
               "model's own files, after proving the key sets and both vocabularies agree."})
    if not loadable:
        raise RuntimeError(f"Exported model is not loadable by the pinned inference stack: {verdict}")
    atomic_json(output/"training-result.json", {"backend":"verl_ppo","mode":"full_game_ppo_lc0","sft_enabled":False,
        "environment":identity['environment'],"internal_validation":config["internal_validation"],
        "stopped_early":stopped_early,"selected_checkpoint_step":int(checkpoint.name.removeprefix("global_step_")),
        "validation":selection,
        "allocated_gpu_count":allocated_gpus,"actor_gpus":actors,"lc0_gpus":int(allocated_gpus>1),
        "runtime":identity["runtime"],
        "checkpoint":str(checkpoint),"model_identity":{**exported,"path":str(target)},"utc":utc()})
    atomic_json(output/"health.json", {"stage":"exported_for_evaluation","updated":utc(),
        "stopped_early":stopped_early,"selected_checkpoint_step":int(checkpoint.name.removeprefix("global_step_")),
        "internal_validation":config["internal_validation"],
        "validation":selection,"model_identity":{**exported,"path":str(target)}})
    return target
