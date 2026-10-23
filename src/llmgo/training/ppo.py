"""On-policy token-clipped PPO with a frozen offline Lc0 critic, without SFT.

A chess decision is one complete generated response. Its TD(0) advantage is
broadcast to generated tokens; PPO clips token probability ratios against a
frozen behavior snapshot across multiple minibatch epochs. There is no trained
critic, group baseline, GAE, or replay of older policy batches.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

from llmgo.rsi.contracts import atomic_json, digest, read_json, utc, finite, file_hash, model_identity
from .rewards import validate_reward, evaluate_action
from .rollout import prompt_for_record


def training_config(config):
    if not isinstance(config, dict):
        raise ValueError("RL configuration must be an object")
    defaults = {"backend": "hf_ppo", "batch_size": 8, "steps": 16,
                "ppo_epochs": 4, "minibatch_size": 2, "clip_range": .2,
                "learning_rate": 1e-5, "kl_beta": .02, "lora_rank": 16,
                "max_response_tokens": 2048, "max_prompt_tokens": 8192,
                "seed": 4170, "checkpoint_every": 4}
    if set(config) - set(defaults):
        raise ValueError("Unknown RL configuration; arbitrary trainer commands are not accepted")
    result = {**defaults, **config}
    if result["backend"] != "hf_ppo":
        raise ValueError("Legacy HF test implementation only; the default train_ppo.py entry uses llmgo.training.verl_ppo. Migrate configs explicitly")
    for key, low, high in (("batch_size", 1, 256), ("steps", 1, 100000),
                           ("ppo_epochs", 1, 16), ("minibatch_size", 1, 256),
                           ("lora_rank", 1, 256), ("max_response_tokens", 16, 32768),
                           ("max_prompt_tokens", 1024, 8192),
                           ("checkpoint_every", 1, 10000), ("seed", 0, 2**31-1)):
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(f"Invalid {key}")
    if result["minibatch_size"] > result["batch_size"]:
        raise ValueError("minibatch_size exceeds batch_size")
    finite(result["learning_rate"], "learning_rate", 1e-8, 1e-3)
    finite(result["kl_beta"], "kl_beta", 0, 1)
    finite(result["clip_range"], "clip_range", .01, .5)
    return result


class BehaviorLogprobs:
    """Capture actual sampling probabilities with only one vocabulary row in memory.

    HF calls processors before sampling. At the next call input_ids reveals the
    previous sampled token; finish captures the final token (including EOS).
    GenerationConfig has no other processors/warpers that alter these logits.
    """
    def __init__(self):
        self.previous = self.normalizer = None
        self.rows = []

    def __call__(self, input_ids, scores):
        if scores.ndim != 2 or scores.shape[0] != 1:
            raise ValueError("Behavior recorder supports exactly one sampled response")
        if self.previous is not None:
            self.rows.append(self.previous[input_ids[0, -1]] - self.normalizer)
        self.previous = scores[0].detach().float().clone()
        self.normalizer = self.previous.logsumexp(-1)
        return scores

    def finish(self, response_ids):
        import torch
        if self.previous is None or len(self.rows) != response_ids.numel()-1:
            raise ValueError("Generated tokens and captured behavior probabilities differ")
        self.rows.append(self.previous[response_ids[-1]] - self.normalizer)
        result = torch.stack(self.rows).cpu().tolist()
        self.previous = self.normalizer = None
        self.rows = []
        return result


def policy_loss(logp, old_logp, reference_logp, advantage, beta=.02, clip_range=.2):
    """One response, equal weight per decision; all response tokens are actions."""
    import torch
    old_logp, reference_logp = old_logp.detach(), reference_logp.detach()
    if logp.ndim != 1 or logp.numel() == 0 or logp.shape != old_logp.shape or logp.shape != reference_logp.shape:
        raise ValueError("PPO requires aligned nonempty generated token logprobs")
    advantage = torch.as_tensor(advantage, device=logp.device, dtype=logp.dtype).detach()
    if advantage.numel() != 1:
        raise ValueError("One chess-decision advantage must be broadcast to response tokens")
    # Use float64 for exponentiation; never recompute the denominator from new logits.
    log_ratio = (logp - old_logp).double()
    ratio = log_ratio.exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1-clip_range, 1+clip_range) * advantage)
    ref_ratio = (reference_logp - logp).double()
    kl = ref_ratio.exp() - ref_ratio - 1
    loss = (-surrogate + beta * kl).mean()
    metrics = {"kl": float(kl.detach().mean()),
               "approx_old_kl": float((ratio - 1 - log_ratio).detach().mean()),
               "clip_fraction": float(((ratio-1).abs() > clip_range).double().mean().detach())}
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite PPO loss; no optimizer step or candidate export")
    return loss, metrics


def update_batch(model, optimizer, batch, config):
    """Keep rollout logprobs/advantages immutable across all PPO epochs."""
    import torch
    if not batch or not any(sample["advantage"] for sample in batch):
        return []
    params = [p for p in model.parameters() if p.requires_grad]
    model.train()
    history = []
    for epoch in range(config["ppo_epochs"]):
        order = list(range(len(batch)))
        random.shuffle(order)
        for offset in range(0, len(order), config["minibatch_size"]):
            indexes = order[offset:offset+config["minibatch_size"]]
            optimizer.zero_grad()
            metrics = []
            for i in indexes:
                sample = batch[i]
                device = next(model.parameters()).device
                ids = torch.tensor([sample["prompt_token_ids"]], device=device)
                response = torch.tensor(sample["generated_token_ids"], device=device)
                logp = _completion_logps(model, ids, response)
                old = torch.tensor(sample["behavior_logprobs"], device=device, dtype=torch.float32)
                reference = torch.tensor(sample["reference_logprobs"], device=device, dtype=torch.float32)
                loss, info = policy_loss(logp, old, reference, sample["advantage"],
                                         config["kl_beta"], config["clip_range"])
                (loss / len(indexes)).backward()
                metrics.append({**info, "loss": float(loss.detach())})
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            # No KL-only pseudo-improvement when the entire rollout has zero signal.
            optimizer.step()
            history.append({"epoch": epoch, "minibatch": offset // config["minibatch_size"],
                            **{k: sum(m[k] for m in metrics)/len(metrics) for k in metrics[0]}})
    return history


def append_event(path, event):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def save_checkpoint(model, optimizer, output, next_step, updates, identity):
    """Only an atomically published completed batch is resumable."""
    import shutil
    import torch
    output = Path(output)
    checkpoint = output / f"checkpoint-{next_step:06d}"
    temporary = output / f".checkpoint-{next_step:06d}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(exist_ok=False)
    model.save_pretrained(temporary)
    torch.save({"optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "random_rng": random.getstate()}, temporary / "optimizer-rng.pt")
    hashes = {p.name: file_hash(p) for p in temporary.iterdir() if p.is_file()}
    atomic_json(temporary / "state.json", {"next_step": next_step, "updates": updates,
                                          "identity": identity, "files": hashes})
    temporary.rename(checkpoint)
    return checkpoint


def restore_checkpoint(model, optimizer, checkpoint, identity, *, load_adapter=None):
    import torch
    checkpoint = Path(checkpoint)
    state = read_json(checkpoint / "state.json")
    if state["identity"] != identity:
        raise ValueError("Resume identity differs from current experiment")
    if not state.get("files") or any(Path(name).name != name or file_hash(checkpoint/name) != sha
                                     for name, sha in state["files"].items()):
        raise ValueError("Resume checkpoint file hash mismatch")
    if load_adapter is None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        load_adapter = lambda model, path: set_peft_model_state_dict(model, load_file(str(path / "adapter_model.safetensors")))
    load_adapter(model, checkpoint)
    saved = torch.load(checkpoint / "optimizer-rng.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["cpu_rng"])
    if saved["cuda_rng"]:
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    random.setstate(saved["random_rng"])
    return state["next_step"], state["updates"]


def load_hf_model(path):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    config = AutoConfig.from_pretrained(path, trust_remote_code=False)
    # Preserve the Qwen3.5 multimodal architecture; only use its text interface.
    cls = AutoModelForImageTextToText if config.model_type in {"qwen3_5", "qwen3_5_moe"} else AutoModelForCausalLM
    return cls.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=False)


def _completion_logps(model, prompt_ids, response_ids):
    import torch
    sequence = torch.cat([prompt_ids, response_ids.unsqueeze(0)], dim=1)
    count = response_ids.numel()
    # Request only needed suffix logits when supported by the current HF model.
    import inspect
    kwargs = {}
    if "logits_to_keep" in inspect.signature((model.get_base_model() if hasattr(model, "get_base_model") else model).forward).parameters:
        kwargs["logits_to_keep"] = count + 1
    output = model(input_ids=sequence, attention_mask=torch.ones_like(sequence), use_cache=False, **kwargs)
    logits = output.logits[:, -(count+1):-1, :].float()
    return torch.log_softmax(logits, -1).gather(-1, response_ids[None, :, None]).squeeze(0).squeeze(-1)


def train(model_path, records, cards, provider, reward_config, config, output, *, resume=None, progress=None):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("RL requires a GPU Worker")
    return _train(model_path, records, cards, provider, reward_config, config, output,
                  resume=resume, progress=progress, device="cuda")


def _train(model_path, records, cards, provider, reward_config, config, output, *, resume=None, progress=None, device="cuda"):
    """Shared training implementation; CPU is used only by the tiny architecture test."""
    import torch
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model
    reward_config = validate_reward(reward_config)
    config = training_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = load_hf_model(model_path)
    suffixes = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    targets = [name for name, _ in model.named_modules() if name.rsplit(".", 1)[-1] in suffixes
               and not any(part in name for part in ("visual", "vision_tower"))]
    if not targets:
        raise RuntimeError("No supported language adapter modules found; inspect the actual Qwen architecture")
    adapter = LoraConfig(r=config["lora_rank"], lora_alpha=2*config["lora_rank"],
                         target_modules=targets,
                         lora_dropout=0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, adapter).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config["learning_rate"], weight_decay=0)
    identity = {"parent_model": str(model_path), "parent_sha256": model_identity(model_path)["sha256"], "config": config,
                "curriculum": digest(records), "skills": digest(cards), "critic": provider.identity, "reward": reward_config}
    start, updates = 0, 0
    if resume:
        start, updates = restore_checkpoint(model, optimizer, resume, identity)
    log = output / "training.jsonl"
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    for step in range(start, config["steps"]):
        batch = []
        attempt = f"batch-{step:06d}-{time.time_ns()}"
        raw_path = output / "rollouts" / f"{attempt}-sampled.jsonl"
        target_path = output / "rollouts" / f"{attempt}-targets.jsonl"
        batch_path = output / "rollouts" / f"{attempt}.json"
        atomic_json(output / "latest.json", {"step": step, "stage": "sampling", "raw_path": str(raw_path),
                                           "targets_path": str(target_path), "rollout_path": str(batch_path)})
        behavior_version = updates
        if progress:
            progress("sampling", "PPO：当前策略采样；Lc0仅评估实际选择的走法", batch=step+1,
                     batches=config["steps"], backend="hf_ppo", policy_version=behavior_version)
        model.eval()
        for index in range(config["batch_size"]):
            record = random.choice(records)
            prompt = prompt_for_record(record, cards)
            ids = tokenizer.apply_chat_template(prompt["messages"], tokenize=True, add_generation_prompt=True,
                                               enable_thinking=True, return_tensors="pt", return_dict=False).to(device)
            if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
                raise ValueError("Tokenizer returned an empty or invalid prompt")
            if ids.shape[1] > config["max_prompt_tokens"]:
                raise ValueError("Prompt too long; refusing to truncate legal moves or skills")
            started = time.monotonic()
            with torch.no_grad():
                # Identity warpers: the scored distribution is exactly the sampling distribution.
                from transformers import GenerationConfig
                generation = GenerationConfig(do_sample=True, temperature=1., top_p=1., top_k=0,
                    repetition_penalty=1., max_new_tokens=config["max_response_tokens"],
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=eos, use_cache=True)
                recorder = BehaviorLogprobs()
                # New HF versions fill unspecified fields from model.generation_config.
                # Replace that fallback temporarily too, so parent warpers/forced EOS
                # cannot change the behavior distribution or introduce forced actions.
                base = model.get_base_model()
                inherited_generation = base.generation_config
                try:
                    base.generation_config = generation
                    full = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                          generation_config=generation, logits_processor=[recorder])
                finally:
                    base.generation_config = inherited_generation
                response = full[0, ids.shape[1]:]
                text = tokenizer.decode(response, skip_special_tokens=False)
                terminated = bool(response.numel() and int(response[-1]) in eos_ids)
                if not response.numel():
                    raise RuntimeError("Empty response has no actions to optimize")
                old = recorder.finish(response)
                recomputed = _completion_logps(model, ids, response).cpu().tolist()
                behavior_gap = max(abs(a-b) for a,b in zip(old, recomputed))
                with model.disable_adapter():
                    reference = _completion_logps(model, ids, response).cpu().tolist()
            batch.append({"index": index, "policy_version": behavior_version,
                "position_id": prompt["position_id"], "record": record, "skills": prompt["skills"],
                "prompt": prompt["prompt"], "prompt_token_ids": ids[0].tolist(),
                "prompt_mask": [0] * ids.shape[1], "generated_token_ids": response.tolist(),
                "response_mask": [1] * response.numel(), "behavior_logprobs": old,
                "reference_logprobs": reference, "behavior_recompute_max_abs_delta": behavior_gap, "raw_output": text,
                "source": "model", "finish_reason": "stop" if terminated else "length",
                "generated_tokens": response.numel(), "seconds": time.monotonic()-started,
                "generation_truncated": not terminated})
            append_event(raw_path, {"identity": identity, **batch[-1]})
            if progress:
                progress("sampling", "PPO：已保存模型回答与行为概率", batch=step+1,
                         sampled=index+1, batch_size=config["batch_size"], backend="hf_ppo")
        # Serialize GPU use: actor offload -> Lc0 scores only sampled moves -> actor reload.
        model.to("cpu")
        if device == "cuda":
            torch.cuda.empty_cache()
        try:
            for index, sample in enumerate(batch):
                sample.update(evaluate_action(sample["record"], sample["raw_output"], provider,
                    truncated=sample["generation_truncated"], reward_config=reward_config))
                append_event(target_path, {"index": index, **sample})
                if progress:
                    progress("labeling", "PPO：Lc0固定critic评分，学生权重已移至CPU", batch=step+1,
                             labeled=index+1, batch_size=len(batch), backend="hf_ppo")
        finally:
            provider.close()
        model.to(device)
        # Durable buffer is evidence, not a replay source. Resume uses completed batch checkpoints.
        atomic_json(batch_path, {"identity": identity, "step": step,
                               "policy_version": behavior_version, "samples": batch})
        if progress:
            progress("updating", "PPO：固定old logprob与Lc0优势，执行多轮clip更新", batch=step+1,
                     backend="hf_ppo", rollout_path=str(batch_path))
        metrics = update_batch(model, optimizer, batch, config) if any(s["advantage"] for s in batch) else []
        updates += len(metrics)
        event = {"step": step, "utc": utc(), "backend": "hf_ppo", "policy_version": behavior_version,
                 "updates": updates, "rollout_path": str(batch_path), "minibatches": metrics,
                 "mean_advantage": sum(s["advantage"] for s in batch)/len(batch),
                 "invalid_responses": sum(s["failure_reason"] is not None for s in batch),
                 "updated": bool(metrics), "skipped_reason": None if metrics else "zero_advantage",
                 "sft_loss": 0, "critic_loss": 0, "identity": identity}
        append_event(log, event)
        atomic_json(output / "latest.json", {"step": step, "stage": "updated", "raw_path": str(raw_path),
            "targets_path": str(target_path), "rollout_path": str(batch_path), "summary": {k: v for k, v in event.items() if k not in {"identity", "minibatches"}},
            "last_minibatch": metrics[-1] if metrics else None})
        if progress:
            progress("updated", "PPO：批次更新完成，数值诊断见training.jsonl", batch=step+1,
                     updates=updates, backend="hf_ppo", mean_advantage=event["mean_advantage"],
                     invalid_responses=event["invalid_responses"], rollout_path=str(batch_path))
        if (step + 1) % config["checkpoint_every"] == 0 or step + 1 == config["steps"]:
            save_checkpoint(model, optimizer, output, step+1, updates, identity)
    if updates == 0:
        raise RuntimeError("No RL updates: all sampled advantages were zero. Change the curriculum; no candidate exported.")
    export = output / "model"
    if export.exists():
        raise FileExistsError(export)
    merged = model.merge_and_unload()
    merged.save_pretrained(export, safe_serialization=True)
    tokenizer.save_pretrained(export)
    # Preserve processor metadata of the original multimodal model.
    import shutil
    for name in ("preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json"):
        source = Path(model_path) / name
        if source.exists():
            shutil.copy2(source, export / name)
    atomic_json(output / "training-result.json", {"backend": "hf_ppo", "updates": updates,
                                                 "model_path": str(export), "identity": identity,
                                                 "loss": "token_clipped_ppo_fixed_lc0_td0_plus_reference_kl",
                                                 "sft_enabled": False})
    return export
