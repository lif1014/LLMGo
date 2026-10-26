"""Round-4 rollout-quality monitoring, stop-loss gate and weight-sync evidence.

Kept free of any `verl` import so the contracts are testable on CPU: the verl plugin
(`verl_registration`) only wires these functions into the trainer hooks.
"""
from pathlib import Path
import json

from llmgo.rsi.contracts import atomic_json, utc


class RolloutQualityError(RuntimeError):
    """Raised before an optimizer update whose on-policy rollouts already collapsed.

    Round-3 (job j000012) lost protocol legality 51/56 -> 1/56 after one iteration and
    then trained on nothing but 16384-token degenerate rollouts for two more steps.
    This gate turns that silent degradation into an explicit, logged failure so a
    broken checkpoint is never exported as a candidate model.
    """


def rollout_quality(batch, *, max_response_length):
    """On-policy verdicts taken from the reward manager's own fields, not from a proxy.

    `failure_reason` / `value_before` / `terminated` are produced inside the LC0 reward
    service (verl_reward.Lc0Scorer) and reach the trainer through the reward loop, so
    this measures exactly what training is being asked to optimize.
    """
    import torch
    with torch.no_grad():
        mask = batch.batch["response_mask"]
        scores = (batch.batch["token_level_scores"].float()*mask).sum(-1)
        advantages = batch.batch["advantages"].float()*mask
        lengths = mask.sum(-1).float()
        active = lengths > 0
        move_advantages = advantages.sum(-1)/lengths.clamp(min=1)
        total = int(scores.numel())
        reasons = [str(x) for x in list(batch.non_tensor_batch.get("failure_reason", []))]
        reason_counts = {}
        for reason in reasons:
            key = reason or "ok"
            reason_counts[key] = reason_counts.get(key,0)+1
        if len(reasons) != total:
            reason_counts = {"unavailable": total}
        values = [float(x) for x in list(batch.non_tensor_batch.get("value_before", []))]
        valid = sum(1 for reason in reasons if not reason)
        return {"samples":total, "valid":valid, "invalid":total-valid,
            "valid_rate":valid/total if total else 0.,
            "failure_counts":reason_counts,
            "score_mean":float(scores.mean()), "score_min":float(scores.min()),
            "score_max":float(scores.max()),
            "advantage_mean":float(advantages.sum()/mask.sum().clamp(min=1)),
            "advantage_min":float(advantages.min()), "advantage_max":float(advantages.max()),
            "advantage_negative_fraction":float(((move_advantages < 0) & active).sum()/active.sum().clamp(min=1)),
            "raw_td_negative_fraction":float((scores < 0).float().mean()),
            "value_before_mean":sum(values)/len(values) if values else None,
            "response_tokens_mean":float(lengths.mean()),
            "response_tokens_max":int(lengths.max()),
            "response_tokens_total":int(lengths.sum()),
            "at_token_cap_fraction":float((lengths >= max_response_length).float().mean())}


def enforce_quality_gate(quality, minimum, *, output):
    """Remove the silent-degradation hole found in round 3 (j000012, steps 2-4).

    Stopping is deliberate: the optimizer state that produced the degenerate rollouts is
    already in the actor, so the honest outcome is a failed run with recorded evidence
    rather than an exported candidate that cannot play chess.
    """
    if quality["valid_rate"] >= minimum:
        return quality
    atomic_json(output/"quality-gate.json",{"reason":"rollout_quality_gate",
        "quality_gate_min_valid":minimum,"quality":quality,
        "note":"Update rejected: the on-policy rollouts were already degenerate. "
               "No model was exported from this run.","utc":utc()})
    raise RolloutQualityError(
        f"step {quality['step']}: valid_rate {quality['valid_rate']:.3f} < {minimum} "
        f"({quality['failure_counts']}); refusing to apply the update")


def generation_throughput(metrics):
    """Real rollout tokens/second, so CUDA-graph claims are measured, not assumed."""
    seconds = metrics.get("timing_s/gen")
    tokens = metrics.get("response_length/mean",0.)*metrics.get("response_length/count",1.)
    try:
        per_token_ms = metrics.get("timing_per_token_ms/gen")
        if per_token_ms:
            return 1000./float(per_token_ms)
        if seconds and tokens:
            return float(tokens)/float(seconds)
    except (TypeError,ZeroDivisionError):
        return None
    return None


def sync_check(trainer, batch, *, samples, output):
    """Recompute actor log-probs from the live FSDP weights and compare with vLLM.

    Bypass mode assigns old_log_probs = rollout_log_probs, so the saved batches cannot
    show whether the trainer and the rollout engine agree. This recomputes them from the
    actor that is about to be updated, on the same batch, before any optimizer step:
    a large disagreement means the resumed/updated weights are not what the rollout used.
    Diagnostic only: a failure here is recorded and never silently ignored nor fatal.
    """
    import torch
    try:
        old, _ = trainer._compute_old_log_prob(batch)
        actor_logprobs = old.batch["old_log_probs"].float()
        rollout_logprobs = batch.batch["rollout_log_probs"].float()
        delta = (actor_logprobs-rollout_logprobs).abs()
        mask = batch.batch["response_mask"].bool()
        if samples and samples < delta.shape[0]:
            # Slice every tensor together: round 5 (j000015/j000016) sliced delta and the
            # mask but then indexed the *unsliced* log-prob tensors with the sliced mask,
            # so all eight sync checks died with IndexError and the post-update
            # probability agreement was never measured.
            delta, mask = delta[:samples], mask[:samples]
            actor_logprobs, rollout_logprobs = actor_logprobs[:samples], rollout_logprobs[:samples]
        values = delta[mask]
        if values.numel() == 0:
            raise ValueError("No response tokens to compare")
        ratio = torch.exp(-values)
        row = {"step":int(trainer.global_steps), "samples":int(delta.shape[0]),
            "tokens":int(values.numel()), "mean_abs_logprob_delta":float(values.mean()),
            "max_abs_logprob_delta":float(values.max()),
            "fraction_ratio_outside_0.8_1.2":float(((ratio < .8) | (ratio > 1.2)).float().mean()),
            "old_logprob_mean":float(actor_logprobs[mask].mean()),
            "rollout_logprob_mean":float(rollout_logprobs[mask].mean()),
            "utc":utc()}
    except Exception as error:  # diagnostic only
        row = {"step":int(trainer.global_steps), "error":f"{type(error).__name__}: {error}",
            "utc":utc()}
    with (output/"sync-check.jsonl").open("a") as stream:
        stream.write(json.dumps(row)+"\n")
    atomic_json(output/"sync-check-latest.json", row)
    return row


