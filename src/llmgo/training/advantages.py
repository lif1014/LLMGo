"""Separate legal-move TD advantages from model-output protocol penalties."""
from __future__ import annotations

import math
from pathlib import Path

from llmgo.rsi.contracts import atomic_json, utc

MODE = "lc0_td0_protocol_guarded"
PROTOCOL_FAILURES = {"invalid_final_answer", "incomplete_thinking", "length"}


def guarded_advantages(token_level_rewards, response_mask, failure_reasons, *, penalty=1.0):
    import torch
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)) or not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("Protocol failure penalty must be finite and positive")
    if token_level_rewards.ndim != 2 or token_level_rewards.shape != response_mask.shape:
        raise ValueError("Reward/mask must be aligned [moves, tokens]")
    if not bool(torch.isfinite(token_level_rewards).all()) or not bool(((response_mask == 0) | (response_mask == 1)).all()):
        raise ValueError("Nonfinite rewards or nonbinary response mask")
    reasons = list(failure_reasons)
    if len(reasons) != token_level_rewards.shape[0] or any(not isinstance(r, str) or r and r not in PROTOCOL_FAILURES for r in reasons):
        raise ValueError("Missing or unsupported protocol verdict; do not infer validity from LC0 score")
    with torch.no_grad():
        mask = response_mask.to(dtype=token_level_rewards.dtype)
        lengths = mask.sum(-1)
        active = lengths > 0
        valid = torch.tensor([r == "" for r in reasons], dtype=torch.bool, device=mask.device) & active
        invalid = ~valid & active
        raw = (token_level_rewards*mask).sum(-1)
        baseline = raw[valid].mean() if bool(valid.any()) else raw.new_tensor(0.)
        per_move = torch.where(valid, raw-baseline, raw.new_full(raw.shape, -float(penalty)))
        per_move = torch.where(active, per_move, raw.new_zeros(raw.shape))
        advantages = per_move[:, None]*mask
        if bool((per_move[invalid] >= 0).any()):
            raise ValueError("Protocol guard invariant violated: invalid output has nonnegative advantage")
        rows = [{"index":i, "failure_reason":r or None, "active":bool(active[i]),
                 "response_tokens":int(lengths[i]), "raw_td":float(raw[i]),
                 "final_advantage":float(per_move[i])} for i,r in enumerate(reasons)]
        diag = {"mode":MODE, "valid_baseline":float(baseline), "protocol_failure_penalty":float(penalty),
                "valid":int(valid.sum()), "invalid":int(invalid.sum()),
                "invalid_positive_advantages":int((per_move[invalid] > 0).sum()),
                "invalid_advantage_max":float(per_move[invalid].max()) if bool(invalid.any()) else None,
                "final_advantage_min":float(per_move.min()) if len(rows) else None,
                "final_advantage_max":float(per_move.max()) if len(rows) else None, "samples":rows}
    return advantages, advantages.clone(), diag


def apply_protocol_guard(batch, output, step, *, penalty=1.0):
    """Wire the final advantages before saving the batch and calling the optimizer."""
    reasons = batch.non_tensor_batch.get("failure_reason")
    if reasons is None:
        raise ValueError("Reward manager did not provide protocol verdicts")
    advantages, returns, diag = guarded_advantages(batch.batch["token_level_rewards"],
        batch.batch["response_mask"], reasons, penalty=penalty)
    batch.batch["advantages"] = advantages
    batch.batch["returns"] = returns
    diag.update(step=int(step), utc=utc())
    for row, ref in zip(diag["samples"], batch.non_tensor_batch.get("sample_path", [])):
        row["sample_path"] = str(ref)
    atomic_json(Path(output)/"protocol-advantages"/f"step-{step:08d}.json", diag)
    atomic_json(Path(output)/"protocol-advantages-latest.json", diag)
    return {key:value for key,value in diag.items() if key not in {"samples", "utc"}}
