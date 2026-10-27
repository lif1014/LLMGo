"""Explicit PPO stop rules and fixed, disjoint validation checkpoint receipts."""
from __future__ import annotations

import json
import math
from pathlib import Path

from llmgo.rsi.contracts import atomic_json, digest, read_json, utc
from .quality import RolloutQualityError


class TrainingSafetyStop(RolloutQualityError):
    """An intentional stop; only previously validated checkpoints may be exported."""


def safety_stop(output, reason, evidence):
    def safe(value):
        if isinstance(value,float) and not math.isfinite(value):
            return str(value)
        if isinstance(value,dict):
            return {k:safe(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):
            return [safe(v) for v in value]
        return value
    evidence = safe(evidence)
    atomic_json(Path(output)/"stability-stop.json", {
        "reason":reason, "evidence":evidence, "utc":utc(),
        "action":"refused unsafe continuation; inspect previously validated checkpoints"})
    atomic_json(Path(output)/"health.json",{"stage":"safety_stopped","reason":reason,"evidence":evidence,"updated":utc()})
    raise TrainingSafetyStop(f"{reason}: {evidence}")


def batch_diagnostics(batch, *, think_end_id=None, eos_ids=()):
    """Coefficient statistics, never described as parameter-gradient contributions."""
    import torch
    with torch.no_grad():
        t = batch.batch
        mask = t["response_mask"].bool()
        advantages = t["advantages"].float()
        reasons = list(batch.non_tensor_batch["failure_reason"])
        invalid = torch.tensor([bool(x) for x in reasons], device=mask.device)
        lengths = mask.sum(-1)
        per_move = (advantages*mask).sum(-1)/lengths.clamp(min=1)
        result = {}
        for label, selected in (("valid",~invalid),("invalid",invalid)):
            active = selected & (lengths > 0)
            m = mask & active[:,None]
            result[label] = {"samples":int(active.sum()),"tokens":int(m.sum()),
                "per_move_advantage_sum":float(per_move[active].sum()),
                "positive_token_coefficient_mass":float((advantages.clamp(min=0)*m).sum()),
                "negative_token_coefficient_mass":float((-advantages.clamp(max=0)*m).sum())}
        old = t.get("rollout_log_probs", t.get("old_log_probs"))
        ref = t.get("ref_log_prob")
        if old is not None and ref is not None:
            difference = (ref.float()-old.float()).clamp(-20,20)
            k3 = (torch.expm1(difference)-difference).clamp(min=0)
            result["reference_kl_token_mean"] = float((k3*mask).sum()/mask.sum().clamp(min=1))
            per = (k3*mask).sum(-1)/lengths.clamp(min=1)
            result["reference_kl_move_mean"] = float(per[lengths>0].mean())
        responses = t.get("responses")
        if responses is not None and think_end_id is not None:
            boundary = (responses == think_end_id) & mask
            # Reasoning includes the first closing marker; answer begins after it.
            before = boundary.cumsum(-1) == 0
            reasoning = mask & (before | (boundary & (boundary.cumsum(-1)==1)))
            for label, m in (("reasoning",reasoning),("answer",mask & ~reasoning)):
                result[label] = {"tokens":int(m.sum()),
                    "positive_token_coefficient_mass":float((advantages.clamp(min=0)*m).sum()),
                    "negative_token_coefficient_mass":float((-advantages.clamp(max=0)*m).sum())}
            result["missing_think_boundary"] = int((~boundary.any(-1) & (lengths>0)).sum())
            result["eos_tokens_in_loss"] = sum(int(((responses == tok)&mask).sum()) for tok in eos_ids)
        return result


def enforce_update_safety(quality, diagnostics, sync, config, output):
    step = quality["step"]
    stage = quality.get("stage", "before_update")
    if (not quality["valid"] or not math.isfinite(quality["valid_rate"])
            or quality["valid_rate"] < config["quality_gate_min_valid"]):
        safety_stop(output, "rollout_quality_gate", {"stage":stage,"step":step,"quality":quality,
            "minimum_valid_rate":config["quality_gate_min_valid"]})
    limit = config["reference_kl_stop"]
    value = diagnostics.get("reference_kl_move_mean")
    if limit and (value is None or not math.isfinite(value) or value > limit):
        safety_stop(output,"reference_kl_gate",{"stage":stage,"step":step,
            "reference_kl_move_mean":value,"limit":limit})
    if config["sync_max_mean_abs_logprob_delta"]:
        value = (sync or {}).get("mean_abs_logprob_delta")
        if value is None or not math.isfinite(value) or value > config["sync_max_mean_abs_logprob_delta"]:
            safety_stop(output,"actor_rollout_sync_gate",{"stage":stage,"step":step,"sync":sync})


def select_validated_checkpoint(output, actors, *, validation_identity=None):
    from .verl_ppo import complete_checkpoint
    output = Path(output)
    eligible = []
    baseline = read_json(output/"validation-baseline.json")
    expected = validation_identity or baseline["validation_identity"]
    for receipt in (output/"validation").glob("step-*.json"):
        row = read_json(receipt)
        step = row["step"]
        if (not row.get("passed") or row.get("stage") != "after_update" or step<1
                or row.get("validation_identity") != expected):
            continue
        checkpoint = complete_checkpoint(output/"checkpoints",actors,allowed_steps={step})
        if checkpoint is None or not (checkpoint/"games.json").is_file():
            continue
        eligible.append(((row["valid_rate"],row["score"],step),checkpoint,row))
    if not eligible:
        raise RuntimeError("No complete, post-update game-validated checkpoint eligible for export")
    _,checkpoint,row = max(eligible,key=lambda item:item[0])
    return checkpoint,row


def select_export_checkpoint(output, actors, *, internal_validation, experiment_digest):
    """Select a validated checkpoint, or an explicitly unvalidated completed update."""
    if internal_validation:
        return select_validated_checkpoint(output, actors)
    from .verl_ppo import complete_checkpoint
    output = Path(output)
    eligible = []
    for path in (output/'checkpoints').glob('global_step_*'):
        try:
            step = int(path.name.removeprefix('global_step_'))
        except ValueError:
            continue
        checkpoint = complete_checkpoint(output/'checkpoints',actors,
            allowed_steps={step},require_game_state=True)
        if checkpoint is None or step < 1:
            continue
        state = read_json(checkpoint/'games.json')
        update = state.get('update',{})
        metrics = update.get('metrics',{})
        if (state.get('identity') != experiment_digest or state.get('step') != step
                or state.get('internal_validation') is not False
                or state.get('validation_baseline') is not None
                or update.get('stage') != 'update_completed' or update.get('step') != step
                or metrics.get('training/global_step') != step
                or not all(math.isfinite(float(v)) for v in metrics.values() if isinstance(v,(int,float)))):
            continue
        eligible.append((step,checkpoint))
    if not eligible:
        raise RuntimeError('No complete checkpoint with a finite optimizer update eligible for export')
    step,checkpoint = max(eligible,key=lambda item:item[0])
    return checkpoint,{'step':step,'enabled':False,'stage':'disabled','passed':None,
        'reason':'internal_validation_disabled','formal_evaluation_required':True}
