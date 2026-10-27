"""CPU contracts for round-4 rollout quality monitoring, the stop-loss gate and sync evidence."""
import json
from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")

from llmgo.training.quality import (RolloutQualityError, enforce_quality_gate,
                                   generation_throughput, rollout_quality, sync_check)


class Trainer:
    def __init__(self, grade=None):
        self.grade = grade
        self.global_steps = 3
        self.calls = 0

    def _compute_old_log_prob(self, batch):
        self.calls += 1
        if isinstance(self.grade, Exception):
            raise self.grade
        tensors = {"old_log_probs": batch.batch["rollout_log_probs"]+self.grade}
        return SimpleNamespace(batch=tensors), 0.0


def batch(rows, response_length=4, reasons=(None,), scores=None):
    mask = torch.zeros(len(rows), response_length)
    for index, length in enumerate(rows):
        mask[index, :length] = 1
    if scores is None:
        scores = [-1. if reason else .5 for reason in reasons]
    return SimpleNamespace(batch={"response_mask":mask,
            "token_level_scores":torch.tensor([[score]+[0.]*(response_length-1) for score in scores]),
            "advantages":torch.tensor([[score]*response_length for score in scores]),
            "rollout_log_probs":torch.zeros(len(rows), response_length)},
        non_tensor_batch={"failure_reason":list(reasons),
            "value_before":[.25]*len(rows)})


def test_quality_uses_the_reward_managers_own_verdicts():
    rows = batch([4, 2, 4, 4], reasons=("", "", "length", "invalid_final_answer"),
        scores=[.5, .5, -1., -1.])
    quality = rollout_quality(rows, max_response_length=4)
    assert quality["samples"] == 4 and quality["valid"] == 2
    assert quality["valid_rate"] == .5
    assert quality["failure_counts"] == {"ok":2, "length":1, "invalid_final_answer":1}
    # The reported mean is token-weighted (what the policy gradient actually sees).
    assert quality["advantage_mean"] == pytest.approx(-5/14)
    assert quality["advantage_min"] == -1. and quality["advantage_max"] == .5
    assert quality["advantage_negative_fraction"] == .5
    assert quality["value_before_mean"] == .25
    assert quality["response_tokens_mean"] == pytest.approx(3.5)
    assert quality["response_tokens_total"] == 14
    assert quality["at_token_cap_fraction"] == .75


def test_quality_reports_unavailable_instead_of_zero_when_reasons_are_missing():
    rows = batch([4, 4])
    rows.non_tensor_batch.pop("failure_reason")
    quality = rollout_quality(rows, max_response_length=4)
    assert quality["valid"] == 0 and quality["failure_counts"] == {"unavailable":2}




def test_throughput_is_taken_from_the_measured_generation_timer():
    assert generation_throughput({"timing_per_token_ms/gen":2.}) == pytest.approx(500.)
    assert generation_throughput({"timing_s/gen":60., "response_length/mean":3000.}) == pytest.approx(50.)
    assert generation_throughput({}) is None


def test_sync_check_records_agreement_and_never_raises(tmp_path):
    rows = batch([4, 4])
    trainer = Trainer(.01)
    row = sync_check(trainer, rows, samples=16, output=tmp_path)
    assert row["tokens"] == 8 and row["mean_abs_logprob_delta"] == pytest.approx(.01)
    assert row["mean_abs_logprob_delta"] > 0
    assert json.loads((tmp_path/"sync-check-latest.json").read_text())["step"] == 3
    broken = sync_check(Trainer(RuntimeError("no worker")), rows, samples=16, output=tmp_path)
    assert "no worker" in broken["error"] and "mean_abs_logprob_delta" not in broken
    assert len((tmp_path/"sync-check.jsonl").read_text().splitlines()) == 2


def test_gate_threshold_is_configurable_and_rejects_collapsed_rollouts():
    from llmgo.training.verl_ppo import training_config
    assert training_config({},8)["quality_gate_min_valid"] == .5
    assert training_config({"quality_gate_min_valid":.5},8)["quality_gate_min_valid"] == .5
    for value in (-.1, 1.1, "0.5"):
        with pytest.raises(ValueError):
            training_config({"quality_gate_min_valid":value},8)
    collapsed = rollout_quality(batch([4, 4], reasons=("length", "length")), max_response_length=4)
    assert collapsed["valid_rate"] < .5
    assert issubclass(RolloutQualityError, RuntimeError)


def test_gate_refuses_a_collapsed_update_and_records_the_refusal(tmp_path):
    quality = dict(rollout_quality(batch([4, 4], reasons=("length", "length")), max_response_length=4),
                   step=2)
    with pytest.raises(RolloutQualityError, match="refusing to apply the update"):
        enforce_quality_gate(quality, .5, output=tmp_path)
    record = json.loads((tmp_path/"quality-gate.json").read_text())
    assert record["reason"] == "rollout_quality_gate"
    assert record["quality_gate_min_valid"] == .5
    assert record["quality"]["failure_counts"] == {"length":2}
    assert not (tmp_path/"metrics.jsonl").exists()


def test_gate_accepts_healthy_rollouts_without_writing_anything(tmp_path):
    quality = dict(rollout_quality(batch([4]*5, reasons=("",)*(4)+("length",)), max_response_length=4),
                   step=1)
    assert quality["valid_rate"] == .8
    assert enforce_quality_gate(quality, .5, output=tmp_path) is quality
    assert list(tmp_path.iterdir()) == []








def test_sync_check_compares_a_sample_subset_without_shape_mismatch(tmp_path):
    """Round-5 regression: j000015/j000016 sliced the mask but not the log-prob tensors.

    Both jobs asked for 16 consistency samples out of 28/56 rollouts, so all eight sync
    checks died with IndexError and the post-update actor/rollout agreement - the whole
    point of running the control - was never measured.
    """
    rows = batch([4, 4, 4, 4, 4])
    row = sync_check(Trainer(.02), rows, samples=2, output=tmp_path)
    assert "error" not in row
    assert row["samples"] == 2 and row["tokens"] == 8
    assert row["mean_abs_logprob_delta"] == pytest.approx(.02)
    # Trainer grades the actor's log-probs upward by .02 against the rollout's zeros.
    assert row["old_logprob_mean"] == pytest.approx(row["rollout_logprob_mean"] + .02)


def test_export_preflight_records_a_skipped_pinned_check_for_a_five_x_export(tmp_path):
    """A 5.x-only tokenizer class must be reported, never treated as verified."""
    import importlib.util
    from pathlib import Path as PathlibPath
    from llmgo.training.verl_ppo import _static_tokenizer_ok
    export = tmp_path/"export"; export.mkdir()
    (export/"tokenizer_config.json").write_text(json.dumps({"tokenizer_class":"TokenizersBackend"}))
    assert _static_tokenizer_ok(export) is False
    (export/"tokenizer_config.json").write_text(json.dumps({"tokenizer_class":"Qwen2Tokenizer"}))
    assert _static_tokenizer_ok(export) is True
