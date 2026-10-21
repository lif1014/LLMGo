"""Real CPU PPO optimizer/checkpoint checks; not a Qwen GPU smoke test."""
import copy
import math
import random
from types import SimpleNamespace
import pytest

torch = pytest.importorskip("torch")

from llmgo.training.ppo import (policy_loss, update_batch, _completion_logps,
    save_checkpoint, restore_checkpoint, training_config)


@pytest.mark.parametrize("advantage,ratio,clipped", [(1., 1.5, True), (-1., .5, True),
    (1., .5, False), (-1., 1.5, False)])
def test_ppo_clip_direction_and_fixed_behavior(advantage, ratio, clipped):
    old = torch.tensor([-2.], requires_grad=True)
    ref = torch.tensor([-2.], requires_grad=True)
    logp = torch.tensor([-2.+math.log(ratio)], requires_grad=True)
    loss, metrics = policy_loss(logp, old, ref, advantage, 0., .2)
    loss.backward()
    assert old.grad is ref.grad is None
    assert metrics["clip_fraction"] == 1
    if clipped:
        assert logp.grad.item() == 0
    else:
        assert logp.grad.item()*advantage < 0


def test_equal_behavior_still_has_policy_gradient_and_reference_kl():
    for advantage in (1., -1.):
        logp = torch.tensor([-.7], requires_grad=True)
        loss, info = policy_loss(logp, logp.detach().clone(), logp.detach().clone(), advantage)
        loss.backward()
        assert info["kl"] == pytest.approx(0)
        assert logp.grad.item()*advantage < 0
    logp = torch.tensor([-.2], requires_grad=True)
    loss, info = policy_loss(logp, torch.tensor([-1.2]), torch.tensor([-1.2]), 0., 1.)
    loss.backward()
    assert info["kl"] > 0 and logp.grad.item() > 0


class TinyPolicy(torch.nn.Module):
    """Bigram language policy, testing the real trainer math without HF downloads."""
    def __init__(self):
        super().__init__()
        self.transition = torch.nn.Parameter(torch.zeros(4, 4))

    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=0):
        logits = self.transition[input_ids]
        return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path/"weights.pt")


def batch_for(model):
    rows = []
    for token, advantage in ((1, 1.), (2, -1.)):
        prompt, response = torch.tensor([[0]]), torch.tensor([token, 3])
        old = _completion_logps(model, prompt, response).detach().tolist()
        rows.append({"prompt_token_ids": [0], "generated_token_ids": [token, 3],
                     "behavior_logprobs": old, "reference_logprobs": old[:], "advantage": advantage})
    return rows


def test_real_multi_epoch_ppo_update_saves_and_restores_identically(tmp_path):
    random.seed(4170)
    torch.manual_seed(4170)
    cfg = training_config({"batch_size": 2, "minibatch_size": 1, "ppo_epochs": 4, "kl_beta": 0.})
    model = TinyPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1, weight_decay=0)
    batch = batch_for(model)
    frozen = copy.deepcopy(batch)
    initial = model.transition.detach().clone()
    metrics = update_batch(model, optimizer, batch, cfg)
    assert len(metrics) == 8 and batch == frozen
    assert not torch.equal(initial, model.transition)
    assert model.transition[0, 1] > model.transition[0, 2]
    assert any(row["clip_fraction"] > 0 for row in metrics)
    identity = {"backend": "hf_ppo", "critic": "frozen-test"}
    checkpoint = save_checkpoint(model, optimizer, tmp_path, 1, len(metrics), identity)
    expected_random = random.random(), torch.rand(3)
    next_batch = batch_for(model)
    expected_metrics = update_batch(model, optimizer, next_batch, cfg)
    expected_weights = model.transition.detach().clone()
    resumed = TinyPolicy()
    resumed_optim = torch.optim.AdamW(resumed.parameters(), lr=.1, weight_decay=0)
    state = restore_checkpoint(resumed, resumed_optim, checkpoint, identity,
        load_adapter=lambda model, path: model.load_state_dict(torch.load(path/"weights.pt", weights_only=True)))
    assert state == (1, 8)
    assert random.random() == expected_random[0]
    assert torch.equal(torch.rand(3), expected_random[1])
    actual_metrics = update_batch(resumed, resumed_optim, next_batch, cfg)
    assert actual_metrics == expected_metrics
    assert torch.equal(resumed.transition, expected_weights)
    # Modified checkpoint data and experiment identity cannot silently resume.
    with pytest.raises(ValueError, match="identity"):
        restore_checkpoint(resumed, resumed_optim, checkpoint, {"different": True})
    with (checkpoint/"weights.pt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        restore_checkpoint(resumed, resumed_optim, checkpoint, identity)


def test_response_logprobs_exclude_prompt_and_align_first_generated_token():
    model = TinyPolicy()
    with torch.no_grad():
        model.transition[0] = torch.tensor([0., 1., 2., 3.])
        model.transition[1] = torch.tensor([3., 2., 1., 0.])
    actual = _completion_logps(model, torch.tensor([[2, 0]]), torch.tensor([1, 3]))
    expected = torch.stack((model.transition[0].log_softmax(-1)[1], model.transition[1].log_softmax(-1)[3]))
    assert actual.shape == (2,)
    assert torch.allclose(actual, expected)


def test_nonfinite_ppo_loss_aborts_before_optimizer():
    with pytest.raises(RuntimeError, match="Nonfinite"):
        policy_loss(torch.tensor([float("inf")], requires_grad=True), torch.tensor([0.]), torch.tensor([0.]), 1.)


def test_behavior_recorder_captures_sampled_distribution_and_eos():
    from llmgo.training.ppo import BehaviorLogprobs
    recorder = BehaviorLogprobs()
    logits = [torch.tensor([[1., 2., 3.]]), torch.tensor([[3., 2., 1.]])]
    first = recorder(torch.tensor([[9, 9]]), logits[0])
    second = recorder(torch.tensor([[9, 9, 2]]), logits[1])
    assert first is logits[0] and second is logits[1]
    actual = recorder.finish(torch.tensor([2, 0]))
    expected = [float(logits[0].log_softmax(-1)[0,2]), float(logits[1].log_softmax(-1)[0,0])]
    assert actual == pytest.approx(expected)
    assert recorder.previous is None


def test_zero_advantage_batch_does_not_make_kl_only_update():
    model = TinyPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    batch = batch_for(model)
    for sample in batch:
        sample["advantage"] = 0.
        sample["reference_logprobs"] = [-9., -9.]
    before = model.transition.detach().clone()
    assert update_batch(model, optimizer, batch, training_config({})) == []
    assert torch.equal(before, model.transition)
