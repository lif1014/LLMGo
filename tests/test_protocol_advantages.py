"""Regression for invalid answers rewarded after LC0 TD centering."""
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
from llmgo.training.advantages import apply_protocol_guard, guarded_advantages
from llmgo.training.verl_ppo import hydra_overrides, training_config


def batch(raw, reasons, lengths=None):
    lengths = lengths or [3]*len(raw)
    mask = torch.zeros((len(raw), max(lengths) or 1))
    reward = torch.zeros_like(mask)
    for i, n in enumerate(lengths):
        mask[i,:n] = 1
        if n: reward[i,n-1] = raw[i]
    return SimpleNamespace(batch={'response_mask':mask, 'token_level_rewards':reward},
                           non_tensor_batch={'failure_reason':reasons})


def test_real_failure_pattern_cannot_become_positive_after_centering(tmp_path):
    # Real step3 sample163: raw TD 0, old full-batch mean about -0.223.
    b = batch([0., -.5, -.2, .1], ['invalid_final_answer', '', '', ''])
    diag = apply_protocol_guard(b, tmp_path, 3)
    assert (b.batch['advantages'][0,:3] == -1).all()
    assert diag['invalid_positive_advantages'] == 0
    # Legal samples alone set the baseline. Original ordering remains intact.
    expected = torch.tensor([-.5, -.2, .1]) - torch.tensor([-.5, -.2, .1]).mean()
    assert torch.allclose(b.batch['advantages'][1:,0], expected)
    assert torch.allclose(b.batch['advantages'][1:,0].mean(), torch.tensor(0.), atol=1e-7)
    record = json.loads((tmp_path/'protocol-advantages/step-00000003.json').read_text())
    assert record['samples'][0]['raw_td'] == 0 and record['samples'][0]['final_advantage'] == -1


@pytest.mark.parametrize('failure', ['invalid_final_answer','length','incomplete_thinking'])
def test_failure_sign_is_independent_of_other_moves_and_response_length(failure):
    b = batch([0., -2., 1.], [failure, '', ''], [1000, 2, 1])
    a, _, d = guarded_advantages(b.batch['token_level_rewards'], b.batch['response_mask'],
                                  b.non_tensor_batch['failure_reason'], penalty=.5)
    assert (a[0,:1000] == -.5).all() and (a[1,2:] == 0).all()
    assert d['valid_baseline'] == -.5 and d['invalid_advantage_max'] == -.5


def test_all_failed_and_fully_masked_rows_are_finite_without_positive_failures():
    b = batch([0., -.1, 0.], ['invalid_final_answer','length',''], [3, 2, 0])
    a, ret, d = guarded_advantages(b.batch['token_level_rewards'], b.batch['response_mask'],
                                  b.non_tensor_batch['failure_reason'])
    assert torch.isfinite(a).all() and torch.equal(a, ret)
    assert (a[2] == 0).all() and d['valid'] == 0 and d['invalid'] == 2
    assert d['invalid_advantage_max'] == -1


def test_protocol_failures_do_not_shift_the_legal_move_baseline():
    x = batch([.2, -.3], ['', ''])
    y = batch([.2, -.3, -2.], ['', '', 'invalid_final_answer'])
    a, _, _ = guarded_advantages(x.batch['token_level_rewards'], x.batch['response_mask'], ['',''])
    b, _, _ = guarded_advantages(y.batch['token_level_rewards'], y.batch['response_mask'], ['','','invalid_final_answer'])
    assert torch.allclose(a, b[:2])


@pytest.mark.parametrize('reason', [None, 'transport_error'])
def test_missing_or_infrastructure_verdict_is_not_used_as_a_chess_reward(reason):
    b = batch([0.], [reason])
    with pytest.raises(ValueError):
        guarded_advantages(b.batch['token_level_rewards'], b.batch['response_mask'], [reason])


def test_default_config_wires_guard_at_the_full_batch_preupdate_boundary(tmp_path):
    cfg = training_config({}, 8)
    assert cfg['advantage_estimator'] == 'lc0_td0_protocol_guarded'
    argv = hydra_overrides('/model', '/data', tmp_path, cfg, 8, '/teacher')
    assert 'algorithm.adv_estimator="lc0_td0"' in argv
    assert '+research.advantage_estimator="lc0_td0_protocol_guarded"' in argv
    assert '+research.protocol_failure_penalty=1.0' in argv
    for bad in [0., -1., float('nan'), True]:
        with pytest.raises(ValueError): training_config({'protocol_failure_penalty':bad}, 8)
