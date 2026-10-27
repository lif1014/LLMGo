import json
import pytest

from llmgo.training.validation import minimum_valid_rate, validation_identity
from llmgo.training.verl_ppo import training_config
from llmgo.training.stability import enforce_update_safety, TrainingSafetyStop


def test_original_policy_does_not_need_ninety_percent_before_learning(tmp_path):
    cfg=training_config({})
    rate=35/39
    quality={'step':0,'stage':'validation','valid':35,'valid_rate':rate}
    enforce_update_safety(quality,{'reference_kl_move_mean':.00048},
        {'mean_abs_logprob_delta':.0132},cfg,tmp_path)
    assert rate >= minimum_valid_rate(cfg)
    floor=minimum_valid_rate(cfg,{'valid_rate':rate})
    assert floor == pytest.approx(rate-.1)
    assert .79 < floor < .8
    assert minimum_valid_rate(cfg,{'valid_rate':.55}) == .5


def test_baseline_identity_tracks_sampling_seed_and_experiment():
    cfg=training_config({})
    a=validation_identity(cfg,'original')
    assert a != validation_identity({**cfg,'validation_seed':99},'original')
    assert a != validation_identity(cfg,'different-model')


def test_validation_failure_preserves_actual_stage(tmp_path):
    with pytest.raises(TrainingSafetyStop):
        enforce_update_safety({'step':0,'stage':'validation','valid':0,'valid_rate':0},
            {},None,training_config({}),tmp_path)
    row=json.loads((tmp_path/'stability-stop.json').read_text())
    assert row['evidence']['stage']=='validation'
