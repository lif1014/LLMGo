"""Regression coverage for configurable caps and bounded whole-group retries."""
import ast
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from llmgo.training.chess_r1_config import overrides, token_budgets, validate_recipe
from llmgo.training.chess_r1_resample import bounded_replacement_groups

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def recipe():
    return json.loads((ROOT/'configs/grpo/default.json').read_text())


@pytest.mark.parametrize('response', [512, 8192, 16384])
def test_launcher_and_native_verl_agree_on_response_budget(recipe, tmp_path, response):
    spec = importlib.util.spec_from_file_location('train_chess_r1', ROOT/'scripts/train_chess_r1.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    recipe.update(max_response_tokens=response, max_prompt_tokens=2048)
    launcher.validate_recipe(recipe)
    settings = {key: json.loads(value) for item in overrides(recipe, tmp_path)
                for key, value in [item.split('=', 1)]
                if key in {'data.max_prompt_length', 'data.max_response_length',
                           'actor_rollout_ref.rollout.max_model_len'}}
    assert settings == {'data.max_prompt_length': 2048, 'data.max_response_length': response,
                        'actor_rollout_ref.rollout.max_model_len': 2048+response}


@pytest.mark.parametrize('response', [0, 256, 16385, True, 8192.5, '8192'])
def test_invalid_budget_is_rejected_consistently(recipe, tmp_path, response):
    recipe['max_response_tokens'] = response
    with pytest.raises(ValueError, match='max_response_tokens'):
        validate_recipe(recipe)
    with pytest.raises(ValueError, match='max_response_tokens'):
        overrides(recipe, tmp_path)


def test_defaults_and_fixed_protocol_guards(recipe):
    recipe.pop('max_response_tokens')
    assert token_budgets(recipe) == (1024, 16384)
    for change in ({'enable_thinking': True}, {'harness': True}, {'skills': ['prompt']},
                   {'sft': True}, {'answer_protocol': 'changed'}, {'checkpoint_every': False}):
        with pytest.raises(ValueError):
            validate_recipe({**recipe, **change})


def test_configuration_guard_remains_active_with_python_optimization():
    program = """from llmgo.training.chess_r1_config import token_budgets
try:
    token_budgets({'max_response_tokens': 20000})
except ValueError:
    raise SystemExit(0)
raise SystemExit(1)
"""
    subprocess.run([sys.executable, '-O', '-c', program], check=True)


def test_refill_is_bounded_by_missing_groups_and_available_batch():
    assert bounded_replacement_groups(2, 128) == 10
    assert bounded_replacement_groups(126, 128) == 128
    assert bounded_replacement_groups(16, 4) == 4
    assert bounded_replacement_groups(0, 128) == 0
    for args in ((-1, 128), (2, 0), (2, 128, -1), (True, 128)):
        with pytest.raises(ValueError):
            bounded_replacement_groups(*args)


class Mask:
    def __init__(self, lengths):
        self.lengths = np.array(lengths)

    def sum(self, _):
        return self.lengths

    def __getitem__(self, indices):
        return Mask(self.lengths[indices])


class Proto:
    """Only the DataProto transport surface used by the native callback."""
    def __init__(self, batch=None, non_tensor_batch=None, meta_info=None):
        self.batch = batch or {}
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}

    def __len__(self):
        return len(self.non_tensor_batch['uid'])

    def __getitem__(self, indices):
        return Proto({k: v[indices] for k, v in self.batch.items()},
                     {k: v[indices] for k, v in self.non_tensor_batch.items()}, self.meta_info.copy())

    def union(self, other):
        self.batch.update(other.batch)
        self.non_tensor_batch.update(other.non_tensor_batch)
        return self

    def repeat(self, count, interleave):
        assert interleave
        return Proto(non_tensor_batch={k: np.repeat(v, count) for k, v in self.non_tensor_batch.items()},
                     meta_info=self.meta_info.copy())

    @classmethod
    def from_single_dict(cls, values):
        return cls(non_tensor_batch=values.copy())

    @classmethod
    def concat(cls, groups):
        return cls({k: Mask(np.concatenate([p.batch[k].lengths for p in groups])) for k in groups[0].batch},
                   {k: np.concatenate([p.non_tensor_batch[k] for p in groups]) for k in groups[0].non_tensor_batch})


@pytest.mark.parametrize('cap', [512, 8192, 16384])
def test_native_callback_drops_entire_capped_groups_and_refills_only_a_small_batch(tmp_path, monkeypatch, cap):
    # Execute the actual callback without importing the CUDA/verl module. This
    # tests filtering, refill sizing and metadata alignment together.
    tree = ast.parse((ROOT/'src/llmgo/training/chess_r1_verl.py').read_text())
    callback = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    and node.name == 'complete_groups')
    fake_verl = types.ModuleType('verl')
    fake_verl.DataProto = Proto
    monkeypatch.setitem(sys.modules, 'verl', fake_verl)
    primary = Proto(non_tensor_batch={'uid': np.array([f'original-{n}' for n in range(128)], dtype=object)})
    loader = iter(itertools.repeat({'uid': primary.non_tensor_batch['uid'].copy()}))
    class Loader:
        source_epoch = 0
        def __next__(self):
            return next(loader)
    trainer = types.SimpleNamespace(_training_batch=primary, global_steps=4, chess_metrics={},
        train_dataloader=Loader(), config=types.SimpleNamespace(data=types.SimpleNamespace(max_response_length=cap),
            actor_rollout_ref=types.SimpleNamespace(rollout=types.SimpleNamespace(n=8))))
    generated_sizes, records = [], []
    def generate(inputs):
        generated_sizes.append(len(inputs))
        lengths = np.full(len(inputs), 64)
        if len(generated_sizes) == 1:
            lengths[0], lengths[8] = cap, cap+1
        return Proto({'response_mask': Mask(lengths), 'attention_mask': Mask(lengths+100)})
    namespace = dict(self=trainer, np=np, uuid=__import__('uuid'), original_generate=generate,
        bounded_replacement_groups=bounded_replacement_groups,
        RayPPOTrainer=types.SimpleNamespace(_get_gen_batch=lambda _, batch: batch),
        root=lambda: tmp_path, atomic=lambda path, row: records.append(row), progress=lambda *a, **k: None)
    exec(compile(ast.Module(body=[callback], type_ignores=[]), '<native-complete-groups>', 'exec'), namespace)
    retained = namespace['complete_groups'](primary.repeat(8, interleave=True))
    assert generated_sizes == [1024, 80]  # Two missing groups plus eight spare.
    assert len(retained) == 1024
    uids = retained.non_tensor_batch['uid']
    assert 'original-0' not in uids and 'original-1' not in uids
    assert len(set(uids)) == 128 and all(list(uids).count(uid) == 8 for uid in set(uids))
    assert max(retained.batch['response_mask'].sum(-1)) < cap
    assert records[0]['capped_responses'] == 2 and records[0]['pending_responses'] == 16
    assert records[1]['pending_responses'] == 0
    assert np.array_equal(primary.non_tensor_batch['uid'], uids[::8])
