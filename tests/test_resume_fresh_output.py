"""The reviewed training supervisor creates status files before the first fit."""
import json
from pathlib import Path

import pytest

from llmgo.training.chess_r1_resume import FRESH_OUTPUT_FILES, resume_plan


def supervisor_output(tmp_path: Path) -> Path:
    """Reproduce the state left by rsi.training_loop before chess_r1_gpu_entry runs."""
    output = tmp_path / 'grpo'
    (output / 'logs').mkdir(parents=True)
    (output / 'run.lock').write_text('')
    (output / 'recipe.json').write_text(json.dumps({'steps': 150, 'run_id': 'fixture'}))
    (output / 'progress.json').write_text(json.dumps({'stage': 'training', 'step': 5}))
    return output


def test_supervisor_status_files_allow_a_fresh_experiment(tmp_path):
    output = supervisor_output(tmp_path)
    assert {'run.lock', 'recipe.json', 'logs', 'progress.json'} <= FRESH_OUTPUT_FILES
    assert resume_plan(output, {'steps': 150}, 'release') == {'step': 0, 'checkpoint': None, 'attempt': 0}


@pytest.mark.parametrize('relative', ['updates/step-00000005.json', 'native-checkpoints/global_step_5/data.pt',
                                      'oracle.log', 'result.json'])
def test_uncommitted_training_artifacts_still_block_a_restart(tmp_path, relative):
    output = supervisor_output(tmp_path)
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{}')
    with pytest.raises(ValueError, match='no experiment identity'):
        resume_plan(output, {'steps': 150}, 'release')


def test_identity_is_required_before_trusting_receipts(tmp_path):
    output = supervisor_output(tmp_path)
    (output / 'checkpoints').mkdir()
    (output / 'checkpoints' / 'receipt-00000005.json').write_text('{}')
    with pytest.raises(ValueError, match='no experiment identity'):
        resume_plan(output, {'steps': 150}, 'release')
