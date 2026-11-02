"""Resume only this experiment's committed, complete native verl checkpoints."""
import json
from pathlib import Path


def checkpoint_manifest(folder, gpus):
    folder = Path(folder)
    files = [folder/'data.pt', folder/'actor/huggingface/config.json']
    files += [folder/'actor'/f'{kind}_world_size_{gpus}_rank_{rank}.pt'
              for kind in ('model', 'optim', 'extra_state') for rank in range(gpus)]
    result = {}
    for path in files:
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError('Incomplete native checkpoint: '+str(path))
        result[str(path.relative_to(folder))] = path.stat().st_size
    return result


def latest_complete_checkpoint(output, gpus):
    output = Path(output)
    valid = []
    for path in (output/'checkpoints').glob('receipt-*.json'):
        try:
            receipt = json.loads(path.read_text())
            step = int(receipt['step'])
            folder = output/'native-checkpoints'/f'global_step_{step}'
            if not receipt['native_verl'] or not receipt['resume_contains_optimizer_rng_and_data']:
                continue
            if Path(receipt['path']) != folder or step <= 0:
                continue
            manifest = checkpoint_manifest(folder, gpus)
            if manifest != receipt.get('file_sizes'):
                continue
            valid.append((step, folder))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return max(valid, default=(0, None), key=lambda item: item[0])


def resume_plan(output, recipe, source_release):
    """Caller holds a run lock. Never load weights from an uncommitted save."""
    output = Path(output)
    identity = output/'identity.json'
    if not identity.exists():
        if any(p.name not in {'run.lock', 'recipe.json', 'logs'} for p in output.iterdir()):
            raise ValueError('Existing output has no experiment identity')
        return {'step': 0, 'checkpoint': None, 'attempt': 0}
    if not recipe.get('resume_on_restart'):
        raise ValueError('This recipe does not allow reusing an output directory')
    original = json.loads(identity.read_text())
    if original['recipe'] != recipe or original['source_release'] != str(source_release):
        raise ValueError('Resume requires the same frozen recipe and source release')
    step, checkpoint = latest_complete_checkpoint(output, int(recipe.get('allocated_gpus', 8)))
    attempts = output/'attempts'
    number = 1
    while (attempts/f'restart-{number:04d}').exists():
        number += 1
    archive = attempts/f'restart-{number:04d}'
    archive.mkdir(parents=True)
    for relative in ('failure.json', 'progress.json', 'latest.json', 'oracle/ready.json',
                     'worker-compatibility'):
        path = output/relative
        if path.exists():
            target = archive/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)
    # Keep abandoned work for inspection; native fit may safely recreate these
    # filenames when replaying steps after the last durable checkpoint.
    for relative in ('updates', 'tables', 'sampling', 'native-checkpoints'):
        for path in (output/relative).glob('*'):
            prefix = 'global_step_' if relative == 'native-checkpoints' else 'step-'
            if not path.name.startswith(prefix):
                continue
            try:
                saved_step = int(path.name[len(prefix):].split('-')[0].split('.')[0])
            except ValueError:
                continue
            if saved_step > step:
                target = archive/relative/path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                path.rename(target)
    return {'step': step, 'checkpoint': str(checkpoint) if checkpoint else None,
            'attempt': number, 'archive': str(archive)}
