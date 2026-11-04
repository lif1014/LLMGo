#!/usr/bin/env python3
"""Export a completed native verl checkpoint as a verified HF model."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE/'src'))
from llmgo.rsi.contracts import atomic_json, model_identity
from llmgo.training.export_weights import normalize_export_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--step', type=int)
    parser.add_argument('--target', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    recipe = json.loads((output/'recipe.json').read_text())
    step = args.step if args.step is not None else recipe['steps']
    if args.step is None:
        result = json.loads((output/'result.json').read_text())
        if result.get('status') != 'completed' or result.get('steps') != step:
            raise RuntimeError('Training has no complete final result')
    if step <= 0 or step > recipe['steps']:
        raise ValueError('Invalid export step')
    target = args.target.resolve() if args.target else output/'model'
    if not target.is_relative_to(output.parent):
        raise ValueError('Exports must stay in this experiment')
    receipt_path = output/'checkpoints'/f"receipt-{step:08d}.json"
    submission = target.with_name(target.name+'.submission.json') if args.target else output/'submission.json'
    if submission.exists():
        old = json.loads(submission.read_text())
        if (old.get('model_path') != str(target) or old.get('checkpoint_receipt') != str(receipt_path)
                or model_identity(target)['sha256'] != old['model_identity']['sha256']):
            raise RuntimeError('Existing export identity changed')
        return
    receipt = json.loads(receipt_path.read_text())
    if receipt['step'] != step or not receipt.get('native_verl'):
        raise RuntimeError('Missing native final checkpoint')
    checkpoint = Path(receipt['path'])
    for rel, size in receipt['file_sizes'].items():
        if (checkpoint/rel).stat().st_size != size:
            raise RuntimeError('Incomplete checkpoint shard: '+rel)
    if target.exists():
        raise RuntimeError('Unreceipted model export requires review: '+str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name+f'.partial-{time.time_ns()}')
    subprocess.run([sys.executable,'-m','verl.model_merger','merge','--backend','fsdp',
                    '--local_dir',str(checkpoint/'actor'),'--target_dir',str(temp),
                    '--use_cpu_initialization'],check=True)
    base = Path(recipe['model_path'])
    for file in base.iterdir():
        # Shard indexes describe the merged weights, not the base checkpoint.
        # Copying one can hide a valid single shard or overwrite a new index.
        if file.name.endswith(('.safetensors.index.json', '.bin.index.json')):
            continue
        if file.name in {'export-receipt.json', 'repair-receipt.json'}:
            continue
        if file.is_file() and file.suffix in ('.json','.jinja','.txt','.model') and file.name!='config.json':
            shutil.copy2(file,temp/file.name)
    normalization = normalize_export_weights(temp,base)
    atomic_json(temp/'export-receipt.json',{'checkpoint':str(receipt_path),'normalization':normalization})
    os.replace(temp,target)
    identity = model_identity(target)
    atomic_json(submission,{'model_path':str(target),'model_identity':identity,
                            'checkpoint_receipt':str(receipt_path),'status':'ready'})


if __name__ == '__main__':
    main()
