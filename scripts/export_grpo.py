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
    args = parser.parse_args()
    output = args.output.resolve()
    recipe = json.loads((output/'recipe.json').read_text())
    result = json.loads((output/'result.json').read_text())
    if result.get('status') != 'completed' or result.get('steps') != recipe['steps']:
        raise RuntimeError('Training has no complete final result')
    target = output/'model'
    submission = output/'submission.json'
    if submission.exists():
        old = json.loads(submission.read_text())
        if old.get('model_path') != str(target) or model_identity(target)['sha256'] != old['model_identity']['sha256']:
            raise RuntimeError('Existing export identity changed')
        return
    receipt_path = output/'checkpoints'/f"receipt-{recipe['steps']:08d}.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt['step'] != recipe['steps'] or not receipt.get('native_verl'):
        raise RuntimeError('Missing native final checkpoint')
    checkpoint = Path(receipt['path'])
    for rel, size in receipt['file_sizes'].items():
        if (checkpoint/rel).stat().st_size != size:
            raise RuntimeError('Incomplete checkpoint shard: '+rel)
    if target.exists():
        raise RuntimeError('Unreceipted model export requires review: '+str(target))
    temp = output/f'model.partial-{time.time_ns()}'
    subprocess.run([sys.executable,'-m','verl.model_merger','merge','--backend','fsdp',
                    '--local_dir',str(checkpoint/'actor'),'--target_dir',str(temp),
                    '--use_cpu_initialization'],check=True)
    base = Path(recipe['model_path'])
    for file in base.iterdir():
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
