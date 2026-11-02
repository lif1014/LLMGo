#!/usr/bin/env python3
"""Run a frozen Chess-R1 GRPO recipe inside an already submitted GPU job."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('LLMGO_ASSET_ROOT', '/user/lyh4170/LLMGo!'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('recipe', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python', default='/user/lyh4170/venvs/llmgo-verl-ppo-tf54-20261024/bin/python')
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    output = args.output.resolve()
    if not output.is_relative_to((ROOT/'var/runs').resolve()):
        raise ValueError('Experiment output must be under LLMGo!/var/runs')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', output.name):
        raise ValueError('Invalid experiment ID')
    if recipe.get('framework') != 'verl' or recipe.get('algorithm') != 'grpo':
        raise ValueError('Expected a frozen GRPO recipe')
    output.mkdir(parents=True, exist_ok=True)
    (output/'logs').mkdir(exist_ok=True)
    cache = ROOT/'var/runtime'/(output.parent.name if output.name == 'grpo' else output.name)
    for name in ('flashinfer', 'inductor', 'torch-extensions', 'vllm'):
        (cache/name).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(SOURCE/'src'), TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='offline',
               MAX_JOBS='4', FLASHINFER_NVCC_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='2',
               VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS='1800',
               LLMGO_RUNTIME_ROOT=str(ROOT/'var/r'),
               FLASHINFER_WORKSPACE_BASE=str(cache/'flashinfer'),
               TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
               TORCH_EXTENSIONS_DIR=str(cache/'torch-extensions'),
               VLLM_CACHE_ROOT=str(cache/'vllm'))
    with (output/'logs/train.log').open('a') as log:
        completed = subprocess.run([args.python, '-B', str(SOURCE/'scripts/train_chess_r1.py'),
                                    '--recipe', str(args.recipe), '--output', str(output)],
                                   cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT)
    return completed.returncode


if __name__ == '__main__':
    sys.exit(main())
