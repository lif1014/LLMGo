#!/usr/bin/env python3
"""Freeze one Chess-R1 recipe, train in verl, and publish a complete model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('LLMGO_ASSET_ROOT','/user/lyh4170/LLMGo!'))
VERL_PYTHON = '/user/lyh4170/venvs/llmgo-verl-ppo-tf54-20261024/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(os.environ['LLMGO_OUTPUT']) if os.environ.get('LLMGO_OUTPUT') else None)
    parser.add_argument('--model', default=os.environ.get('LLMGO_MODEL'))
    parser.add_argument('--template', type=Path, default=SOURCE/'configs/grpo/default.json')
    args = parser.parse_args()
    if args.output is None:
        raise ValueError('Pass --output or LLMGO_OUTPUT')
    output = args.output.resolve()
    if not output.is_relative_to((ROOT/'var/runs').resolve()):
        raise ValueError('GRPO runs belong under project var/runs')
    output.mkdir(parents=True,exist_ok=True)
    training_output = output/'grpo'
    training_output.mkdir(exist_ok=True)
    recipe_path = training_output/'recipe.json'
    recipe = json.loads(args.template.read_text())
    if args.model:
        recipe['model_path'] = str(Path(args.model).resolve())
    recipe['run_id'] = output.name
    recipe['run_directory'] = output.name
    recipe['attempt'] = output.name
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None and visible.strip() not in ('','-1'):
        recipe['allocated_gpus'] = len([x for x in visible.split(',') if x.strip()])
    if recipe_path.exists():
        if json.loads(recipe_path.read_text()) != recipe:
            raise RuntimeError('Frozen GRPO recipe differs from this run')
    else:
        temp = recipe_path.with_suffix('.tmp')
        temp.write_text(json.dumps(recipe,ensure_ascii=False,indent=2)+'\n')
        temp.replace(recipe_path)
    env = dict(os.environ,LLMGO_ASSET_ROOT=str(ROOT))
    subprocess.run([sys.executable,'-B',str(SOURCE/'scripts/chess_r1_gpu_entry.py'),
                    str(recipe_path),'--output',str(training_output)],env=env,check=True)
    (output/'logs').mkdir(exist_ok=True)
    with (output/'logs/export.log').open('a') as log:
        subprocess.run([VERL_PYTHON,'-B',str(SOURCE/'scripts/export_grpo.py'),
                        '--output',str(training_output)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    submission = json.loads((training_output/'submission.json').read_text())
    target = output/'submission.json'
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(submission,ensure_ascii=False,indent=2)+'\n')
    temp.replace(target)


if __name__ == '__main__':
    main()
