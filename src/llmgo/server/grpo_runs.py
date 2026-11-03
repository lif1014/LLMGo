"""Bounded, read-only access to experiments stored in LLMGo!/var/runs."""
from __future__ import annotations

import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query

from llmgo.rsi.contracts import identifier, inside, read_json

router = APIRouter(prefix='/api/grpo')


def run_root():
    return Path(os.environ.get('LLMGO_RUN_ROOT', '/user/lyh4170/LLMGo!/var/runs')).resolve()


def run_dir(name):
    try:
        path = inside(run_root(), run_root()/identifier(name))
        if not path.is_dir():
            raise ValueError('Unknown run')
        return path
    except ValueError as error:
        raise HTTPException(404, str(error)) from error


def safe_artifact(root, relative):
    try:
        path = inside(root, root/relative)
        if (path.suffix not in {'.json', '.jsonl', '.log', '.txt', '.yaml'} or
                not path.is_file()):
            raise ValueError('Unreadable run artifact')
        return path
    except ValueError as error:
        raise HTTPException(404, str(error)) from error


@router.get('/runs')
def runs():
    root = run_root()
    if not root.is_dir():
        return []
    rows = []
    for directory in sorted(root.iterdir(), key=lambda item:item.stat().st_mtime, reverse=True)[:100]:
        if not directory.is_dir():
            continue
        try:
            identifier(directory.name)
        except ValueError:
            continue
        progress = directory/'grpo/progress.json'
        rows.append({'id':directory.name, 'updated':directory.stat().st_mtime,
                     'stage':read_json(progress).get('stage') if progress.is_file() else None,
                     'trained':(directory/'grpo/result.json').is_file(),
                     'exported':(directory/'submission.json').is_file(),
                     'final_tested':(directory/'final-test/result.json').is_file()})
    return rows


@router.get('/runs/{name}/files')
def files(name: str):
    root = run_dir(name)
    allowed = ['recipe.json','submission.json','progress.json','result.json','failure.json',
               'research.log','logs/export.log','grpo/recipe.json','grpo/progress.json',
               'grpo/result.json','grpo/failure.json','grpo/identity.json',
               'grpo/resolved-config.json','grpo/oracle.log','grpo/logs/train.log']
    allowed += ['final-test/protocol.json','final-test/progress.json','final-test/result.json','final-test/vllm.log']
    allowed += ['scaling/curve.json','scaling/best.json','scaling/cost.json',
                'scaling/pending-review.json','scaling/final.json','scaling/identity.json']
    allowed += [p.relative_to(root).as_posix() for pattern in
                ('grpo/updates/*.json','grpo/checkpoints/receipt-*.json','grpo/sampling/*.json',
                 'grpo/tables/*.json','evaluations/*/summary.json','evaluations/*/progress.json',
                 'final-test/game_results/*.json','final-test/games/*/summary.json',
                 'final-test/games/*/events.jsonl','logs/*.log',
                 'scaling/reviews/*/*.json','scaling/evaluations/*/*/metrics.json',
                 'scaling/evaluations/*/*/games/*/events.jsonl',
                 'scaling/evaluations/*/*/games/*/summary.json')
                for p in sorted(root.glob(pattern))[-30:]]
    result = []
    for relative in dict.fromkeys(allowed):
        path = root/relative
        if path.is_file():
            result.append({'path':relative,'bytes':path.stat().st_size,
                           'modified':path.stat().st_mtime})
    return result


@router.get('/runs/{name}/artifact')
def artifact(name: str, path: str, offset: int = Query(0, ge=0),
             length: int = Query(131072, ge=1, le=524288)):
    target = safe_artifact(run_dir(name),path)
    with target.open('rb') as stream:
        stream.seek(offset)
        content = stream.read(length)
        return {'text':content.decode('utf-8','replace'),'offset':offset,
                'next_offset':stream.tell(),'eof':stream.tell()>=target.stat().st_size}
