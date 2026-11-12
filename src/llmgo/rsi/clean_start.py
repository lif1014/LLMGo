"""Build a new research source without old sessions, results, recipes or Git history."""
import shutil
from pathlib import Path

from .contracts import atomic_json,digest,file_hash,read_json

DOCS=('lichess-data.md','training-reviews.md','meta-execution.md')


def prepare(repo,target,site_path):
    repo,target=Path(repo),Path(target)
    if target.exists():raise FileExistsError('Clean start already exists; never overwrite a study')
    target.mkdir(parents=True)
    for name in ('src','scripts','tests','.agents'):
        shutil.copytree(repo/name,target/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for path in repo.glob('requirements*.txt'):shutil.copy2(path,target/path.name)
    (target/'configs/grpo').mkdir(parents=True);shutil.copy2(repo/'configs/grpo/default.json',target/'configs/grpo/default.json')
    site=read_json(site_path);site['repo']=str(target)
    atomic_json(target/'configs/rsi/site.qwen-meta.json',site)
    # Launcher gating in the shared Codex wrapper recognizes the same workspace roots.
    atomic_json(target/'configs/rsi/site.beijing.json',site)
    docs=target/'docs';docs.mkdir()
    for name in DOCS:shutil.copy2(repo/'docs'/name,docs/name)
    scope='''# Phase 1 research contract

Start from assets/models/Qwen3.5-4B. This is a new study with no imported findings,
trained checkpoints, Memory, sessions, historical scores or teacher answers.
Use RL only; no SFT. You may change RL/reward/hyperparameters/data selection.
Student uses the fixed Chess-R1 prompt and non-thinking; harness absent, skills.json=[].
Meta uses Qwen3.5-4B thinking and Codex tools in an on-demand one-GPU task.
Read all moves of losing games before forming the next hypothesis.
Default training data is boxed-nonthinking-v2; the verified complete Lichess training pool is optional.
Deliver selection code/query, selected IDs, category statistics and a failure-grounded hypothesis.
Every5 GRPO steps save the full optimizer/RNG/dataloader checkpoint, export and validate L2/L3
40 games each. Decide continue/replan/finish only after the complete results.
Referee:300 plies, fivefold repetition and current standard draw rules; no inference request deadline.
Final DeepSeek non-thinking games remain outside development selection.
Training submits HIGH/model-autosft with one HIGH slot. The inference card is only for Meta;
Meta tasks use NORMAL/model-autosft or the operator-managed PREEMPTABLE/model-new fallback.
All artifacts stay in this study's campaign/workspace/run paths. Do not read other studies.
'''
    for name in ('phase1-rl-scope.md','grpo-pipeline.md','rsi-operations.md'):(docs/name).write_text(scope)
    (target/'README.md').write_text(scope)
    (target/'assets').symlink_to(repo/'assets',target_is_directory=True)
    # Keep artifact locations compatible with the existing dashboard and Executor.
    # This is routing, not a read-isolation boundary.
    (target/'var').symlink_to(repo/'var',target_is_directory=True)
    manifest={str(p.relative_to(target)):file_hash(p) for p in target.rglob('*') if p.is_file() and not p.is_relative_to(target/'assets') and not p.is_relative_to(target/'var')}
    atomic_json(target/'clean-start.json',{'schema_version':1,'source_files':manifest,'source_sha256':digest(manifest),
        'excluded':['Git history','var/runs','campaign Memory','Codex sessions','historical benchmark configs','operator research conclusions'],
        'isolation':'Clean input/source only; same-account shared-filesystem read access is not sandboxed'})
    return target/'configs/rsi/site.qwen-meta.json'
