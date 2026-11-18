"""Build a new research source without old sessions, results, recipes or Git history."""
import shutil
from pathlib import Path

from .contracts import atomic_json,digest,file_hash,read_json

DOCS=('lichess-data.md','training-reviews.md','meta-execution.md','qwen38-meta.md',
      'meta-research-operations.md','clean-grpo-pipeline.md','meta-workspace-agents.md')
POOL=Path('assets/datasets/lichess/20261103/boxed-nonthinking-v2')


def prepare_assets(repo,target,site):
    """Expose initial models, reward dependencies and the training-only pool."""
    for relative in ('assets/models/Qwen3.5-4B','assets/models/searchless-270M',
                     'assets/engines','assets/vendor/Chess-R1-830c0597'):
        link=target/relative;link.parent.mkdir(parents=True,exist_ok=True)
        link.symlink_to(repo/relative,target_is_directory=True)
    if site.get('meta_model_path'):
        meta=Path(site['meta_model_path'])
        link=target/'assets/models'/meta.name
        if not link.is_symlink():link.symlink_to(meta,target_is_directory=True)
    source,pool=repo/POOL,target/POOL
    pool.mkdir(parents=True)
    for name in ('train.parquet','index.parquet','stats.json','invalid.jsonl'):
        (pool/name).symlink_to(source/name)
    # Keep the verified file identities without publishing historical dataset paths.
    manifest=read_json(source/'manifest.json')
    proof=read_json(source/'verification.json')
    public={k:v for k,v in manifest.items() if k not in ('reference','heldouts')}
    public['source']={k:v for k,v in manifest.get('source',{}).items() if k!='path'}
    atomic_json(pool/'manifest.json',public)
    atomic_json(pool/'verification.json',{k:v for k,v in proof.items() if k!='heldouts'})


def prepare(repo,target,site_path):
    repo,target=Path(repo),Path(target)
    if target.exists():raise FileExistsError('Clean start already exists; never overwrite a study')
    target.mkdir(parents=True)
    for name in ('src','scripts','tests','.agents'):
        shutil.copytree(repo/name,target/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Pool building and the old opponent reference runner are operator tools.
    for name in ('prepare_lichess_pool.py','run_deepseek_loose.py'):
        (target/'scripts'/name).unlink(missing_ok=True)
    for path in repo.glob('requirements*.txt'):shutil.copy2(path,target/path.name)
    (target/'configs/grpo').mkdir(parents=True);shutil.copy2(repo/'configs/grpo/default.json',target/'configs/grpo/default.json')
    site=read_json(site_path);site['repo']=str(target)
    atomic_json(target/'configs/rsi/site.qwen-meta.json',site)
    atomic_json(target/'configs/rsi/site.qwen38-meta.json',site)
    # Launcher gating in the shared Codex wrapper recognizes the same workspace roots.
    atomic_json(target/'configs/rsi/site.beijing.json',site)
    docs=target/'docs';docs.mkdir()
    for name in DOCS:shutil.copy2(repo/'docs'/name,docs/name)
    scope='''# Phase 1 research contract

Start from assets/models/Qwen3.5-4B. This is a new study with no imported findings,
trained checkpoints, Memory, sessions, historical scores or teacher answers.
Use RL only; no SFT. You may change RL/reward/hyperparameters/data selection.
Student uses the fixed Chess-R1 prompt and non-thinking; harness absent, skills.json=[].
Meta uses the model in the site profile with thinking and Codex tools in an on-demand one-GPU task.
Read all moves of losing games before forming the next hypothesis.
The only supplied data source is assets/datasets/lichess/20261103/boxed-nonthinking-v2.
Inspect verification.json and index.parquet, then independently select training data.
No training subset is preselected: set your experiment recipe's train_file to your selected parquet.
The selector helper is optional; your own query/program is also allowed. Do not read other datasets.
Deliver selection code/query, selected IDs, category statistics and a failure-grounded hypothesis.
Every5 GRPO steps save the full optimizer/RNG/dataloader checkpoint, export and validate L2/L3
40 games each. Decide continue/replan/finish only after the complete results.
Referee:300 plies, fivefold repetition and current standard draw rules; no inference request deadline.
Final DeepSeek non-thinking games remain outside development selection.
Student training uses model-autosft/HIGH (one active or uncertain HIGH job).
Meta inference and standalone validation use model-autosft/NORMAL; Meta may also use
model-new, model-claw-tool or model-codeagent2/PREEMPTABLE. The inference card is only for Meta. The framework selects available cards; never submit Meta cards yourself.
All artifacts stay in this study's campaign/workspace/run paths. Do not read other studies.
'''
    (docs/'phase1-rl-scope.md').write_text(scope)
    shutil.copy2(repo/'docs/clean-grpo-pipeline.md',docs/'grpo-pipeline.md')
    shutil.copy2(repo/'docs/meta-research-operations.md',docs/'rsi-operations.md')
    (target/'README.md').write_text(scope)
    shutil.copy2(repo/'docs/meta-workspace-agents.md',target/'AGENTS.md')
    prepare_assets(repo,target,site)
    # Keep artifact locations compatible with the existing dashboard and Executor.
    # This is routing, not a read-isolation boundary.
    (target/'var').symlink_to(repo/'var',target_is_directory=True)
    manifest={str(p.relative_to(target)):file_hash(p) for p in target.rglob('*') if p.is_file() and not p.is_relative_to(target/'assets') and not p.is_relative_to(target/'var')}
    atomic_json(target/'clean-start.json',{'schema_version':1,'source_files':manifest,'source_sha256':digest(manifest),
        'data_policy':{'source_pool':str(POOL),'preselected_training_data':False,'selection_required':True},
        'excluded':['Git history','var/runs','campaign Memory','Codex sessions','historical benchmark configs','operator research conclusions','historical training subsets','held-out parquet files','raw unpartitioned puzzles'],
        'isolation':'Clean input/source only; same-account shared-filesystem read access is not sandboxed'})
    return target/'configs/rsi/site.qwen38-meta.json'
