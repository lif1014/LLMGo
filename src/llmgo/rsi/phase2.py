"""Phase 2 instructions, prompt-candidate history and delivery metadata.

Training restrictions are instructions to the Meta-Agent, not admission rules.
"""
from pathlib import Path
import hashlib
import shutil
import sys

from .contracts import REPO, atomic_json, digest, file_hash, read_json, utc

SCOPE = 'phase2_prompt_only'


def prepare_source(repo, target, site_path):
    """Prepare independent Phase 2 input, sharing immutable assets and artifact routing."""
    repo, target = Path(repo), Path(target)
    if target.exists():
        raise FileExistsError('Phase 2 source already exists; keep its study intact')
    target.mkdir(parents=True)
    for name in ('src', 'scripts', 'tests', 'examples/phase2'):
        shutil.copytree(repo/name, target/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('rsi-phase2', 'rsi-memory', 'rsi-executor', 'rsi-meta-execution'):
        shutil.copytree(repo/'.agents/skills'/name, target/'.agents/skills'/name)
    for path in repo.glob('requirements*.txt'):
        shutil.copy2(path, target/path.name)
    docs = target/'docs'; docs.mkdir()
    for name in ('phase2-meta-prompt.md', 'phase2-prompt-research.md', 'meta-execution.md'):
        shutil.copy2(repo/'docs'/name, docs/name)
    # Commands and onboarding stay Phase 2-specific; do not import Phase 1 RL guidance.
    shutil.copy2(repo/'docs/phase2-prompt-research.md', docs/'rsi-operations.md')
    shutil.copy2(repo/'docs/phase2-meta-prompt.md', target/'AGENTS.md')
    shutil.copy2(repo/'docs/phase2-prompt-research.md', target/'README.md')
    (target/'configs/grpo').mkdir(parents=True)
    shutil.copy2(repo/'configs/grpo/default.json', target/'configs/grpo/default.json')
    site = read_json(site_path)
    site.update(repo=str(target), research_scope=SCOPE)
    launcher = target/'scripts/codex-phase2'
    launcher.write_text('#!'+site.get('control_python', sys.executable)+'\n'
        'import sys\nfrom pathlib import Path\n'
        'sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))\n'
        'from llmgo.rsi.codex_bridge import launch\n'
        'raise SystemExit(launch(sys.argv[1:]))\n')
    launcher.chmod(0o755)
    site['codex_bin'] = str(launcher)
    atomic_json(target/'configs/rsi/site.phase2-deepseek.json', site)
    atomic_json(target/'configs/rsi/site.beijing.json', site)
    (target/'assets').symlink_to(repo/'assets', target_is_directory=True)
    (target/'var').symlink_to(repo/'var', target_is_directory=True)
    files = {str(p.relative_to(target)):file_hash(p) for p in target.rglob('*')
             if p.is_file() and not p.is_relative_to(target/'assets') and not p.is_relative_to(target/'var')}
    atomic_json(target/'clean-start.json', {'research_scope':SCOPE, 'source_files':files,
        'source_sha256':digest(files), 'excluded':['campaign Memory', 'Codex sessions',
            'historical results', 'Phase 1 training instructions'],
        'isolation':'Separate source and campaign state; same-account shared filesystem is not an access sandbox.'})
    return target/'configs/rsi/site.phase2-deepseek.json'


def meta_instructions(state):
    text = (REPO/'docs/phase2-meta-prompt.md').read_text()
    return text + f"\n本研究起始 checkpoint：{state['initial']['model_path']}。\n研究证据目录：{state['site']['campaign_root']}/{state['campaign_id']}。\n"


def campaign_site(site, campaign_id):
    runtime = Path(site['repo'])/'var/runtime'/('p2-'+hashlib.sha256(campaign_id.encode()).hexdigest()[:12])
    return {**site, 'codex_home': str(Path(site['repo'])/'var/meta/phase2'/campaign_id),
            'runtime_root': str(runtime), 'meta_bridge_root': str(runtime/'meta-bridge')}


def prepare_meta_home(original_site, site):
    """Reuse provider/login configuration, with independent instructions/sessions."""
    home = Path(site['codex_home']); home.mkdir(parents=True, exist_ok=True)
    original = Path(original_site.get('codex_home', home))
    for name in ('config.toml', 'auth.json'):
        if (original/name).is_file() and not (home/name).exists():
            shutil.copy2(original/name, home/name)
            (home/name).chmod(0o600)
    (home/'AGENTS.md').write_text((REPO/'docs/phase2-meta-prompt.md').read_text())


def turn_prompt(root, workspace, revision):
    return (f"读取 {workspace/'context.json'}、docs/phase2-prompt-research.md 和 "
            ".agents/skills/rsi-phase2/SKILL.md，继续 Phase 2 harness/skills 研究。"
            f"完整逐手棋局与 Memory 在 {root}。先分析败局，再修改提示或 skills。"
            "禁止任何训练或权重更新，违反则本阶段成果判0分。"
            "默认 proposal.json 的 jobs=[]，提交 L2/L3 各40局开发评测；"
            "模型 non-thinking，每个评测 vLLM max_num_seqs=20，300半步上限与五次重复判和。"
            "完成候选与本机检查后使用本轮 finish_research.py 交接。"
            "继续迭代用 decision=continue；交付用 decision=finish，"
            "final_candidate=best 可选此前观测最佳的完整候选，默认交付本轮候选。若已有足够结果可设evaluation=false直接交付best，无需重跑评测。"
            f"操作员恢复版本：{revision}。")


def publish_curve(root, state):
    points = []
    baseline = state.get('baseline_panel', {}).get('player', {})
    if baseline.get('complete'):
        points.append({'round':0, 'candidate_id':'initial', 'hypothesis':'Original Chess-R1 prompt baseline',
                       'model_path':state['initial']['model_path'], 'model_sha256':state['initial'].get('model_sha256'), 'score':baseline['score'],
                       'levels':baseline['levels'], 'source_sha256':state['initial']['source']['sha256']})
    for index, attempt in enumerate(state.get('attempts', []), 1):
        panel = attempt['panel']['player']
        points.append({'round':index, 'candidate_id':attempt['candidate_id'],
                       'hypothesis':attempt['hypothesis'], 'score':panel.get('score'),
                       'model_path':attempt.get('model_path'), 'model_sha256':attempt.get('model_sha256'), 'source_sha256':attempt.get('source_sha256'),
                       'levels':panel.get('levels', {}), 'status':attempt.get('status','evaluated' if panel.get('levels') else 'unevaluated'),
                       'evaluations':attempt.get('evaluations', [])})
    best = None
    for point in points:
        if point['score'] is not None:
            best = point['score'] if best is None else max(best, point['score'])
        point['best_observed'] = best
    atomic_json(Path(root)/'prompt-curve.json', {'research_scope':SCOPE,
        'initial_model':state['initial']['model_path'], 'points':points,
        'note':'Adaptive development measurements; observed best is not proof of monotonic improvement.'})


def deliver(root, state, selected, reason):
    result = {'research_scope':SCOPE, 'candidate_id':selected['id'],
        'model_path':selected['model_path'], 'model_sha256':selected.get('model_sha256'),
        'source':selected['source'], 'skills_path':selected.get('skills_path'),
        'skills_sha256':selected.get('skills_sha256'),
        'evaluations':selected.get('evaluations',state['baseline'] if selected['id']=='initial' else []),
        'max_model_len':state.get('inference_max_model_len',32768), 'reason':reason,
        'model_origin':state.get('model_origin'), 'completed':utc()}
    atomic_json(Path(root)/'final-candidate.json',result)
    state.update(phase='COMPLETE', final_candidate=result,
        final_checkpoint={k:result[k] for k in ('model_path','model_sha256','candidate_id')})
    return result
