"""Explicit operator import of complete, pre-update validation evidence only.

This is neither checkpoint recovery nor replay training. Source and observation
identities must match; the old failure and its raw episodes remain untouched.
"""
import ast
from pathlib import Path

from llmgo.rsi.contracts import atomic_json, digest, file_hash, read_json, utc
from .stability import enforce_update_safety
from .validation import minimum_valid_rate, validation_identity


def _function(path, name):
    tree=ast.parse(Path(path).read_text())
    return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),
        include_attributes=False)


def import_baseline(previous_output, destination, identity, *, previous_manifest,
                    current_source, engine_evidence, approved_agent_hashes=None):
    """Import an unchanged policy's original observations after a gate-only fix.

    approved_agent_hashes explicitly attests an operator-reviewed logging-only
    revision. Arbitrary collector, prompt, sampling or environment edits fail.
    """
    old=Path(previous_output)/'training';out=Path(destination)
    prior=read_json(old/'identity.json');manifest=read_json(previous_manifest)
    if manifest['sha256'] != digest(manifest['payload']):raise ValueError('Manifest integrity changed')
    payload=manifest['payload'];source=Path(payload['candidate_source']['root'])
    for name,sha in payload['candidate_source']['files'].items():
        if file_hash(source/name)!=sha:raise ValueError('Previous source changed: '+name)
    if (old/'training-result.json').exists() or list((old/'checkpoints').glob('global_step_*')):
        raise ValueError('Baseline import requires no prior checkpoint/update')
    if (old/'metrics.jsonl').exists():
        import json
        if any(json.loads(line)['stage']=='update_completed'
               for line in (old/'metrics.jsonl').read_text().splitlines() if line):
            raise ValueError('Previous optimizer update found')
    for key in ('model','objective','cards','reward','runtime','allocated_gpus','verl','lc0_binary'):
        if prior[key]!=identity[key]:raise ValueError('Baseline identity changed: '+key)
    if prior.get('environment')!=identity.get('environment'):
        raise ValueError('Baseline environment changed')
    gates={'quality_gate_min_valid','validation_min_valid','validation_max_drop'}
    if {k:v for k,v in prior['training'].items() if k not in gates} != {
            k:v for k,v in identity['training'].items() if k not in gates}:
        raise ValueError('Only protocol gate changes may reuse these observations')
    harness={k.removeprefix('src/llmgo/harness/'):v
        for k,v in payload['candidate_source']['files'].items() if k.startswith('src/llmgo/harness/')}
    if harness!=identity['harness']:raise ValueError('Harness changed')
    for key in ('binary_sha256','nnue_sha256'):
        if engine_evidence[key]!=identity['fairy'][key]:raise ValueError('Fairy identity changed')
    for name in ('game.py','rollout.py','trajectories.py','quality.py','vllm_compat.py'):
        if prior['code'][name]!=identity['code'][name]:raise ValueError('Collector dependency changed: '+name)
    current_source=Path(current_source)
    if _function(source/'src/llmgo/training/verl_games.py','complete_games') != _function(
            current_source/'src/llmgo/training/verl_games.py','complete_games'):
        raise ValueError('Complete-game collector changed')
    agents=(prior['code']['verl_agent.py'],identity['code']['verl_agent.py'])
    if agents[0]!=agents[1] and agents!=tuple(approved_agent_hashes or ()):
        raise ValueError('Agent changed without reviewed logging-only hash pair')
    quality=read_json(old/'quality-latest.json');diag=read_json(old/'update-diagnostics-latest.json')
    sync=read_json(old/'sync-check-latest.json')
    if any(x.get('step')!=0 for x in (quality,diag,sync)) or quality['stage']!='validation':
        raise ValueError('Expected initial, pre-update validation')
    summaries=[];evidence={};requests=[]
    for folder in sorted((old/'episodes').glob('validation-00000000-*')):
        summary=read_json(folder/'summary.json');trajectory=read_json(folder/'trajectory.json')
        if (not summary['completed'] or summary['aborted'] or not trajectory['complete']
                or trajectory['training'] or trajectory.get('advantages_used_for_training',trajectory['training'])):
            raise ValueError('Partial or training episode cannot be an imported baseline')
        rows=trajectory['transitions']
        expected_policy=digest({'identity':digest(prior),'completed_updates':0})
        if (not rows or not rows[-1]['done'] or rows[-1]['reward']!=summary['result']
                or any(row['policy_version']!=expected_policy or row['student_color']!=summary['student_color']
                       or row['turn_index']!=i or row.get('aborted')
                       or (i<len(rows)-1 and (row['done'] or row['reward']!=0))
                       for i,row in enumerate(rows))):
            raise ValueError('Episode/policy/terminal consistency failed')
        game_requests=[r for row in rows for r in row['requests']]
        if len(game_requests)!=summary['requests']:raise ValueError('Missing requests')
        requests.extend(game_requests);summaries.append(summary)
        for name in ('summary.json','trajectory.json','events.jsonl'):
            p=folder/name;evidence[str(p)]=file_hash(p)
    cfg=identity['training']
    if len(summaries)!=cfg['validation_games'] or [x['student_color'] for x in summaries]!=[
            'white' if i%2==0 else 'black' for i in range(cfg['validation_games'])]:
        raise ValueError('Missing or unbalanced validation games')
    valid=sum(not r['failure_reason'] for r in requests)
    if (not requests or len(requests)!=quality['samples'] or valid!=quality['valid']
            or valid/len(requests)!=quality['valid_rate']
            or sum(r['response_tokens'] for r in requests)!=quality['response_tokens_total']):
        raise ValueError('Request evidence does not match quality report')
    if out.exists() and any(out.iterdir()):raise ValueError('Import destination must be empty')
    out.mkdir(parents=True,exist_ok=True)
    enforce_update_safety(quality,diag,sync,cfg,out)
    if quality['valid_rate']<minimum_valid_rate(cfg):raise ValueError('Baseline below absolute floor')
    provenance={'previous_output':str(previous_output),'previous_manifest_sha256':manifest['sha256'],
        'previous_identity':digest(prior),'current_identity':digest(identity),
        'approved_logging_only_agent_hashes':list(agents),'engine_evidence':engine_evidence,
        'purpose':'initial validation only; never replayed as gradient data',
        'evidence':evidence,'utc':utc()}
    for name in ('identity.json','quality-latest.json','update-diagnostics-latest.json','sync-check-latest.json'):
        provenance['evidence'][str(old/name)]=file_hash(old/name)
    row={'step':0,'stage':'baseline','validation_identity':validation_identity(cfg,digest(identity)),
        'valid_rate':quality['valid_rate'],'passed':True,'minimum_valid_rate':minimum_valid_rate(cfg),
        'score':sum((s['result']+1)/2 for s in summaries)/len(summaries),
        'wins':sum(s['result']==1 for s in summaries),'losses':sum(s['result']==-1 for s in summaries),
        'draws':sum(s['result']==0 for s in summaries),
        'protocol_forfeits':sum(s['reason']=='protocol_forfeit' for s in summaries),
        'games':summaries,'reference_kl_move_mean':diag['reference_kl_move_mean'],'sync':sync,
        'imported':True,'provenance_digest':digest(provenance),'utc':utc()}
    atomic_json(out/'baseline-import.json',provenance)
    atomic_json(out/'validation-baseline.json',row)
    atomic_json(out/'validation/step-00000000.json',row)
    atomic_json(out/'validation-latest.json',row)
    return row
