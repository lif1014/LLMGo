"""Deterministic, stdlib-only research evidence; official metrics remain authoritative."""
from collections import Counter
from pathlib import Path
import statistics

from .contracts import atomic_json,file_hash,read_json,inside

MOVE_TYPES={'student_move','opponent_move','forced_move'}


def parse_game(path,expected_hash=None):
    """Read plain JSONL once, retaining every ply and exact raw event byte references."""
    import json
    path=Path(path)
    if expected_hash and file_hash(path)!=expected_hash:raise ValueError('Raw game hash mismatch: '+str(path))
    moves=[];decisions=[];finish=None;start=None;offset=0
    with path.open('rb') as stream:
        for raw in stream:
            event=json.loads(raw);kind=event.get('type')
            ref={'path':str(path),'offset':offset,'length':len(raw)};offset+=len(raw)
            if kind=='episode_started':start=event
            elif kind in MOVE_TYPES:moves.append({**event,'raw_reference':ref})
            elif kind in {'model_request_started','model_response','fallback'}:
                # Keep full answers/prompts in the original event, never truncate the move timeline.
                decisions.append({'type':kind,'ply':event.get('ply'),'request_id':event.get('request_id'),
                    'failure_reason':event.get('failure_reason') or event.get('reason'),'raw_reference':ref})
            elif kind=='episode_finished':finish=event
    if start is None or finish is None:raise ValueError('Incomplete game: '+str(path))
    plies=[m['ply'] for m in moves]
    if plies!=list(range(1,finish['ply']+1)):raise ValueError('Missing/duplicate/reordered moves: '+str(path))
    return {'game_id':start['episode_id'],'student_color':start['student_color'],
        'plies':finish['ply'],'result':finish['result'],'reason':finish['reason'],
        'raw_events':str(path),'raw_sha256':file_hash(path),'moves':moves,'decision_events':decisions}


def summarize_metrics(path,expected_hash,output):
    path=Path(path);output=Path(output)
    if expected_hash and file_hash(path)!=expected_hash:raise ValueError('Official metrics hash mismatch: '+str(path))
    data=read_json(path);stats=data['statistics'];games=data['games'];counts=Counter(g['diagnostics']['result'] for g in games)
    observed=(counts[1],counts[0],counts[-1],len(games))
    if observed!=tuple(stats[k] for k in ('wins','draws','losses','games')):raise ValueError('Official metrics/receipts disagree')
    score=(stats['wins']+.5*stats['draws'])/stats['games']
    if abs(score-stats['score'])>1e-10:raise ValueError('Official metrics score is inconsistent')
    rows=[]
    for receipt in games:
        parsed=parse_game(receipt['raw_events'],receipt.get('event_hashes',{}).get('raw_events'))
        diagnostic=receipt['diagnostics']
        if (parsed['result'],parsed['plies'],parsed['reason'])!=(diagnostic['result'],diagnostic['plies'],receipt['result_reason']):
            raise ValueError('Official receipt/raw game disagree: '+parsed['game_id'])
        target=output/'games'/(parsed['game_id']+'.json');atomic_json(target,parsed)
        rows.append({k:parsed[k] for k in ('game_id','student_color','plies','result','reason','raw_events')}|{'timeline':str(target)})
    lengths=[r['plies'] for r in rows];losses=[r for r in rows if r['result']==-1]
    return {'metrics':str(path),'metrics_sha256':file_hash(path),'protocol':data['protocol'],
        'official_statistics':stats,'win_rate':stats['wins']/stats['games'],'score':score,
        'plies':{'min':min(lengths),'median':statistics.median(lengths),'max':max(lengths)},
        'loss_length_bins_plies':{'at_most_8':sum(g['plies']<=8 for g in losses),
            '9_to_20':sum(8<g['plies']<=20 for g in losses),'over_20':sum(g['plies']>20 for g in losses)},
        'games':rows}


def publish_evidence(root,state,*,review=None):
    """Controller builds pinned reports before waking Meta; never calls an LLM or GPU."""
    root=Path(root);refs=[]
    for row in state.get('baseline',[])+state.get('pending_evaluations',[])+(state.get('working') or {}).get('evaluations',[]):
        if isinstance(row,dict) and row.get('ref'):refs.append(row['ref'])
    if review:refs.extend(review['payload']['evaluations'])
    reports=[];seen=set()
    for ref in refs:
        if ref['path'] in seen:continue
        seen.add(ref['path']);path=inside(state['site']['user_root'],ref['path'])
        folder=root/'research-evidence'/file_hash(path)
        report=summarize_metrics(path,ref['sha256'],folder)
        atomic_json(folder/'report.json',report)
        reports.append({'report':str(folder/'report.json'),'metrics':str(path),'metrics_sha256':ref['sha256'],
            'official_statistics':{k:report['official_statistics'][k] for k in ('wins','draws','losses','games','score')},'protocol':report['protocol']})
    result={'schema_version':1,'campaign_id':root.name,'analysis_python':state['site']['control_python'] if 'control_python' in state['site'] else '/user/lyh4170/venvs/llmgo-rsi-control-20261022/bin/python',
        'source_of_truth':'Pinned official metrics; derived timelines/statistics are cross-checked, not new evaluation results',
        'reports':reports,'policy':'Evaluation protocol is fixed. Display limits never change game length or the 300-ply referee cap.',
        'objective':'First train RL against L2 failure modes; keep L2/L3 40-game validation every5 GRPO steps. RL strategy remains autonomous.'}
    target=root/'research-evidence/index.json';atomic_json(target,result)
    lines=['# Controller 核驗的本研究證據', '',result['objective'], '',
        '正式 metrics 是唯一成績來源；以下是核對逐局收據與原始棋譜後的衍生報告。',
        '評測協議固定：L2/L3各40局、300半步上限、五次重複判和。不得截成4步評測。', '']
    for row in reports:
        st=row['official_statistics'];level=row['protocol'].get('opponent',{}).get('level','read protocol')
        lines.extend([f"L{level}: {st['wins']}勝 / {st['draws']}和 / {st['losses']}負；共{st['games']}局；得分率{st['score']:.2%}",
            '正式來源：'+row['metrics'], '完整雙方逐手與原文索引：'+row['report'], ''])
    (root/'research-evidence/README.md').write_text('\n'.join(lines)+'\n')
    return {'index':str(target),'read_first':str(root/'research-evidence/README.md'),'analysis_python':result['analysis_python'],'reports':reports}
