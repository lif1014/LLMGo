"""Isolated, factual and executable Meta tasks; never submits Student work."""
from __future__ import annotations
import json
import shutil
import time
from pathlib import Path

from .contracts import REPO,atomic_json,digest,read_json,utc,file_hash


def cases():
    items=[]
    for i,(wins,draws,losses) in enumerate([(3,6,31),(17,9,14),(1,11,28),(40,0,0),(0,40,0),(0,0,40)]):
        expected={'wins':wins,'draws':draws,'losses':losses,'games':40,
                  'win_rate':wins/40,'score':(wins+.5*draws)/40}
        items.append({'id':f'metrics-{i}','kind':'facts','files':{'metrics.json':{'statistics':{k:expected[k] for k in ['wins','draws','losses','games']},'unverified_note':'A narrative note is not the formal result.'}},
                      'task':'Read metrics.json. Write observations.json with wins,draws,losses,games,win_rate and score. Both rates are fractions, not percentages. Use formal counts only.', 'expected':expected})
    for plies in [13,44,102]:
        events=[]
        for ply in range(1,plies+1):
            events.extend([{'type':'model_response','ply':ply,'text':'fixture'},
                           {'type':'student_move' if ply%2 else 'opponent_move','ply':ply,'san':'fixture'}])
        items.append({'id':f'plies-{plies}','kind':'facts','files':{'events.jsonl':events,'summary.json':{'plies':plies,'display_first_moves':events[:4]}},
                      'task':'Read the whole events.jsonl and summary.json. Write observations.json with plies and student_decisions. Do not count JSONL lines or display_first_moves as game length.',
                      'expected':{'plies':plies,'student_decisions':(plies+1)//2}})
    items.extend([
        {'id':'mixture-unit','kind':'facts','files':{'selection.json':{'unit':'puzzles','groups':{'fork':{'puzzles':10,'positions':17},'pin':{'puzzles':20,'positions':53}}}},
         'task':'Read selection.json. In observations.json report fork_puzzles,pin_puzzles,total_positions and ratio_unit.',
         'expected':{'fork_puzzles':10,'pin_puzzles':20,'total_positions':70,'ratio_unit':'puzzles'}},
        {'id':'mixture-shortage','kind':'facts','files':{'preview.json':{'status':'insufficient','requested_puzzles':4,'distinct_eligible_puzzles':3,'exported':False}},
         'task':'Read preview.json. In observations.json report status and exported. Do not claim that a dataset was created.',
         'expected':{'status':'insufficient','exported':False}},
        {'id':'multi-label','kind':'facts','files':{'stats.json':{'positions':70,'themes':{'fork':40,'pin':50},'classification_note':'Themes are multi-label counts of positions.'}},
         'task':'Read stats.json. In observations.json report positions,theme_total and multi_label (boolean). Do not change formal position count to the sum of tags.',
         'expected':{'positions':70,'theme_total':90,'multi_label':True}},
    ])
    for key,value in [('prompt_batch_size',64),('reward.format',.2)]:
        items.append({'id':'recipe-'+key.replace('.','-'),'kind':'recipe','change':key,'value':value,
                      'task':f'Copy configs/grpo/default.json to configs/grpo/experiment.json and change only {key} to {value}. Write an executable RL proposal pointing to that experiment template. Do not execute training.'})
    for i,(step,action,selected) in enumerate([(5,'continue',5),(5,'replan',5),(5,'finish',0),(5,'finish',5),(10,'continue',10),(150,'finish',150)]):
        payload={'step':step,'max_steps':150,'available_steps':sorted({0,5,step})}
        items.append({'id':f'review-{i}','kind':'review','review':{'request_id':digest(payload),'payload':payload},'action':action,'selected':selected,
                      'task':f'This is a checkpoint review. The requested decision is action={action}, selected_step={selected}. Read context.json and write proposal.json.training_review with the exact current request_id, action, a nonempty reason and selected_step. Do not create a new training job.'})
    return items


def _same(actual,expected):
    if isinstance(expected,float):return type(actual) in {int,float} and abs(actual-expected)<1e-9
    return type(actual)==type(expected) and actual==expected


def grade(case,workspace):
    from .proposal import validate_workspace_proposal
    workspace=Path(workspace)
    proposal=read_json(workspace/'proposal.json')
    validate_workspace_proposal(workspace,proposal)
    if case['kind']=='facts':
        observed=read_json(workspace/'observations.json')
        return all(k in observed and _same(observed[k],v) for k,v in case['expected'].items())
    if case['kind']=='review':
        value=proposal['training_review']
        return value['action']==case['action'] and value['selected_step']==case['selected']
    before=read_json(workspace/'configs/grpo/default.json')
    after=read_json(workspace/'configs/grpo/experiment.json')
    expected=json.loads(json.dumps(before));parts=case['change'].split('.');target=expected
    for part in parts[:-1]:target=target[part]
    target[parts[-1]]=case['value']
    commands=proposal.get('jobs',[])
    return (after==expected and bool(commands) and any(
        '--template' in c.get('argv',[]) and 'configs/grpo/experiment.json' in c['argv']
        and 'scripts/run_grpo.py' in c['argv'] for c in commands))


def summarize(results):
    measured=[r for r in results if r.get('checks')]
    categories={}
    for kind in ['facts','recipe','review']:
        rows=[r for r in results if r['kind']==kind]
        categories[kind]={'passed':sum(r['passed'] for r in rows),'cases':len(rows)}
    return {'cases':len(results),'passed':sum(r['passed'] for r in results),
            'completion_rate':sum(r['passed'] for r in results)/len(results) if results else None,
            'first_check_schema_passes':sum(r['checks'][0]['ok'] for r in measured),
            'schema_measured_cases':len(measured),'schema_check_failures':sum(not c['ok'] for r in results for c in r.get('checks',[])),
            'categories':categories,'seconds':sum(r['seconds'] for r in results),
            'note':'First-check rate is measured at the required schema-check tool, not hidden drafts; repeated cases are not independent samples.'}


def run_benchmark(site,memory,attempt,backend_class):
    """One GPU lifetime, fresh Codex home/session for every case."""
    attempt=Path(attempt);count=site.get('meta_benchmark_repetitions',3)
    selected=cases();results=[]
    atomic_json(attempt/'benchmark-progress.json',{'stage':'running','completed':0,'total':len(selected)*count})
    for repeat in range(count):
        for case in selected:
            folder=attempt/'benchmark'/f"{case['id']}-r{repeat+1}";workspace=folder/'workspace';workspace.mkdir(parents=True)
            context={'engineering_benchmark':True,'case':case['id'],'repeat':repeat+1}
            if case['kind']=='review':context['training_review']=case['review']
            atomic_json(workspace/'context.json',context)
            for name,value in case.get('files',{}).items():
                if name.endswith('.jsonl'):(workspace/name).write_text(''.join(json.dumps(v)+'\n' for v in value))
                else:atomic_json(workspace/name,value)
            if case['kind']=='recipe':
                path=workspace/'configs/grpo/default.json';path.parent.mkdir(parents=True)
                shutil.copy2(REPO/'configs/grpo/default.json',path)
            input_hashes={str(p.relative_to(workspace)):file_hash(p) for p in workspace.rglob('*') if p.is_file()}
            check=workspace/'check_case.py'
            check.write_text("import sys,json\nfrom pathlib import Path\nsys.path.insert(0,"+repr(str(REPO/'src'))+")\nfrom llmgo.rsi.proposal import validate_workspace_proposal\np=Path(__file__).parent\ntry:\n validate_workspace_proposal(p,json.loads((p/'proposal.json').read_text()));record={'ok':True}\nexcept (ValueError,OSError) as e:record={'ok':False,'error':str(e)}\nwith (p/'checks.jsonl').open('a') as f:f.write(json.dumps(record)+'\\n')\nprint(json.dumps(record));raise SystemExit(0 if record['ok'] else 2)\n")
            home=folder/'codex-home';home.mkdir();shutil.copy2(attempt/'codex-home/config.toml',home/'config.toml')
            backend=backend_class({**site,'codex_home':str(home),'meta_bridge_root':str(folder/'bridge')},memory)
            backend.session_path=folder/'session.json';backend.session={}
            prompt=("工程驗收，只有本目錄內的合成資料；不是正式研究，不啟動任何GPU工作，不執行訓練，不讀其他研究或其他案例。"
                    "先使用工具讀context.json與資料，再完成指定任務。"+case['task']+
                    "非review案例也要寫proposal.json；facts案例用hypothesis字串、jobs空陣列、evaluation=false。"
                    "交接前必須用控制Python執行本目錄check_case.py，修正錯誤再重跑。最後使用本次finish_command交接ready。")
            started=time.monotonic();row={'id':case['id'],'kind':case['kind'],'repeat':repeat+1,'passed':False}
            try:
                answer=backend.turn(workspace,prompt)
                row['answer']=answer
                row['passed']=answer['status']=='ready' and grade(case,workspace)
            except Exception as e:row['error']=str(e)
            row['inputs_unchanged']=all((workspace/n).is_file() and file_hash(workspace/n)==identity for n,identity in input_hashes.items())
            row['passed']=row['passed'] and row['inputs_unchanged']
            row['checks']=[json.loads(line) for line in (workspace/'checks.jsonl').read_text().splitlines()] if (workspace/'checks.jsonl').exists() else []
            row['passed']=row['passed'] and bool(row['checks']) and row['checks'][-1]['ok']
            row['seconds']=time.monotonic()-started
            row['token_usage']=backend.session.get('token_usage')
            atomic_json(folder/'result.json',row);results.append(row)
            atomic_json(attempt/'benchmark-progress.json',{'stage':'running','completed':len(results),'total':len(selected)*count,'last_case':row['id'],'summary':summarize(results)})
    result={'model':site['meta_model'],'inference':{k:site.get(k) for k in ('meta_precision','meta_context_length','meta_max_output_tokens','meta_compact_tokens')},
            'repetitions':count,'summary':summarize(results),'results':results,'finished':utc()}
    atomic_json(attempt/'benchmark-result.json',result)
    atomic_json(attempt/'benchmark-progress.json',{'stage':'completed','completed':len(results),'total':len(results),'summary':result['summary']})
    return {'status':'needs_input','summary':f"Engineering benchmark completed: {result['summary']['passed']}/{len(results)}; no Student training launched",'proposal_path':None}


def compare(left,right):
    """Case-paired majority comparison; repetitions are not independent cases."""
    import collections
    import math
    for result in [left,right]:
        if result.get('admission',{}).get('platform_status')!='Succeeded':
            raise ValueError('Only platform-Succeeded, adopted benchmark results can be compared')
    def votes(result):
        groups=collections.defaultdict(list)
        for row in result['results']:groups[row['id']].append(row['passed'])
        return {key:sum(values)>len(values)/2 for key,values in groups.items()}
    a,b=votes(left),votes(right)
    if set(a)!=set(b):raise ValueError('Paired comparison requires the same completed case IDs')
    better=sum(b[k] and not a[k] for k in a);worse=sum(a[k] and not b[k] for k in a)
    n=better+worse
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(better,worse)+1))/2**n) if n else 1.
    return {'left_model':left['model'],'right_model':right['model'],
            'left_summary':left['summary'],'right_summary':right['summary'],
            'paired_cases':len(a),'right_only_success':better,'left_only_success':worse,
            'both_success':sum(a[k] and b[k] for k in a),'both_fail':sum(not a[k] and not b[k] for k in a),
            'exact_mcnemar_p_two_sided':p,'right_improved_at_p05':better>worse and p<.05,
            'scope':'Engineering factual/schema/tool cases, not Student chess improvement. Majority within repeated case; case IDs are the paired units.',
            'left_admission':left['admission'],'right_admission':right['admission']}
