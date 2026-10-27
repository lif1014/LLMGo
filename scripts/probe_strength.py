#!/usr/bin/env python3
"""Authorized 10-game L2/L3 diagnostic, using the unchanged protected referee.

This is not a 20-game candidate promotion or a full-panel score. The existing
referee already supports balanced ten-game baseline protocols; reuse that mode
without inserting probe results into the campaign's baseline/best state.
"""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protected-root',type=Path,required=True)
    p.add_argument('--candidate-root',type=Path,required=True)
    p.add_argument('--site',required=True)
    p.add_argument('--model-path',required=True)
    p.add_argument('--levels',default='2,3')
    p.add_argument('--gpus',type=int,default=8,choices=range(1,9))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    sys.path.insert(0,str(args.protected_root/'src'))
    from llmgo.rsi.artifacts import source_snapshot,verify_source
    from llmgo.rsi.contracts import atomic_json,protocol,load_site,model_identity,file_hash,digest,inside,read_json
    from llmgo.rsi.evaluator import evaluate,engine_identity
    from llmgo.rsi.ladder import opponent
    protected=read_json(args.protected_root/'source-manifest.json')
    verify_source(protected)
    site=load_site(args.site)
    out=inside(site['user_root'],args.output)
    out.mkdir(parents=True,exist_ok=False)
    candidate=source_snapshot(args.candidate_root,out/'candidate-source')
    cards=Path(candidate['root'])/'skills.json'
    if not cards.is_file():cards=Path(candidate['root'])/'src/llmgo/harness/skills/cards.json'
    identity=model_identity(args.model_path)
    rows=[]
    for level in [int(x) for x in args.levels.split(',')]:
        spec=opponent(level);folder=out/f'L{level}';folder.mkdir()
        payload={'resources':{'gpu_count':args.gpus},'site':site,'source':protected,
            'reference_source':protected,'candidate_source':candidate,'role':'baseline',
            'job_id':f'fullgame-probe-L{level}','candidate_id':candidate['sha256'][:12],'track':'player',
            'output':str(folder),'model_path':str(Path(args.model_path).resolve()),
            'expected_model_sha256':identity['sha256'],'skills_path':str(cards),
            'skills_sha256':file_hash(cards),'protocol':protocol(spec,site['engine'],baseline=True),
            'engine_identity':engine_identity(site['engine'],spec),
            'seed_schedule':[100000*level+i for i in range(5)],
            'comparison_id':f'fullgame-ppo:L{level}:paired-10'}
        atomic_json(folder/'evaluation-input.json',payload)
        metrics=read_json(evaluate(payload))
        rows.append({'level':level,'statistics':metrics['statistics'],'path':str(folder/'metrics.json')})
        atomic_json(out/'comparison.json',{'purpose':'authorized_10_game_probe_not_promotion',
            'model_identity':identity,'candidate_source':candidate['sha256'],'results':rows,
            'complete_requested':len(rows)==len(args.levels.split(','))})
    print(json.dumps({'ok':True,'path':str(out/'comparison.json'),'results':rows}),flush=True)


if __name__=='__main__':main()
