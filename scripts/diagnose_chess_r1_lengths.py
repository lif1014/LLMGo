#!/usr/bin/env python3
"""One-GPU matched-puzzle prompt/sampling diagnostic; never updates weights."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def save(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=json.loads(a.recipe.read_text())
    a.output.mkdir(parents=True,exist_ok=False)
    from llmgo.training.vllm_compat import install_rope_compat
    install_rope_compat()
    import chess
    import torch
    from vllm import LLM,SamplingParams
    from llmgo.training.chess_r1 import extract_answer
    assert torch.cuda.device_count()==1
    path=Path(r['dataset']);raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==r['dataset_sha256']
    inputs=[json.loads(line) for line in raw.decode().splitlines()]
    save(a.output/'identity.json',{'recipe':r,'gpu':torch.cuda.get_device_name(0),
                                 'weights_updated':False,'requests':len(inputs)})
    save(a.output/'progress.json',{'stage':'initializing_vllm','requests':len(inputs)})
    model=LLM(model=r['model_path'],tensor_parallel_size=1,gpu_memory_utilization=.4,
              max_model_len=24576,max_num_seqs=32,max_num_batched_tokens=8192,
              enable_chunked_prefill=True,enable_prefix_caching=True,language_model_only=True,
              compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32],
                                  'cudagraph_mode':'PIECEWISE'})
    params=[SamplingParams(n=1,max_tokens=16384,seed=row['seed'],ignore_eos=False,
                           repetition_penalty=1.,**row['sampling']) for row in inputs]
    save(a.output/'effective-sampling.json',[{'case':row['case'],'position':row['position'],
                                            'sampling_params':str(sp)} for row,sp in zip(inputs,params)])
    save(a.output/'progress.json',{'stage':'generating','requests':len(inputs),'gpu_count':1})
    started=time.monotonic()
    outputs=model.generate([{'prompt_token_ids':row['prompt_token_ids']} for row in inputs],
                           params,use_tqdm=False)
    seconds=time.monotonic()-started
    statistics_rows=[]
    with (a.output/'raw-output.jsonl').open('w') as f:
        for row,output in zip(inputs,outputs,strict=True):
            sample=output.outputs[0];ids=sample.token_ids;text=sample.text
            truncated=sample.finish_reason=='length' or len(ids)>=16384
            final=text.rsplit('</think>',1)[-1].strip()
            if row['answer_notation']=='san':
                answer=extract_answer(text,row['fen'],finish_reason='length' if truncated else None)
                legal=answer.legal
            else:
                import re
                boxes=re.findall(r'\\boxed\s*\{([^{}]+)\}',final)
                legal=not truncated and '</think>' in text and len(boxes)==1 and boxes[0].strip() in {
                    move.uci() for move in chess.Board(row['fen']).legal_moves}
            ngrams=Counter(tuple(ids[i:i+16]) for i in range(max(0,len(ids)-15)))
            repeated=(sum(ngrams.values())-len(ngrams))/max(1,sum(ngrams.values()))
            record={'case':row['case'],'position':row['position'],'puzzle_id':row['puzzle_id'],
                    'prompt_tokens':len(row['prompt_token_ids']),'response_tokens':len(ids),
                    'truncated':int(truncated),'closing_think':int('</think>' in text),
                    'legal_final':int(legal),'finish_reason':sample.finish_reason,
                    'repeated_16gram_fraction':repeated}
            statistics_rows.append(record)
            f.write(json.dumps({**record,'response':text,'token_ids':ids})+'\n')
    by_case=defaultdict(list)
    for row in statistics_rows:by_case[row['case']].append(row)
    cases=[];scalars={}
    for case,rows in by_case.items():
        summary={'case':case,'requests':len(rows),
                 'response_tokens_mean':statistics.mean(x['response_tokens'] for x in rows),
                 'response_tokens_median':statistics.median(x['response_tokens'] for x in rows),
                 'truncated_fraction':statistics.mean(x['truncated'] for x in rows),
                 'thinking_completed_fraction':statistics.mean(x['closing_think'] for x in rows),
                 'legal_final_fraction':statistics.mean(x['legal_final'] for x in rows),
                 'repeated_16gram_fraction':statistics.mean(x['repeated_16gram_fraction'] for x in rows)}
        cases.append(summary)
        scalars.update({f'diagnostics/{case}/{key}':value for key,value in summary.items()
                        if isinstance(value,(int,float))})
    save(a.output/'result.json',{'status':'passed','diagnostic_only':True,'weights_updated':False,
                              'seconds':seconds,'cases':cases})
    packet={'step':0,'time':time.time(),'metrics':scalars,
            'tables':{'case_summary':cases,'requests':statistics_rows}}
    (a.output/'metrics.jsonl').write_text(json.dumps(packet)+'\n')
    save(a.output/'progress.json',{'stage':'completed','gpu_count':1,'requests':len(inputs)})


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        # The shell retains the traceback locally; no raw text is relayed to W&B.
        raise
