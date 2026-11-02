#!/usr/bin/env python3
"""Isolated one-GPU vLLM batch benchmark; no RL update or model export.

Each capacity runs in a fresh child so graph/KV allocations cannot leak between
cases. Fixed 512-token decoding is a throughput fixture, independent of the
production 16,384-token response budget.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def child(a):
    from llmgo.training.vllm_compat import install_rope_compat
    install_rope_compat()
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    assert torch.cuda.device_count() == 1
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    rows = pq.read_table(a.dataset).slice(0,16).to_pylist()
    sequences = tokenizer.apply_chat_template([r['prompt'] for r in rows], tokenize=True,
                                              add_generation_prompt=True, enable_thinking=True)
    if hasattr(sequences, 'keys'):
        sequences = sequences['input_ids']
    model = LLM(model=a.model, tensor_parallel_size=1, gpu_memory_utilization=.4,
                max_model_len=17408, max_num_seqs=a.capacity, max_num_batched_tokens=8192,
                enable_chunked_prefill=True, enable_prefix_caching=True, language_model_only=True,
                trust_remote_code=False, compilation_config={
                    'cudagraph_capture_sizes':[n for n in [1,2,4,8,16,32,64,128] if n<=a.capacity],
                    'cudagraph_mode':'PIECEWISE'})
    actual = model.llm_engine.vllm_config.scheduler_config.max_num_seqs
    assert actual == a.capacity
    model.generate([{'prompt_token_ids':sequences[0]}],
                   SamplingParams(temperature=1.,max_tokens=16,ignore_eos=True),use_tqdm=False)
    filler = tokenizer.encode(' I compare the candidate moves and calculate the likely reply.',add_special_tokens=False)
    results=[]
    for context in [0,8192]:
        prompts=[]
        for seq in sequences:
            ids=list(seq)
            if context:
                ids=(ids+filler*(1+context//len(filler)))[:context]
            prompts.append({'prompt_token_ids':ids})
        params=SamplingParams(n=8,temperature=1.,top_p=.95,top_k=20,
                              presence_penalty=1.5,repetition_penalty=1.,seed=1,
                              max_tokens=512,ignore_eos=True,logprobs=0)
        started=time.monotonic()
        generated=model.generate(prompts,params,use_tqdm=False)
        seconds=time.monotonic()-started
        lengths=[len(sample.token_ids) for request in generated for sample in request.outputs]
        assert len(lengths)==128 and min(lengths)==max(lengths)==512
        free,total=torch.cuda.mem_get_info(0)
        row={'capacity':a.capacity,'context_tokens':context or max(map(len,sequences)),
             'synthetic_context':int(context>0),'requests':len(lengths),'response_tokens':sum(lengths),
             'seconds':seconds,'tokens_per_second':sum(lengths)/seconds,
             'gpu_used_gib':(total-free)/2**30,'gpu_total_gib':total/2**30}
        results.append(row)
        save(a.output/f'capacity-{a.capacity}.json',results)
        print(json.dumps(row),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--capacity',type=int)
    a=p.parse_args()
    if a.capacity:
        child(a);return
    a.output.mkdir(parents=True,exist_ok=False)
    all_rows=[]
    try:
        for capacity in [32,64,128]:
            save(a.output/'progress.json',{'stage':'benchmarking_rollout','capacity':capacity,'gpu_count':1})
            with (a.output/f'capacity-{capacity}.log').open('w') as log:
                subprocess.run([sys.executable,__file__,'--model',a.model,'--dataset',a.dataset,
                                '--output',str(a.output),'--capacity',str(capacity)],
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            all_rows.extend(json.loads((a.output/f'capacity-{capacity}.json').read_text()))
        scalars={f'benchmark/capacity_{r["capacity"]}/context_{r["context_tokens"]}/tokens_per_second':
                 r['tokens_per_second'] for r in all_rows}
        row={'step':0,'time':time.time(),'metrics':scalars,'tables':{'rollout_capacity':all_rows}}
        (a.output/'metrics.jsonl').write_text(json.dumps(row)+'\n')
        save(a.output/'result.json',{'status':'passed','gpu_count':1,'tensor_parallel_size':1,
                                    'production_response_budget_unchanged':16384,
                                    'sampling':{'temperature':1.,'top_p':.95,'top_k':20,
                                                'presence_penalty':1.5,'repetition_penalty':1.},
                                    'cases':all_rows})
        save(a.output/'progress.json',{'stage':'completed','gpu_count':1})
    except BaseException as error:
        save(a.output/'failure.json',{'type':type(error).__name__,'error':str(error)})
        raise


if __name__=='__main__':main()
