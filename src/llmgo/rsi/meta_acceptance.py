"""Real long-input inference checks; no Student games or training."""
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import uuid

from .contracts import atomic_json,read_json,utc


def check_long_context(site,base_url,attempt):
    attempt=Path(attempt)
    records=[]
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for target,budget in [(60000,65536),(120000,8192)]:
        nonce=uuid.uuid4().hex
        prompt_path=attempt/f'context-input-{target}.json'
        code="""import sys,json
from transformers import AutoTokenizer
model,out,target,nonce=sys.argv[1:];target=int(target)
t=AutoTokenizer.from_pretrained(model)
unit='This paragraph is padding for an engineering context test. '
n=max(1,(target-200)//len(t.encode(unit,add_special_tokens=False)))
text=unit*(n//2)+'\\nThe verification key is '+nonce+'.\\n'+unit*(n-n//2)+'\\nReturn only the verification key stated in the middle of this document.'
messages=[{'role':'user','content':text}]
ids=t.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False,enable_thinking=True,preserve_thinking=True,reasoning_effort='xhigh')
json.dump({'messages':messages,'input_tokens':len(ids)},open(out,'w'))
"""
        p=subprocess.run([site.get('meta_inference_python',site['inference_python']),'-I','-c',code,site['meta_model_path'],str(prompt_path),str(target),nonce],capture_output=True,text=True)
        if p.returncode:raise RuntimeError('Long-context tokenizer failed: '+p.stderr[-1200:])
        source=read_json(prompt_path)
        body={'model':site['meta_model'],'messages':source['messages'],'max_tokens':budget,
              'reasoning_effort':site.get('meta_reasoning_effort','xhigh'),
              'chat_template_kwargs':{'enable_thinking':True,'preserve_thinking':True,'reasoning_effort':site.get('meta_reasoning_effort','xhigh')},
              **site.get('meta_sampling',{})}
        if source['input_tokens']+budget>site['meta_context_length']:raise ValueError('Long-context test exceeds declared window')
        started=time.monotonic()
        request=urllib.request.Request(base_url+'/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with opener.open(request,timeout=None) as response:result=json.load(response)
        atomic_json(attempt/f'context-response-{target}.json',result)
        choice=result['choices'][0];content=choice['message'].get('content') or ''
        row={'target_input_tokens':target,'tokenized_input':source['input_tokens'],'requested_output_budget':budget,
             'usage':result.get('usage'),'finish_reason':choice.get('finish_reason'),'key_recovered':nonce in content,
             'seconds':time.monotonic()-started}
        records.append(row)
        atomic_json(attempt/'long-context.json',{'status':'running','checks':records,'updated':utc()})
        if not row['key_recovered'] or choice.get('finish_reason')=='length':raise RuntimeError('Long-context retrieval failed; see context response')
    value={'status':'passed','checks':records,'updated':utc(),
           'note':'Actual long inputs with declared output budgets; not a claim that 64K output tokens were generated.'}
    atomic_json(attempt/'long-context.json',value)
    return value
