#!/usr/bin/env python3
"""Verify actual LiteLLM Responses<->Chat reasoning/tool history on loopback."""
import argparse
import asyncio
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from llmgo.rsi.contracts import atomic_json


def check():
    import litellm
    captured=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));captured.append(body)
            if len(captured)==1:
                message={'role':'assistant','content':None,'reasoning_content':'history-verification-nonce',
                         'tool_calls':[{'id':'call-proof','type':'function','function':{'name':'read_evidence','arguments':'{}'}}]}
                finish='tool_calls'
            else:message={'role':'assistant','content':'Verified'};finish='stop'
            value={'id':'chat-proof','object':'chat.completion','created':1,'model':'Qwen3.8-27B',
                   'choices':[{'index':0,'message':message,'finish_reason':finish}],
                   'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}}
            data=json.dumps(value).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    async def run():
        common={'model':'openai/Qwen3.8-27B','api_base':f'http://127.0.0.1:{server.server_port}/v1','api_key':'local-test',
                'use_chat_completions_api':True,'num_retries':0,'drop_params':True,
                'extra_body':{'chat_template_kwargs':{'enable_thinking':True,'preserve_thinking':True}}}
        initial=[{'role':'user','content':'Read the evidence using the tool.'}]
        first=await litellm.aresponses(input=initial,tools=[{'type':'function','name':'read_evidence','description':'Read evidence','parameters':{'type':'object','properties':{}}}],**common)
        items=[x.model_dump(mode='json',exclude_none=True) if hasattr(x,'model_dump') else x for x in first.output]
        await litellm.aresponses(input=initial+items+[{'type':'function_call_output','call_id':'call-proof','output':'Evidence read'}],**common)
    try:asyncio.run(run())
    finally:server.shutdown();server.server_close()
    assistants=[m for m in captured[1]['messages'] if m['role']=='assistant']
    reasoning=any('history-verification-nonce' in (m.get('reasoning_content') or '') for m in assistants)
    tool_ids=any(any(t.get('id')=='call-proof' for t in m.get('tool_calls',[])) for m in assistants)
    tools=any(m.get('role')=='tool' and m.get('tool_call_id')=='call-proof' for m in captured[1]['messages'])
    if not (reasoning and tool_ids and tools):raise RuntimeError('Reasoning or tool-call history was dropped')
    return {'status':'passed','requests':len(captured),'reasoning_history_preserved':reasoning,'tool_identity_preserved':tool_ids and tools}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    for key in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy','ALL_PROXY','all_proxy']:os.environ.pop(key,None)
    os.environ['LITELLM_LOCAL_MODEL_COST_MAP']='True';os.environ['LITELLM_TELEMETRY']='False'
    try:result=check()
    except Exception as e:
        atomic_json(a.output,{'status':'failed','error':str(e)});raise
    atomic_json(a.output,result);print(json.dumps(result))


if __name__=='__main__':main()
