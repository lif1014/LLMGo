"""Ten concurrent DeepSeek non-thinking games with loose, unambiguous boxed SAN extraction."""
import asyncio, json, os, re, sys, time, uuid, urllib.request, urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT
LEVEL=int(os.environ['CHESS_LEVEL'])
assert LEVEL in (2,3)
RUN_ID=os.environ.get('LLMGO_RUN_ID') or time.strftime('deepseek-%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex[:6]
assert re.fullmatch(r'[A-Za-z0-9_-]+',RUN_ID)
OUT=ROOT/'var/runs'/RUN_ID/'evaluations'/f'deepseek-flash-loose-l{LEVEL}'
KEY=Path(os.environ.get('DEEPSEEK_KEY_FILE','/user/lyh4170/.config/llmgo/deepseek-api-key'))
RELEASE=ROOT
PROXY_URL=os.environ.get('DEEPSEEK_PROXY_URL','http://whitelist-proxy.cybertron.svc.cluster.local:7891')
sys.path.insert(0,str(REPO/'src'))
from llmgo.training.game import ChessEpisode
from llmgo.rsi.ladder import opponent
import importlib.util
spec=importlib.util.spec_from_file_location('eval_deepseek_protocol',RELEASE/'src/llmgo/training/chess_r1.py')
protocol=importlib.util.module_from_spec(spec);sys.modules[spec.name]=protocol;spec.loader.exec_module(protocol)

def atomic(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)

def request(messages,request_id,timeout=90,max_tokens=16384):
 key=KEY.read_text().strip()
 body={'model':'deepseek-flash','thinking':{'type':'disabled'},'messages':messages,
       'temperature':1,'top_p':.95,'max_tokens':max_tokens,'stream':False}
 req=urllib.request.Request('https://api.deepseek.com/chat/completions',data=json.dumps(body).encode(),
       headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','X-Client-Request-Id':request_id})
 # Scope the whitelist proxy to DeepSeek API calls; cctl and local services are unaffected.
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':PROXY_URL,'https':PROXY_URL}))
 for attempt in range(3):
  try:
   with opener.open(req,timeout=timeout) as response:return json.load(response)
  except urllib.error.HTTPError as error:
   body=error.read(400).decode(errors='replace')
   if error.code not in (429,500,502,503,504) or attempt==2:raise RuntimeError(f'DeepSeek HTTP {error.code}: {body}')
  except (TimeoutError,urllib.error.URLError) as error:
   if attempt==2:raise RuntimeError('DeepSeek network '+type(error).__name__)
  time.sleep(2**attempt)

def extract_loose_answer(response,fen,*,finish_reason=None):
 """Accept one unambiguous canonical legal SAN across all output boxes.

 Surrounding prose and repeated identical boxes are allowed. Reasoning fields
 are never searched. Conflicting, malformed or truncated answers fall back.
 """
 if finish_reason in ('length','max_tokens'):
  return protocol.Answer(error='truncated')
 text=response.strip()
 for suffix in ('<|im_end|>','<|endoftext|>'):
  text=text.removesuffix(suffix).strip()
 if '</think>' in text:
  text=text.rsplit('</think>',1)[1].strip()
 if '<think>' in text or '</think>' in text:
  return protocol.Answer(error='incomplete_thinking')
 markers=list(re.finditer(r'\\boxed',text))
 starts=list(re.finditer(r'\\boxed\s*\{',text))
 if not starts:
  return protocol.Answer(error='missing_box')
 if len(markers)!=len(starts):
  return protocol.Answer(error='malformed_box')
 answers=[]
 for start in starts:
  depth=1;end=start.end()
  while end<len(text) and depth:
   depth+=(text[end]=='{')-(text[end]=='}');end+=1
  if depth:
   return protocol.Answer(error='unclosed_box')
  candidate=protocol.extract_answer(text[start.start():end],fen,finish_reason='stop')
  if not candidate.legal:
   return protocol.Answer(san=candidate.san,error=candidate.error or 'invalid_boxed_san')
  answers.append(candidate)
 if len({item.uci for item in answers})!=1:
  return protocol.Answer(error='conflicting_boxed_moves')
 return answers[0]

def summary(rows):
 levels={str(LEVEL):{'games':len(rows),'wins':sum(x['result']==1 for x in rows),'draws':sum(x['result']==0 for x in rows),
 'losses':sum(x['result']==-1 for x in rows),'score_rate':sum((x['result']+1)/2 for x in rows)/len(rows) if rows else None,
 'random_moves':sum(x['random_moves'] for x in rows),'decisions':sum(x['decisions'] for x in rows)}}
 atomic(OUT/'summary.json',{'model':'deepseek-flash','levels':levels,'games':rows});return levels

async def main():
 OUT.mkdir(parents=True,exist_ok=True)
 system=protocol.NONTHINKING_SYSTEM_PROMPT
 atomic(OUT/'protocol.json',{'model_label':'DeepSeek Flash · thinking disabled','model':'deepseek-flash','system_prompt':system,
   'prompt_source':str(ROOT/'assets/datasets/chess-r1/boxed-nonthinking-v2/train.parquet'),'opponents':[opponent(LEVEL)],
   'games_per_level':10,'white':5,'black':5,'scheduling':'independent_games','harness':False,'skills':False,
   'invalid_move_policy':'random_legal','max_plies':200,'thinking':{'type':'disabled'},'temperature':1,
   'answer_protocol':'unique_legal_boxed_san_loose_v1','answer_extraction':'all boxed SAN occurrences must be valid and identical; surrounding text and repeated identical boxes allowed; truncated answers rejected',
   'top_p_requested':.95,'top_p_effective':'1.0 (DeepSeek Flash non-thinking API)',
   'top_k':'unsupported by DeepSeek API','presence_penalty':'deprecated by DeepSeek API'})
 # Fail before opening ten games if this task cannot reach the API.
 first_messages=protocol.messages('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',enable_thinking=False)
 try:await asyncio.to_thread(request,first_messages,f'deepseek-loose-l{LEVEL}-preflight',12)
 except Exception as error:raise RuntimeError('DeepSeek API preflight failed: '+str(error)[:300])
 rows=[];lock=asyncio.Lock()
 async def play(i):
  folder=OUT/'games'/f'L{LEVEL}-{i:02d}'
  if (folder/'summary.json').exists():
   old=json.loads((folder/'summary.json').read_text())
   if old.get('completed'):return old
  if folder.exists():raise RuntimeError('Incomplete prior game requires manual review: '+str(folder))
  episode=ChessEpisode(f'L{LEVEL}-{i:02d}','deepseek-flash','white' if i%2==0 else 'black',LEVEL,
    str(REPO/'assets/engines/fairy-stockfish'),folder,seed=LEVEL*100000+i//2)
  try:
   await asyncio.to_thread(episode.advance)
   while not episode.done:
    messages=protocol.messages(episode.board.fen(),enable_thinking=False)
    request_id=episode.id+'-'+str(len(episode.board.move_stack))+'-'+uuid.uuid4().hex[:8]
    episode.event('model_request_started',request_id=request_id,messages=messages)
    start=time.monotonic();response=await asyncio.to_thread(request,messages,request_id)
    choice=response['choices'][0];content=choice['message']['content'] or '';reasoning=choice['message'].get('reasoning_content') or ''
    finish_reason=choice['finish_reason'];answer=extract_loose_answer(content,episode.board.fen(),finish_reason=finish_reason)
    usage=response.get('usage') or {};tokens=usage.get('completion_tokens',0)
    receipt={'request_id':request_id,'response':content,'raw_output':content,'reasoning':reasoning,
      'response_tokens':tokens,'finish_reason':finish_reason,'failure_reason':answer.error,
      'format_valid':answer.format_valid,'legal':answer.legal,'uci':answer.uci if answer.legal else None,
      'seconds':time.monotonic()-start,'api_model':response.get('model')}
    episode.event('model_response',**receipt)
    await asyncio.to_thread(episode.act,receipt['uci'],[receipt],fallback_reason=answer.error)
   row=json.loads((folder/'summary.json').read_text());row['level']=LEVEL;atomic(folder/'evaluation.json',row)
   async with lock:
    rows.append(row);levels=summary(rows)
    atomic(OUT/'progress.json',{'stage':'playing','completed_games':len(rows),'active_games':10-len(rows),'time':time.time()})
   return row
  except Exception as error:
   await asyncio.to_thread(episode.abort,type(error).__name__+': '+str(error)[:500]);raise
  finally:await asyncio.to_thread(episode.engine.close)
 existing=[]
 for i in range(10):
  p=OUT/'games'/f'L{LEVEL}-{i:02d}'/'evaluation.json'
  if p.exists():existing.append(json.loads(p.read_text()))
 rows.extend(existing);summary(rows)
 atomic(OUT/'progress.json',{'stage':'playing','completed_games':len(rows),'active_games':10-len(rows),'time':time.time()})
 pending=[i for i in range(10) if not (OUT/'games'/f'L{LEVEL}-{i:02d}'/'evaluation.json').exists()]
 results=await asyncio.gather(*(play(i) for i in pending),return_exceptions=True)
 errors=[str(x) for x in results if isinstance(x,BaseException)]
 if errors or len(rows)!=10:raise RuntimeError('Incomplete games: '+str(errors)[:1000])
 atomic(OUT/'result.json',{'status':'completed','model':'deepseek-flash','games':10,'levels':summary(rows),'time':time.time()})

if __name__=='__main__':
 try:asyncio.run(main())
 except Exception as error:
  atomic(OUT/'failure.json',{'type':type(error).__name__,'error':str(error)[:1000]});raise
