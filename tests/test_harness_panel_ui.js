'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const {levelStatistics,candidateRows}=require('../src/llmgo/server/static/rsi-phase2.js');
const game=(id,result,seat='white',reason='checkmate')=>({id,folder:'jobs/j000002/games/'+id,model_seat:seat,
  summary:result===null?null:{result,result_reason:reason,random_moves:1}});
const games=[game('j000002-L2-00','white_win'),game('j000002-L2-01','white_win','black'),
  game('j000002-L2-02','draw'),game('j000002-L2-03',null),game('j000002-L2-04','black_win','white','harness_error'),
  game('j000002-L3-00','black_win')];
assert.deepEqual(levelStatistics(games,2),{wins:1,draws:1,losses:1,completed:3,planned:40,errors:1,formal:false,score:.5});
assert.equal(levelStatistics([...games,games[0]],2).completed,3,'duplicate receipts must not add a win');
assert.equal(levelStatistics([],2).score,null,'no completed games is unknown, not zero strength');
assert.equal(levelStatistics(games,3).losses,1);
const formal={wins:11,draws:7,losses:2,games:20,score:.725};
assert.equal(levelStatistics(games,2,formal,20).formal,true,'historical 20-game protocol is retained');
assert.equal(levelStatistics(games,2,formal,40).formal,false,'incomplete protocol must not become a final score');
assert.equal(candidateRows([{candidate_id:'initial',round:0}],
  [{id:'j1',candidate_id:'initial'},{id:'j2',candidate_id:'r000001',status:'Running'}]).length,2,
  'an evaluating candidate must appear before it enters prompt-curve.json');
assert.equal(candidateRows([],[{id:'old',candidate_id:'r000001'},
  {id:'retry',candidate_id:'r000001'}])[0].job.id,'retry','a retried live candidate must open its current job');

class Element{
  constructor(){this.children=[];this.style={setProperty(k,v){this[k]=v;}};this.value='';this.checked=false;this.listeners={};this.ownText='';}
  append(...nodes){this.children.push(...nodes);}replaceChildren(...nodes){this.children=[...nodes];this.ownText='';}
  set textContent(v){this.ownText=String(v??'');this.children=[];}get textContent(){return this.ownText+this.children.map(c=>c.textContent).join('');}
  setAttribute(k,v){this[k]=v;}addEventListener(k,f){this.listeners[k]=f;}
  get options(){return this.children;}
}
const ids=new Map(),el=id=>{if(!ids.has(id))ids.set(id,new Element());return ids.get(id);};
el('campaign').value='rl-study';el('phase2-follow').checked=true;
let poll,mode='partial',legacy=true,reads=[],opened=[];
const active='harness-base',old='harness-old';
const job={id:'j000002',candidate_id:'r000001',task_id:1655736,role:'candidate',status:'Running'};
const baseline={wins:0,draws:9,losses:31,games:40,score:.1125};
const complete={wins:30,draws:6,losses:4,games:40,score:.825};
const overview=name=>name==='rl-study'?{research_scope:'phase1_rl_only'}:{
  ...(legacy?{}:{research_scope:'phase2_prompt_only'}),model_origin:{model_path:'/models/'+name},
  recent_jobs:[{id:'j000001',candidate_id:'initial',role:'baseline',status:'Succeeded'},job],active_job:job,
  research_control:{paused:name===old},phase:'WAITING'};
const payload=value=>({text:JSON.stringify(value),eof:true});
async function fetch(url){
  reads.push(String(url));const u=new URL(url,'http://localhost/');let value,status=200;
  const campaign=u.pathname.match(/campaigns\/([^/]+)/)?.[1];
  if(u.pathname==='\/api/rsi/campaigns'||u.pathname==='api/rsi/campaigns')value=[];
  if(u.pathname==='/api/rsi/campaigns')value=[{campaign_id:'rl-study'},
    {campaign_id:old,updated:'2026-11-16'},{campaign_id:active,updated:'2026-11-17'}];
  else if(u.pathname.endsWith('/overview'))value=overview(campaign);
  else if(u.pathname.endsWith('/games'))value=games;
  else if(u.pathname.endsWith('/artifact')){
    const p=u.searchParams.get('path');
    if(p==='prompt-curve.json'){
      if(campaign==='rl-study')status=404;
      else value=payload({research_scope:'phase2_prompt_only',initial_model:'/models/'+campaign,
        points:[{candidate_id:'initial',round:0,levels:{'2':baseline,'3':{wins:0,draws:0,losses:40,games:40,score:0}},best_observed:.05625}]});
    }else if(p.endsWith('manifest.json'))value=payload({payload:{resources:{gpu_count:1},candidate_source:{files:{'harness.py':'hash','skills/hygiene.md':'hash'}},
      evaluation_panel:[{protocol:{opponent:{level:2},games:40}},{protocol:{opponent:{level:3},games:40}}]}});
    else if(p.endsWith('progress.json'))value=payload({updated:'2026-11-17T13:00:00Z',failures:[]});
    else if(p.endsWith('metrics.json')){
      if(mode==='partial')status=404;
      else value=payload({statistics:p.includes('/L2/')?complete:{wins:0,draws:0,losses:40,games:40,score:0}});
    }else if(p.endsWith('proposal.json'))value=payload({hypothesis:'Improve tactical safety',research_notes:{intervention:'Add attacked-piece annotations'}});
    else throw Error('Unexpected artifact '+p);
  }else if(value===undefined)throw Error('Unexpected URL '+url);
  return {ok:status===200,status,json:async()=>value};
}
const document={hidden:false,getElementById:el,createElement:()=>new Element(),createElementNS:()=>new Element()};
const context=vm.createContext({document,fetch,URL,URLSearchParams,AbortSignal,console,Date,
  setInterval:f=>{poll=f;},llmgoReplay:{isPaused:()=>false,openCampaignJob:async(...args)=>opened.push(args)}});
const tick=()=>new Promise(resolve=>setImmediate(resolve));
async function settle(){for(let i=0;i<12;i++)await tick();}
(async()=>{
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../src/llmgo/server/static/rsi-phase2.js'),'utf8'),context);
  await settle();
  assert.equal(el('phase2-panel').hidden,false,'harness is visible even when the global selection is RL');
  assert.equal(el('phase2-study').value,active,'active base research is chosen before paused history');
  assert.match(el('phase2-status').textContent,/1655736.*完成 4 \/ 80/);
  assert.match(el('phase2-levels').textContent,/L2 · 3 \/ 40 局.*50.0%.*1 胜 · 1 和 · 1 负/);
  assert.match(el('phase2-levels').textContent,/原始 prompt 基线/);
  assert.match(el('phase2-candidates').textContent,/r000001/,'live candidate is displayed alongside baseline');
  assert.match(el('phase2-plan').textContent,/harness.py.*skills\/hygiene.md/s);
  assert.ok(reads.every(url=>!url.includes('cctl')&&!url.includes('/run')),'refresh only reads the dashboard API');
  await el('phase2-replay').onclick();assert.equal(opened[0][0],active);assert.equal(opened[0][1],'j000002');
  const table=el('phase2-games').children[0],button=table.children[1].children.at(-1).children[0];
  await button.onclick();assert.equal(opened[1][2].id,games[0].id,'a game opens in the shared replay');
  mode='complete';await poll();await settle();
  assert.match(el('phase2-status').textContent,/完成 80 \/ 80/,'formal metrics override a lagging game list');
  assert.match(el('phase2-levels').textContent,/82.5%/);assert.match(el('phase2-updated').textContent,/两档完整结果/);
  el('phase2-study').value=old;el('phase2-study').onchange();await settle();
  assert.match(el('phase2-origin').textContent,/harness-old/);assert.equal(el('phase2-badge').textContent,'研究已暂停');
  el('campaign').value=active;el('campaign').listeners.change();await settle();
  assert.equal(el('phase2-study').value,active,'global Phase2 selection follows the same study');
  console.log('Harness panel: live/formal counts, legacy API, independent selection, paused history and replay passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
