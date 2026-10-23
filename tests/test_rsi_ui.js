/* Actual dashboard script, bounded DOM, paged endpoints and asynchronous races. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
class Element {
  constructor(tag='div'){this.tagName=tag;this.children=[];this.value='';this._text='';this.dataset={};this.hidden=false;}
  set textContent(v){this._text=String(v);this.children=[];}
  get textContent(){return this._text+this.children.map(c=>c.textContent).join('');}
  append(...nodes){for(const n of nodes){this.children.push(n);n.parent=this;}}
  prepend(n){this.children.unshift(n);n.parent=this;}
  replaceChildren(...nodes){this.children=[];this._text='';this.append(...nodes);}
  add(n){this.append(n);}
  get lastChild(){return this.children.at(-1);}
  remove(){this.parent.children.splice(this.parent.children.indexOf(this),1);}
  setAttribute(){}
  showModal(){this.open=true;}
  close(){this.open=false;}
}
const nodes=new Map(),el=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
const row=(i,channel='coach',category='text')=>({ordinal:i,event:{id:'e'+i,ts:'2026-10-21',kind:'meta_event',channel,category,title:'event '+i,text:'visible '+i},offset:i*100,length:100,path:'events.jsonl'});
let teacher=Array.from({length:180},(_,i)=>row(i+1)),studentEvents=[],evaluations=[{candidate_id:'initial',level:1,statistics:{score:.75}}];
let delay=null;
const requests=[];
const document={hidden:false,getElementById:el,createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag)};
function body(url){
  const u=new URL(url,'http://local');
  if(url==='/api/rsi/campaigns')return [];
  if(u.pathname.endsWith('/overview'))return {jobs:{},recent_jobs:[{id:'j1',role:'candidate',status:'Running'}],active_job:{id:'j1'},incumbent:{id:'initial'},phase:'PLANNING',current_level:1,round:1};
  if(u.pathname.endsWith('/executor'))return {mode:'external',report:{status:'waiting',task_id:4170,checks:3,interval_seconds:60}};
  if(u.pathname.endsWith('/activity')){
    const source=u.searchParams.get('source');
    let records=source==='research'?teacher:[{...row(1,'executor','poll'),path:'executor/events.jsonl',event:{...row(1).event,channel:'executor',category:'poll',title:'poll complete',text:'isolated cluster stdout'}}];
    const high=records.at(-1)?.ordinal||0,before=u.searchParams.get('before'),cursor=u.searchParams.get('cursor');
    records=before?records.filter(r=>r.ordinal<Number(before)).slice(-80):cursor!==null?records.filter(r=>r.ordinal>Number(cursor)).slice(0,80):records.slice(-80);
    return {available:true,events:records,cursor:records.at(-1)?.ordinal||Number(cursor)||high,older_cursor:records[0]?.ordinal,has_older:records[0]?.ordinal>1,head:high};
  }
  if(u.pathname.endsWith('/evaluations'))return evaluations;
  if(u.pathname.endsWith('/job-files'))return {files:[],game_results:1,formal_games:10,progress:{stage:'training',curriculum_records:64}};
  if(u.pathname.endsWith('/games'))return [{folder:'jobs/j1/output/harness_runs/g',id:'g',model_seat:'white'}];
  if(u.pathname.endsWith('/game-events')){const out={events:studentEvents.map(event=>({event,offset:0,length:100})),cursor:10};studentEvents=[];return out;}
  if(u.pathname.endsWith('/artifact'))return {text:'raw '+u.searchParams.get('path')+' '+u.searchParams.get('offset')};
  throw Error('unexpected endpoint '+url);
}
const response=data=>({ok:true,json:async()=>data});
const context=vm.createContext({document,URL,URLSearchParams,AbortController,setTimeout,clearTimeout,setInterval:()=>1,console,
  Option:class extends Element{constructor(text,value){super('option');this.textContent=text;this.value=value;}},
  fetch:async url=>{requests.push(url);if(delay&&delay.match(url)){const item=delay;delay=null;return item.promise;}return response(body(url));}
});
const run=source=>vm.runInContext(source,context);
vm.runInContext(fs.readFileSync(path.join(__dirname,'../src/llmgo/server/static/rsi.js'),'utf8'),context);
function defer(match){let resolve;const promise=new Promise(r=>{resolve=r;});delay={match,promise};return data=>resolve(response(data));}
(async()=>{
  await new Promise(resolve=>setImmediate(resolve));run('state.campaign="test"');
  studentEvents=[{type:'position',fen:'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',side:'white'},
    {type:'turn_started',tool_library:['position','candidates']},{type:'model_request',status:'started'},
    {type:'token',channel:'reasoning',text:'Visible ',replace:false},{type:'token',channel:'reasoning',text:'thought',replace:false},
    {type:'token',channel:'content',text:'e2e4',replace:false}];
  await run('refresh()');
  assert.equal(el('board').children.length,64);assert.equal(el('thinking').textContent,'Visible thought\ne2e4');
  assert.equal(el('coach').children.length,80);assert.match(el('coach').textContent,/event 180/);assert.doesNotMatch(el('coach').textContent,/event 99\b/);
  assert.equal(el('coach').children[0].children[1].textContent,'','closed detail bodies are lazy');
  const student=el('student-events').children.at(-1);student.open=true;student.ontoggle();assert.match(student.textContent,/tool_library/);
  assert.match(el('executor-state').textContent,/4170/);assert.match(el('worker-state').textContent,/training/);
  assert.match(el('worker-state').textContent,/1\/10局/);
  assert.match(el('results').textContent,/75.0%/);
  assert.match(el('matrix').textContent,/部分测试 1\/8/);
  run('results([{candidate_id:"baseline",track:"player",level:1,statistics:{score:.5,games:10,wins:4,draws:2,losses:4}},{candidate_id:"candidate",track:"player",level:1,statistics:{score:.5,games:20,wins:8,draws:4,losses:8}}])');
  const cellTitles=n=>[n.title||'',...n.children.flatMap(cellTitles)];
  assert(cellTitles(el('matrix')).some(t=>t.endsWith('10局')));
  assert(cellTitles(el('matrix')).some(t=>t.endsWith('20局')));
  run('results(Array.from({length:8},(_,i)=>({candidate_id:"complete",track:"player",level:i+1,statistics:{score:.5,wins:5,draws:10,losses:5}})).concat([{candidate_id:"complete",track:"model",level:1,statistics:{score:1}}]))');
  assert.match(el('matrix').textContent,/50.0%/);assert.match(el('matrix').textContent,/模型对照/);
  assert.match(el('matrix').textContent,/部分测试 1\/8/);
  assert.doesNotMatch(el('overview').textContent,/当前L|已通关/);
  assert.equal(requests.filter(r=>r.includes('/executor-events?')||r.includes('/events?')).length,0,'automatic refresh never consumes raw research logs');
  assert.doesNotMatch(el('coach').textContent,/isolated cluster/);
  el('coach-filter').value='tool';el('coach-filter').onchange();assert(el('coach').children.every(n=>n.hidden));
  el('coach-filter').value='all';el('coach-filter').onchange();
  await run('readActivity("research",true)');assert.match(el('coach-window').textContent,/历史页/);assert.match(el('coach').textContent,/event 100/);
  const historyRequests=requests.filter(r=>r.includes('source=research')).length;
  await run('refresh()');assert.equal(requests.filter(r=>r.includes('source=research')).length,historyRequests,'older pages suspend that source');
  await run('returnLive("research")');assert.match(el('coach').textContent,/event 180/);
  // An older-page response wins over an in-flight incremental refresh.
  const resolveDelta=defer(url=>url.includes('source=research'));
  const pending=run('readActivity("research")');await run('readActivity("research",true)');
  resolveDelta({available:true,events:[row(999)],cursor:999,older_cursor:999,has_older:true});await pending;
  assert.doesNotMatch(el('coach').textContent,/event 999/);assert.match(el('coach-window').textContent,/历史页/);
  run('usage({total:{inputTokens:999}});consume({event:{kind:"meta_event",channel:"coach",category:"usage",title:"old usage",usage:{total:{inputTokens:1}}},offset:0,length:100,path:"events.jsonl"},"research")');
  assert.match(el('meta-usage').textContent,/输入 999/,'reading history must not roll back the latest token counters');
  await run('returnLive("research")');teacher=teacher.concat(Array.from({length:40},(_,i)=>row(181+i)));
  await run('readActivity("research")');await run('readActivity("research",true)');
  assert.match(el('coach').textContent,/event 140\b/,'history follows the sliding live window without skipping older records');
  // Thousands of chunks for the same item must remain a single bounded card.
  run('clearFeed("coach");for(let i=0;i<5000;i++)consume({event:{kind:"meta_event",channel:"coach",category:"text",title:"stream",thread_id:"t",turn_id:"turn",item_id:"item",stream:"content",stream_key:"t:turn:item:content",mode:"append",text:"x".repeat(100)},offset:i*100,length:100,path:"events.jsonl"},"research")');
  assert.equal(el('coach').children.length,1);assert.equal(el('coach-live').textContent.length,20000);
  assert(run('state.cards.size')<=80);for(let i=0;i<100;i++)run('detail("coach","event",{})');assert.equal(el('coach').children.length,80);
  const n=requests.length;el('pause').onclick();await run('refresh()');assert.equal(requests.length,n);run('state.paused=false');
  document.hidden=true;await run('refresh()');assert.equal(requests.length,n);document.hidden=false;
  // Explicit file reads use bounded pages and stale responses cannot replace a newer file.
  await run('openRaw({path:"one.log",offset:0,length:150000},true)');assert.match(el('raw-content').textContent,/84464/);
  el('raw-prev').onclick();await new Promise(r=>setImmediate(r));assert.match(el('raw-content').textContent,/18928/);
  const resolveOld=defer(url=>url.includes('path=old.log'));const old=run('openRaw({path:"old.log",offset:0,length:100})');
  await run('openRaw({path:"new.log",offset:0,length:100})');resolveOld({text:'obsolete'});await old;
  assert.match(el('raw-content').textContent,/new.log/);assert.doesNotMatch(el('raw-content').textContent,/obsolete/);
  console.log('RSI dashboard DOM checks passed: bounded streaming, filters, history, pause and asynchronous races');
})().catch(error=>{console.error(error);process.exitCode=1;});
