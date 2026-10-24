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
  setAttribute(k,v){(this.attributes??={})[k]=v;}
  showModal(){this.open=true;}
  close(){this.open=false;}
}
const nodes=new Map(),el=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
const row=(i,channel='coach',category='text')=>({ordinal:i,event:{id:'e'+i,ts:'2026-10-21',kind:'meta_event',channel,category,title:'event '+i,text:'visible '+i},offset:i*100,length:100,path:'events.jsonl'});
let teacher=Array.from({length:180},(_,i)=>row(i+1)),studentEvents=[],evaluations=[{candidate_id:'initial',level:1,statistics:{score:.75}}];
let gameList=[{folder:'jobs/j1/output/harness_runs/g',id:'g',model_seat:'white'}],gamePages=null,overviewJob='j1';
let delay=null;
const requests=[];
const document={hidden:false,getElementById:el,createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag)};
function body(url){
  const u=new URL(url,'http://local');
  if(url==='/api/rsi/campaigns')return [];
  if(u.pathname.endsWith('/overview'))return {jobs:{},recent_jobs:[{id:overviewJob,role:'candidate',status:'Running'}],active_job:{id:overviewJob},incumbent:{id:'initial'},phase:'PLANNING',current_level:1,round:1};
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
  if(u.pathname.endsWith('/games'))return gameList;
  if(u.pathname.endsWith('/game-events')){
    if(gamePages?.length)return gamePages.shift();
    const out={events:studentEvents.map(event=>({event,offset:0,length:100})),cursor:10};studentEvents=[];return out;
  }
  if(u.pathname.endsWith('/artifact'))return {text:'raw '+u.searchParams.get('path')+' '+u.searchParams.get('offset')};
  throw Error('unexpected endpoint '+url);
}
const response=data=>({ok:true,json:async()=>data});
let intervalId=0;const intervals=new Map();
const context=vm.createContext({document,URL,URLSearchParams,AbortController,setTimeout,clearTimeout,
  setInterval:(fn,ms)=>{const id=++intervalId;intervals.set(id,{fn,ms});return id;},clearInterval:id=>intervals.delete(id),console,
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
  assert.equal(el('board').children.length,64);assert.equal(el('thinking').textContent,'Visible thought');
  assert.equal(el('student-output').textContent,'e2e4');
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
  // Compact history establishes every position; raw streaming resumes at its byte offset.
  const start='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
  const afterE4='rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1';
  const afterE5='rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2';
  const afterNf3='rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2';
  const history=[{type:'position',fen:start,side:'white'},
    {type:'turn_started',seat:'white'},{type:'model_request',status:'started',seat:'white',prompt:'first prompt'},
    {type:'model_request',status:'finished',seat:'white',text:'e2e4',output_excerpt:{head:'first thought',tail:'',omitted:0}},
    {type:'position',fen:afterE4,side:'black',last_move:{seat:'white',uci:'e2e4'}},
    {type:'position',fen:afterE5,side:'white',last_move:{seat:'black',uci:'e7e5'}},
    {type:'turn_started',seat:'white'},{type:'request',status:'started',seat:'white',prompt:'second prompt'}];
  const rows=history.map((event,i)=>({event:{...event,raw_ref:{offset:i*100,length:100}},offset:i*50,length:50}));
  gamePages=[{source:'short_events.jsonl',events:rows.slice(0,4),cursor:200,raw_cursor:400,more:true},
    {source:'short_events.jsonl',events:rows.slice(4),cursor:400,raw_cursor:800,more:false},
    {source:'events.jsonl',events:[{event:{type:'token',channel:'reasoning',text:'second live thought'},offset:800,length:100}],cursor:900,more:false}];
  run('openGame(state.games[0])');await run('readGame(state.viewer)');
  assert.equal(run('state.viewer.frames.length'),3);assert.equal(run('state.viewer.index'),2);
  assert.match(requests.at(-1),/view=full&cursor=700/);assert.equal(el('thinking').textContent,'second live thought');
  assert.equal(el('student-prompt').textContent,'second prompt');
  el('replay-prev').onclick();assert.equal(run('state.viewer.index'),1);
  assert.equal(el('thinking').textContent,'first thought');assert.equal(el('student-output').textContent,'e2e4');
  assert.equal(el('student-prompt').textContent,'first prompt');
  assert.equal(el('moves').children[1].attributes['aria-current'],'step');
  // Live completion and a different active cluster task cannot pull a replay forward.
  overviewJob='j2';gameList.push({folder:'jobs/j2/output/harness_runs/new',id:'new',model_seat:'black'});
  gamePages=[{source:'events.jsonl',events:[
    {event:{type:'request',status:'finished',seat:'white',reasoning:'second completed thought',text:'g1f3'},offset:900,length:100},
    {event:{type:'position',fen:afterNf3,side:'black',last_move:{seat:'white',uci:'g1f3'}},offset:1000,length:100}],cursor:1100,more:false}];
  await run('refresh()');assert.equal(run('state.job'),'j1');assert.equal(run('state.viewer.index'),1);
  assert.equal(run('state.viewer.frames.length'),4);assert.equal(el('thinking').textContent,'first thought');
  el('replay-range').value='3';el('replay-range').oninput();assert.equal(el('thinking').textContent,'second completed thought');
  el('moves').children[0].onclick();assert.equal(el('student-prompt').textContent,'');assert.equal(el('student-output').textContent,'');
  el('replay-restart').onclick();assert.equal(run('state.viewer.index'),0);
  let timer=run('state.viewer.timer');assert.equal(intervals.get(timer).ms,700);
  intervals.get(timer).fn();assert.equal(run('state.viewer.index'),1);
  el('replay-play').onclick();assert.equal(run('state.viewer.timer'),null);assert(!intervals.has(timer));
  el('replay-play').onclick();timer=run('state.viewer.timer');intervals.get(timer).fn();intervals.get(timer).fn();
  assert.equal(run('state.viewer.index'),3);assert.equal(run('state.viewer.timer'),null);assert(el('replay-next').disabled);
  // Switching games cancels playback and rejects an old game's in-flight response.
  const resolveGame=defer(url=>url.includes('/game-events?'));const oldGame=run('readGame(state.viewer)');
  run('openGame(state.games[1])');resolveGame({source:'events.jsonl',events:rows,cursor:10000,more:false});await oldGame;
  assert.equal(run('state.viewer.info.id'),'new');assert.equal(run('state.viewer.frames.length'),0);
  assert.equal(el('thinking').textContent,'');assert.equal(el('board').children.length,0);
  // A full 200-ply game crosses the per-batch page limit without dropping moves.
  gamePages=Array.from({length:9},(_,page)=>{
    const events=Array.from({length:Math.min(25,201-page*25)},(_,j)=>{
      const index=page*25+j;return {event:{type:'position',fen:start,side:'white',
        ...(index?{last_move:{seat:'white',uci:'e2e4'}}:{}),raw_ref:{offset:index*100,length:100}},offset:index*50,length:50};
    });
    return {source:'short_events.jsonl',events,cursor:Math.min(201,(page+1)*25)*50,
      raw_cursor:Math.min(201,(page+1)*25)*100,more:page<8};
  }).concat([{source:'events.jsonl',events:[],cursor:20100,more:false}]);
  await run('readGame(state.viewer)');await new Promise(resolve=>setTimeout(resolve,30));
  assert.equal(run('state.viewer.frames.length'),201);assert.equal(el('moves').children.length,201);
  assert.equal(run('state.viewer.loading'),false);assert.equal(Number(el('replay-range').max),200);
  el('moves').children[100].onclick();assert.equal(run('state.viewer.index'),100);
  console.log('RSI dashboard DOM checks passed: bounded feeds, replay, live isolation, history, pause and asynchronous races');
})().catch(error=>{console.error(error);process.exitCode=1;});
