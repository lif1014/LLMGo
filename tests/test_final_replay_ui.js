/* Run with: node tests/test_final_replay_ui.js */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const {mapSummary,mapEvent,readEventPage,statistics}=require('../src/llmgo/server/static/rsi-final.js');
const initial='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const after='rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1';
const records=[{type:'episode_started',fen:initial,student_color:'white'},
  {type:'model_request_started',request_id:'turn-1',messages:[{role:'user',content:'检查局面'}]},
  {type:'model_response',response:'棋'.repeat(50000)+' \\boxed{e4}',reasoning:'',finish_reason:'stop'},
  {type:'student_move',fen:after,uci:'e2e4'},
  {type:'episode_finished',result:1,reason:'checkmate'}];
const bytes=Buffer.from(records.map(r=>JSON.stringify(r)+'\n').join(''));
function page(buffer){return async(offset,length)=>({text:buffer.subarray(offset,offset+length).toString('utf8'),next_offset:Math.min(buffer.length,offset+length),eof:offset+length>=buffer.length});}
class Element{
  constructor(){this.children=[];this.style={};this.dataset={};this.value='';this.textContent='';this.checked=false;}
  append(...items){this.children.push(...items);}prepend(...items){this.children.unshift(...items);}
  replaceChildren(...items){this.children=[...items];}setAttribute(key,value){this[key]=value;}
  add(item){this.append(item);}scrollIntoView(){}get options(){return this.children;}
}
(async()=>{
  const summary={completed:true,aborted:false,result:1,student_color:'white',reason:'checkmate',plies:1};
  let cursor=0,rows=[];
  do{const p=await readEventPage(page(bytes),cursor,'white',summary);assert.ok(p.cursor>cursor);cursor=p.cursor;rows.push(...p.events);}while(cursor<bytes.length);
  assert.equal(cursor,bytes.length);
  const response=rows.find(r=>r.event.status==='finished');
  assert.equal(response.event.text,records[2].response);
  assert.equal(response.offset,Buffer.byteLength(records.slice(0,2).map(r=>JSON.stringify(r)+'\n').join('')));
  assert.equal(response.length,Buffer.byteLength(JSON.stringify(records[2])+'\n'));
  const unfinished=await readEventPage(page(Buffer.from('{"type":"episode_st')),0,'white');
  assert.equal(unfinished.cursor,0);assert.equal(unfinished.more,false);
  assert.equal(mapSummary({completed:true,result:1},'black').result,'black_win');
  assert.equal(mapSummary({aborted:true},'white').result,'aborted');
  assert.equal(mapEvent({type:'episode_aborted',error:'provider unavailable'},'white')[0].summary.result,'aborted');
  assert.deepEqual(statistics([{completed:true,result:1},{completed:true,result:0},{completed:true,result:-1},{aborted:true,result:-1}]),
    {wins:1,draws:1,losses:1,completed:3,winRate:1/3,score:.5,formal:false});
  assert.equal(statistics([]).winRate,null);
  assert.equal(statistics([],{status:'completed',games:20,wins:10,draws:2,losses:8,win_rate:.5,score_rate:.55}).completed,20);
  const ids=new Map(),document={baseURI:'http://localhost/',hidden:false,
    getElementById:id=>{if(!ids.has(id))ids.set(id,new Element());return ids.get(id);},
    createElement:()=>new Element(),createElementNS:()=>new Element(),querySelector:()=>new Element()};
  const context=vm.createContext({document,URL,URLSearchParams,AbortController,AbortSignal,console,setTimeout,clearTimeout,setInterval:()=>{},
    Option:function(text,value){const el=new Element();el.textContent=text;el.value=value;return el;},fetch:async()=>({ok:true,json:async()=>[]})});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../src/llmgo/server/static/rsi.js'),'utf8'),context);
  let reads=0;
  const game={id:'deepseek-final-00',folder:'final/run/deepseek-final-00',rawFolder:'final-test/games/deepseek-final-00',
    apiBase:'api/grpo/runs/run',model_seat:'white',task_label:'1473058',opponent_text:'DeepSeek non-thinking',summary:mapSummary(summary,'white'),
    readEvents:async offset=>{reads++;return readEventPage(page(bytes),offset,'white',summary);}};
  await context.llmgoReplay.openFinalGame([game],game.folder);
  assert.ok(reads>=2);assert.equal(ids.get('game-scope').value,'final');
  assert.equal(ids.get('board').children.length,64);
  assert.equal(ids.get('moves').children.length,2);
  assert.match(ids.get('student-prompt').textContent,/检查局面/);
  assert.match(ids.get('student-output').textContent,/boxed\{e4\}/);
  assert.match(ids.get('game-summary').textContent,/学生胜/);
  assert.equal(ids.get('game').options[0].value,game.folder);
  const before=reads;context.llmgoReplay.setFinalGames([game]);
  await new Promise(resolve=>setImmediate(resolve));
  assert.ok(reads>before);assert.equal(ids.get('moves').children.length,2,'Refreshing must not duplicate moves');
  const trace=vm.runInContext('state.viewer.frames[1].view.events.find(e=>e.event.status==="finished")',context);
  assert.equal(trace.path,game.rawFolder+'/events.jsonl');assert.equal(trace.apiBase,game.apiBase);
  console.log('Final replay: partial/formal scores, aborted games, UTF-8 byte cursors, shared board, prompt/output and raw references passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
