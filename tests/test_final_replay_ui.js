/* Run with: node tests/test_final_replay_ui.js */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const {mapSummary,mapEvent,readEventPage,statistics,emptyRandomMoves,tallyRandomMove}=require('../src/llmgo/server/static/rsi-final.js');
const initial='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const after='rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1';
const records=[{type:'episode_started',fen:initial,student_color:'white'},
  {type:'model_request_started',request_id:'turn-1',messages:[{role:'user',content:'检查局面'}]},
  {type:'model_response',response:'棋'.repeat(50000)+' \\boxed{e4}',reasoning:'',finish_reason:'stop'},
  {type:'fallback',reason:'missing_box'},
  {type:'student_move',fen:after,uci:'e2e4',source:'random',fallback_reason:'missing_box'},
  {type:'opponent_move',ply:2,fen:'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2',uci:'e7e5',
    diagnostics:{model:'deepseek-v4.1-flash',response_model:'deepseek-v4.1-flash',raw_output:'\\boxed{e5}',parsed_san:'e5',parsed_uci:'e7e5',fallback_reason:null,finish_reason:'stop',response_tokens:8}},
  {type:'student_move',ply:3,fen:'rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2',uci:'g1f3'},
  {type:'opponent_move',ply:4,fen:'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3',uci:'b8c6',
    diagnostics:{raw_output:'An unboxed answer',parsed_san:null,parsed_uci:null,fallback_reason:'missing_box',finish_reason:'stop'}},
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
  const random=emptyRandomMoves();for(const event of records)tallyRandomMove(random,event);
  assert.equal(random.student_random_moves,1);assert.equal(random.opponent_random_moves,1);
  assert.deepEqual(random.student_fallback_reasons,{missing_box:1});
  assert.deepEqual(random.opponent_fallback_reasons,{missing_box:1});
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
    apiBase:'api/grpo/runs/run',model_seat:'white',task_label:'1473058',opponent_text:'DeepSeek non-thinking',summary:mapSummary(summary,'white'),randomMoves:random,
    readEvents:async offset=>{reads++;return readEventPage(page(bytes),offset,'white',summary);}};
  await context.llmgoReplay.openFinalGame([game],game.folder);
  assert.ok(reads>=2);assert.equal(ids.get('game-scope').value,'final');
  assert.equal(ids.get('board').children.length,64);
  assert.equal(ids.get('moves').children.length,5);
  assert.match(ids.get('student-prompt').textContent,/检查局面/);
  assert.match(ids.get('student-output').textContent,/boxed\{e4\}/);
  assert.match(ids.get('game-summary').textContent,/学生胜/);
  assert.match(ids.get('game-summary').textContent,/学生 1 \/ DeepSeek 1/);
  assert.equal(ids.get('game').options[0].value,game.folder);
  const before=reads;context.llmgoReplay.setFinalGames([game]);
  await new Promise(resolve=>setImmediate(resolve));
  assert.ok(reads>before);assert.equal(ids.get('moves').children.length,5,'Refreshing must not duplicate moves');
  assert.equal(ids.get('opponent-output').textContent,'An unboxed answer');
  assert.match(ids.get('opponent-diagnostics').textContent,/missing_box/);
  assert.match(ids.get('opponent-diagnostics').textContent,/b8c6/);
  vm.runInContext('showFrame(2)',context);
  assert.equal(ids.get('opponent-output').textContent,'\\boxed{e5}');
  assert.match(ids.get('opponent-ply').textContent,/第2半步/);
  assert.match(ids.get('opponent-diagnostics').textContent,/e7e5/);
  assert.match(ids.get('student-output').textContent,/boxed\{e4\}/,'Opponent output must not replace the student answer');
  vm.runInContext('showFrame(3)',context);
  assert.equal(ids.get('opponent-output').textContent,'\\boxed{e5}','A student move retains the preceding opponent response, not a future response');
  vm.runInContext('showFrame(1)',context);
  assert.match(ids.get('opponent-output').textContent,/尚无/,'Earlier positions must not expose later opponent answers');
  assert.equal(ids.get('opponent-raw').disabled,true);
  const opponentTrace=vm.runInContext('state.viewer.frames[2].opponent.raw',context);
  assert.equal(opponentTrace.path,game.rawFolder+'/events.jsonl');
  assert.equal(JSON.parse(bytes.subarray(opponentTrace.offset,opponentTrace.offset+opponentTrace.length).toString()).type,'opponent_move');
  const trace=vm.runInContext('state.viewer.frames[1].view.events.find(e=>e.event.status==="finished")',context);
  assert.equal(trace.path,game.rawFolder+'/events.jsonl');assert.equal(trace.apiBase,game.apiBase);
  console.log('Final replay: partial/formal scores, aborted games, UTF-8 byte cursors, shared board, prompt/output and raw references passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
