'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
class Element{
  constructor(){this.children=[];this.style={};this.value='';this.textContent='';this.checked=false;}
  append(...items){this.children.push(...items);}replaceChildren(...items){this.children=[...items];}
  get options(){return this.children;}
}
const ids=new Map(),document={hidden:false,getElementById:id=>{
  if(!ids.has(id))ids.set(id,new Element());return ids.get(id);
},createElement:()=>new Element()};
const bankCalls=[];
const protocol=local=>({opponent:{kind:local?'local_model':'api',model:local?'Qwen3-8B-Chess':'deepseek-v4.1-flash'},
  student_protocol:{model_name:'Qwen3.5-4B'},games:40,white:20,black:20,parallel_games:40,max_plies:300,fivefold_repetition:true});
const context=vm.createContext({document,URLSearchParams,AbortSignal,TextEncoder,console,setInterval:()=>{},
  Option:function(text,value){const e=new Element();e.textContent=text;e.value=value;return e;},
  llmgoReplay:{isPaused:()=>false,setFinalGames:(games,key)=>bankCalls.push(key)},
  fetch:async url=>{
    const local=url.includes('/local/');
    if(url==='api/grpo/runs')return {ok:true,json:async()=>[{id:'deepseek'},{id:'local'}]};
    if(url.endsWith('/files'))return {ok:true,json:async()=>[{path:'final-test/protocol.json',modified:1,bytes:100}]};
    const path=new URL('http://fixture/'+url).searchParams.get('path');
    if(path==='final-test/protocol.json')return {ok:true,json:async()=>({text:JSON.stringify(protocol(local)),eof:true})};
    return {ok:false,status:404};
  }});
vm.runInContext(fs.readFileSync('src/llmgo/server/static/rsi-final.js','utf8'),context);
(async()=>{
  for(let i=0;i<20;i++)await new Promise(resolve=>setImmediate(resolve));
  assert.equal(ids.get('final-run').options.length,1);
  assert.equal(ids.get('final-run').value,'deepseek');
  assert.equal(ids.get('local-final-run').options.length,1);
  assert.equal(ids.get('local-final-run').value,'local');
  assert.match(ids.get('local-final-detail').textContent,/Qwen3-8B-Chess/);
  assert.ok(bankCalls.includes('deepseek-panel')&&bankCalls.includes('qwen-chess-panel'));
  console.log('Independent local and DeepSeek panels: filtered runs, isolated selections and replay banks passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
