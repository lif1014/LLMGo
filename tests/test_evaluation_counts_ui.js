/* Run with: node tests/test_evaluation_counts_ui.js */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const staticRoot=path.join(__dirname,'../src/llmgo/server/static');
class Element {
  constructor(){this.children=[];this.dataset={};this.style={};this.value='';this.textContent='';}
  append(...items){this.children.push(...items);}
  prepend(...items){this.children.unshift(...items);}
  replaceChildren(...items){this.children=[...items];}
  setAttribute(name,value){this[name]=value;}
  add(option){this.children.push(option);}
  get options(){return this.children;}
}
function environment(){
  const elements=new Map(),main=new Element();
  const document={baseURI:'http://localhost/',hidden:false,
    getElementById:id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},
    createElement:()=>new Element(),createElementNS:()=>new Element(),querySelector:()=>main};
  const context=vm.createContext({document,URL,URLSearchParams,AbortController,AbortSignal,
    setInterval:()=>{},setTimeout,clearTimeout,console,
    Option:function(text,value){const el=new Element();el.textContent=text;el.value=value;return el;},
    fetch:async()=>({ok:true,json:async()=>[]})});
  return {context,elements,main};
}
const stats=(games,wins,draws)=>({games,wins,draws,losses:games-wins-draws,score:(wins+.5*draws)/games});
const points=[20,40].map((games,i)=>({candidate_id:'candidate-'+games,lineage_step:i,
  score:.5,best_observed:.5,model_path:'fixture',levels:[2,3].map(level=>({level,
    statistics:stats(games,level===2?10:2,0),ref:{path:''}}))}));
const rows=points.flatMap(p=>p.levels.map(level=>({...level,candidate_id:p.candidate_id,
  track:'player',ref:{job_id:p.candidate_id}})));
const dashboard=environment();
vm.runInContext(fs.readFileSync(path.join(staticRoot,'rsi.js'),'utf8'),dashboard.context);
dashboard.context.fixtureRows=rows;
vm.runInContext(`
  state.evaluations=fixtureRows;
  state.gamesScope='campaign';
  for(const games of [20,40]){
    const id='candidate-'+games;
    state.jobs.push({id,kind:'evaluate_panel',levels:[2,3]});
    state.manifests.set(id,{payload:{protocol:{games:2*games,levels:[2,3]},
      evaluation_panel:[2,3].map(level=>({protocol:{opponent:{level},games}}))}});
    if(plannedGames(id,2)!==games||plannedGames(id,3)!==games)throw Error('Wrong planned batch size');
    if(jobStats(id,2).winRate!==10/games)throw Error('Wrong per-level denominator');
    if(jobStats(id).completed!==2*games)throw Error('Wrong combined batch size');
  }
  results(fixtureRows);
`,dashboard.context);
const table=dashboard.elements.get('matrix').children.find(el=>el.children[0]?.children[0]?.textContent==='候选 / 验证');
assert.ok(table,'Cross-level results table exists');
for(const [i,games] of [20,40].entries()){
  const cells=table.children[i+1].children;
  assert.equal(Number(cells[1].textContent),games);
  assert.equal(Number(cells[5].textContent),games);
  assert.equal(cells[2].textContent,(100*10/games).toFixed(1)+'%');
}
const scaling=environment();
let code=fs.readFileSync(path.join(staticRoot,'rsi-chess-r1.js'),'utf8');
code=code.replace('  async function openArtifact(run,rel){','  globalThis.renderCurveForTest=renderCurve;\n  async function openArtifact(run,rel){');
vm.runInContext(code,scaling.context);
scaling.context.renderCurveForTest(points);
const curveTable=scaling.main.children[0].children[3].children.find(el=>el.children[0]?.children[0]?.textContent==='累计 step');
assert.ok(curveTable,'Checkpoint results table exists');
for(const [i,games] of [20,40].entries()){
  const cells=curveTable.children[i+1].children;
  assert.equal(Number(cells[1].textContent),games);
  assert.equal(Number(cells[5].textContent),games);
  assert.equal(cells[3].textContent,(100*10/games).toFixed(1)+'%');
  assert.equal(cells[7].textContent,(100*2/games).toFixed(1)+'%');
}
console.log('Mixed 20/40-game batches: planned counts, separate L2/L3 rates and both dashboard tables passed');
