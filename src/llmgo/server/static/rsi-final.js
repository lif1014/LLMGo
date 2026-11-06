/* Held-out matches use the existing read-only run artifacts and shared replay. */
(() => {
  'use strict';
  const encoder=new TextEncoder();
  const percent=v=>typeof v==='number'?(100*v).toFixed(1)+'%':'—';
  function opponentName(protocol){return protocol?.opponent?.kind==='local_model'?(protocol.opponent.model||'本地模型'):'DeepSeek';}
  function mapSummary(summary,side){
    if(!summary)return null;
    const aborted=summary.aborted===true;
    if(!summary.completed&&!aborted)return null;
    const value=summary.result,other=side==='white'?'black':'white';
    return {result:aborted?'aborted':value===0?'draw':value===1?side+'_win':value===-1?other+'_win':null,
      result_reason:aborted?'aborted':summary.reason,model_decisions:summary.decisions,
      request_count:summary.requests,completion_tokens:summary.response_tokens,
      random_moves:summary.random_moves,fallback_reasons:summary.fallback_reasons||{}};
  }
  function mapEvent(event,side,summary){
    if(event.type==='episode_started')return [{type:'position',fen:event.fen,side:event.fen.split(' ')[1]==='w'?'white':'black'}];
    if(event.type==='model_request_started')return [{type:'turn_started',seat:side},
      {type:'model_request',status:'started',seat:side,request_id:event.request_id,
        prompt:JSON.stringify(event.messages||[],null,2)}];
    if(event.type==='model_response')return [{type:'model_request',status:'finished',seat:side,
      text:event.response??event.raw_output??'',reasoning:event.reasoning||'',
      finish_reason:event.finish_reason,failure_reason:event.failure_reason,response_tokens:event.response_tokens}];
    if(['student_move','opponent_move','forced_move'].includes(event.type)){
      const next=event.fen.split(' ')[1]==='w'?'white':'black';
      const position={type:'position',fen:event.fen,side:next,last_move:{uci:event.uci,seat:next==='white'?'black':'white'}};
      return event.type==='opponent_move'?[{type:'opponent_response',seat:position.last_move.seat,ply:event.ply,
        played_uci:event.uci,diagnostics:event.diagnostics||{}},position]:[position];
    }
    if(event.type==='fallback')return [{...event,seat:side}];
    if(event.type==='episode_finished')return [{type:'game_finished',summary:mapSummary(summary||{...event,completed:true},side)}];
    if(event.type==='episode_aborted')return [{type:'game_finished',summary:mapSummary({...event,aborted:true},side)}];
    return [];
  }
  async function readRecordPage(getPage,cursor){
    let length=131072,page,end;
    for(;;){
      page=await getPage(cursor,length);end=page.text.lastIndexOf('\n')+1;
      if(end||page.eof)break;
      if(length===524288)throw Error('单条事件超过524288字节，请使用原始日志查看');
      length=Math.min(524288,length*2);
    }
    // Only advance across complete UTF-8 JSONL records. A partial append is
    // retried on the next refresh, preserving the exact raw byte references.
    const text=page.text.slice(0,end),events=[];let offset=cursor;
    for(const line of text.split('\n').slice(0,-1)){
      const bytes=encoder.encode(line+'\n').length;
      if(line.trim())events.push({event:JSON.parse(line),offset,length:bytes});
      offset+=bytes;
    }
    return {events,cursor:offset,raw_cursor:offset,source:'events.jsonl',more:!page.eof&&offset>cursor};
  }
  async function readEventPage(getPage,cursor,side,summary){
    const raw=await readRecordPage(getPage,cursor);
    return {...raw,events:raw.events.flatMap(row=>mapEvent(row.event,side,summary).map(event=>({...row,event})))};
  }
  function emptyRandomMoves(){return {cursor:0,student_random_moves:0,opponent_random_moves:0,student_moves:0,opponent_moves:0,
    student_fallback_reasons:{},opponent_fallback_reasons:{},completed:false,aborted:false};}
  function tallyRandomMove(counts,event){
    let reasons,reason;
    if(event.type==='student_move'){
      counts.student_moves++;
      if(event.source==='random'){counts.student_random_moves++;reasons=counts.student_fallback_reasons;reason=event.fallback_reason||'unspecified';}
    }else if(event.type==='opponent_move'){
      counts.opponent_moves++;
      reason=event.diagnostics?.fallback_reason;
      if(reason){counts.opponent_random_moves++;reasons=counts.opponent_fallback_reasons;}
    }else if(event.type==='episode_finished')counts.completed=true;
    else if(event.type==='episode_aborted')counts.aborted=true;
    if(reasons)reasons[reason]=(reasons[reason]||0)+1;
  }
  function statistics(games,formal){
    const valid=games.filter(g=>g.completed&&!g.aborted&&[-1,0,1].includes(g.result));
    const counts={wins:valid.filter(g=>g.result===1).length,draws:valid.filter(g=>g.result===0).length,losses:valid.filter(g=>g.result===-1).length};
    if(formal?.status==='completed')return {...counts,completed:formal.games,wins:formal.wins,draws:formal.draws,losses:formal.losses,
      winRate:formal.win_rate,score:formal.score_rate,formal:true};
    return {...counts,completed:valid.length,winRate:valid.length?counts.wins/valid.length:null,
      score:valid.length?(counts.wins+.5*counts.draws)/valid.length:null,formal:false};
  }
  const helpers={mapSummary,mapEvent,readRecordPage,readEventPage,statistics,emptyRandomMoves,tallyRandomMove,opponentName};
  if(typeof module!=='undefined'&&module.exports)module.exports=helpers;
  globalThis.llmgoFinal=helpers;
  if(typeof document==='undefined')return;
  let discoveryPromise=null,discoveryTime=0;
  function mount(sectionId,prefix,local){
  const el=id=>document.getElementById(id.startsWith('final-')?prefix+id.slice(5):id),section=el(sectionId);if(!section)return;
  let selected='',busy=false,lastDiscovery=0,experiments=[],generation=0,discoveryWarnings=[];
  const cache=new Map(),moveCounts=new Map();
  const endpoint=run=>'api/grpo/runs/'+encodeURIComponent(run);
  async function get(url,optional=false){
    for(let attempt=0;attempt<2;attempt++){
      const response=await fetch(url,{cache:'no-store',signal:AbortSignal.timeout(15000)});
      if(optional&&response.status===404)return null;
      if([502,503,504].includes(response.status)&&!attempt)continue;
      if(!response.ok)throw Error('HTTP '+response.status);
      return response.json();
    }
  }
  async function object(run,files,path){
    const file=files.find(f=>f.path===path);if(!file)return null;
    const key=run+'/'+path,tag=file.modified+':'+file.bytes;
    if(cache.get(key)?.tag===tag)return cache.get(key).data;
    const page=await get(endpoint(run)+'/artifact?'+new URLSearchParams({path,length:524288}));
    if(!page.eof)throw Error(path+' 超过单次读取窗口');
    const data=JSON.parse(page.text);cache.set(key,{tag,data});return data;
  }
  async function randomMoves(run,file,seed){
    const key=run+'/'+file.path,tag=file.modified+':'+file.bytes;
    let entry=moveCounts.get(key);
    if(entry?.tag===tag)return entry.counts;
    let counts=entry?.counts||emptyRandomMoves();
    if(counts.cursor>file.bytes)counts=emptyRandomMoves();
    if(seed&&seed.cursor<=file.bytes&&seed.cursor>=counts.cursor)counts={...seed,
      student_fallback_reasons:{...seed.student_fallback_reasons},opponent_fallback_reasons:{...seed.opponent_fallback_reasons}};
    while(counts.cursor<file.bytes){
      const page=await readRecordPage((offset,length)=>get(endpoint(run)+'/artifact?'+new URLSearchParams({path:file.path,offset,length})),counts.cursor);
      if(page.cursor===counts.cursor)break;
      for(const row of page.events)tallyRandomMove(counts,row.event);
      counts.cursor=page.cursor;
    }
    moveCounts.set(key,{tag,counts});return counts;
  }
  async function parallel(items,fn){
    let index=0;const result=new Array(items.length);
    await Promise.all(Array.from({length:Math.min(4,items.length)},async()=>{
      while(index<items.length){const i=index++;result[i]=await fn(items[i]);}
    }));return result;
  }
  async function discover(){
    if(!discoveryPromise||Date.now()-discoveryTime>60000){
    discoveryTime=Date.now();
    discoveryPromise=(async()=>{
    const runs=await get('api/grpo/runs');
    const found=await parallel(runs,async run=>{
      try{
        const files=await get(endpoint(run.id)+'/files');
        const protocol=await object(run.id,files,'final-test/protocol.json');
        return protocol?{id:run.id,files,protocol}:null;
      }catch(error){return {id:run.id,error:error.message};}
    });
    return found;
    })();
    }
    const found=await discoveryPromise;
    discoveryWarnings=found.filter(row=>row?.error);
    const previousRuns=new Map(experiments.map(row=>[row.id,row]));
    experiments=found.flatMap(row=>!row?[]:row.error?(previousRuns.has(row.id)?[previousRuns.get(row.id)]:[]):Boolean(row.protocol.opponent?.kind==='local_model')===local?[row]:[]);
    lastDiscovery=Date.now();
    if(!experiments.length&&discoveryWarnings.length)throw Error('实验目录暂不可读，请稍后刷新');
    const previous=el('final-run').value;
    el('final-run').replaceChildren(...experiments.map(run=>new Option(run.id,run.id)));
    if(!el('final-latest').checked&&experiments.some(r=>r.id===previous))el('final-run').value=previous;
    else if(experiments.length)el('final-run').value=experiments[0].id;
  }
  const seat=side=>side==='white'?'白':'黑';
  const outcome=s=>!s?'进行中':s.result==='aborted'?'程序异常，不计胜负':s.result==='draw'?'和棋':s.student_win?'学生胜':'学生负';
  async function loadRun(run,version){
    let files=await get(endpoint(run)+'/files');
    // Older running Web servers list only 30 games per artifact pattern.
    // A disk-derived index exposes all 40 without restarting that service.
    const replayPage=await get(endpoint(run)+'/artifact?'+new URLSearchParams({path:'final-test/replay-index.json',length:524288}),true);
    if(replayPage){
      if(!replayPage.eof)throw Error('棋局索引未完整返回');
      files=Array.from(new Map([...files,...JSON.parse(replayPage.text).files].map(f=>[f.path,f])).values());
    }
    const paths=['final-test/protocol.json','final-test/progress.json','final-test/result.json','submission.json','progress.json','failure.json'];
    const [protocol,progress,formal,submission,outerProgress,failure]=await Promise.all(paths.map(p=>object(run,files,p)));
    const opponent=opponentName(protocol);
    const derivedPage=await get(endpoint(run)+'/artifact?'+new URLSearchParams({path:'final-test/random-moves.json',length:524288}),true);
    if(derivedPage&&!derivedPage.eof)throw Error('随机走子统计未完整返回');
    const derived=derivedPage?JSON.parse(derivedPage.text):null;
    const eventFiles=files.filter(f=>/^final-test\/games\/[^/]+\/events.jsonl$/.test(f.path));
    const rows=await parallel(eventFiles,async file=>{
      const id=file.path.split('/')[2],rawFolder='final-test/games/'+id;
      const summary=await object(run,files,rawFolder+'/summary.json');
      const random=await randomMoves(run,file,derived?.games?.[id]);
      const index=Number(id.match(/(\d+)$/)?.[1]),side=summary?.student_color||(index%2?'black':'white');
      const mapped=mapSummary(summary,side);
      if(mapped)mapped.student_win=summary.result===1;
      const game={id,folder:'final/'+run+'/'+id,rawFolder,apiBase:endpoint(run),model_seat:side,
        task_label:submission?.id||submission?.task_id||run,opponent_name:opponent,opponent_text:opponent+' non-thinking',summary:mapped,randomMoves:random};
      game.readEvents=cursor=>readEventPage((offset,length)=>get(game.apiBase+'/artifact?'+new URLSearchParams({path:file.path,offset,length})),cursor,side,summary);
      return {game,summary,index,random};
    });
    if(version!==generation||el('final-run').value!==run)return;
    rows.sort((a,b)=>a.index-b.index);
    const stats=statistics(rows.map(r=>r.summary).filter(Boolean),formal),total=protocol.games;
    const failed=Boolean(failure)||progress?.stage==='failed'||outerProgress?.stage==='failed';
    const last=progress?.updated||formal?.completed||protocol.started;
    const stamp=last?new Date(last).toLocaleString():'未记录';
    const studentRandom=rows.reduce((n,r)=>n+r.random.student_random_moves,0),opponentRandom=rows.reduce((n,r)=>n+r.random.opponent_random_moves,0);
    const stages={prepared:'已提交，等待任务启动',waiting_model_download:'等待对手模型下载与校验',
      loading_student:'加载学生模型',loading_opponent:'加载 '+opponent,playing:'对局进行中'};
    el('final-status').textContent=(stats.formal?'完整测试结果':failed?'测试异常，已完成局保留':local?(stages[progress?.stage||outerProgress?.stage]||'等待对局结果'):'暂时统计 · 尚未发布完整结果')+' · '+run+(discoveryWarnings.length?' · '+discoveryWarnings.length+'项历史目录暂不可读':'');
    el('final-status').className='grpo-message'+(failed?' error':'');
    el('final-progress').style.width=Math.min(100,100*stats.completed/total)+'%';
    el('final-detail').textContent='任务 '+(submission?.id||submission?.task_id||'未记录')+' · '+stats.completed+'/'+total+'局 · 最后落盘 '+stamp+' · 随机后备累计：学生 '+studentRandom+' / '+opponent+' '+opponentRandom+' · 每15秒读磁盘，不查询排程器。';
    const kpis=[['有效完成',stats.completed+'/'+total],['学生胜',stats.wins],['和棋',stats.draws],['学生负',stats.losses],['胜率',percent(stats.winRate)],['得分率',percent(stats.score)]];
    el('final-kpis').replaceChildren(...kpis.map(([label,value])=>{const card=document.createElement('div');card.className='grpo-kpi';const title=document.createElement('span');title.className='grpo-kpi-label';title.textContent=label;const number=document.createElement('strong');number.textContent=value;card.append(title,number);return card;}));
    el('final-protocol').textContent='学生：'+(protocol.student_protocol?.model_name||'—')+' · 模型 SHA：'+(protocol.student_model_identity?.sha256||'—')+'\n请求对手：'+(protocol.opponent?.model||'—')+' · '+(protocol.opponent?.base_url||'')+' · non-thinking\n棋步上限：'+protocol.max_plies+'半步 · 五次重复：'+(protocol.fivefold_repetition===true?'自动判和':protocol.fivefold_repetition===false?'关闭':'历史记录未注明')+'\n答案提取：'+(protocol.answer_parser||'历史版本，见冻结源；随机步按当时实际执行统计')+'\n白/黑：'+protocol.white+'/'+protocol.black+' · 最多并行：'+protocol.parallel_games+'\n只有完整结果标为正式；未完成测试的胜率仅基于已完成局，不合并不同重试。';
    const table=document.createElement('table'),head=document.createElement('tr');
    for(const title of ['对局','学生执色','结果','学生随机后备',opponent+'随机后备','半步','终局原因','棋盘']){const th=document.createElement('th');th.textContent=title;head.append(th);}table.append(head);
    for(const row of rows){const tr=document.createElement('tr');
      for(const value of [row.index+1,seat(row.game.model_seat),outcome(row.game.summary),row.random.student_random_moves,row.random.opponent_random_moves,row.summary?.plies??'—',row.summary?.reason||'—']){const td=document.createElement('td');td.textContent=value;tr.append(td);}
      const td=document.createElement('td'),button=document.createElement('button');button.textContent=row.summary?'复盘':'查看对局';
      button.onclick=()=>{if(globalThis.llmgoReplay)globalThis.llmgoReplay.openFinalGame(rows.map(r=>r.game),row.game.folder,sectionId);else el('final-status').textContent='棋盘脚本尚未就绪，请稍后重试';};td.append(button);tr.append(td);table.append(tr);
    }
    el('final-games').replaceChildren(table);
    if(globalThis.llmgoReplay)globalThis.llmgoReplay.setFinalGames(rows.map(r=>r.game),sectionId);
  }
  async function refresh(){
    if(busy||document.hidden||globalThis.llmgoReplay?.isPaused())return;busy=true;
    try{
      if(!experiments.length||Date.now()-lastDiscovery>60000)await discover();
      const run=el('final-run').value;
      if(!run){el('final-status').textContent=local?'暂无 Qwen3-8B-Chess 独立对战。':'暂无已启动的 DeepSeek 最终测试。';return;}
      if(run!==selected){selected=run;generation++;el('final-games').replaceChildren();}
      await loadRun(run,generation);
    }catch(error){el('final-status').textContent='测试记录暂不可读：'+error.message;el('final-status').className='grpo-message error';}
    finally{busy=false;if(selected!==el('final-run').value)refresh();}
  }
  el('final-run').onchange=()=>{el('final-latest').checked=false;generation++;refresh();};
  el('final-latest').onchange=()=>{lastDiscovery=0;refresh();};
  refresh();setInterval(refresh,15000);
  }
  mount('deepseek-panel','final',false);
  mount('qwen-chess-panel','local-final',true);
})();
