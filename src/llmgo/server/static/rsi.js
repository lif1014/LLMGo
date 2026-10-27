"use strict";
const el=id=>document.getElementById(id), freshStream=()=>({cursor:null,before:null,follow:true,request:0,recent:[]});
const state={campaign:"",generation:0,busy:false,paused:false,requests:new Set(),cards:new Map(),streams:{research:freshStream(),executor:freshStream()},
  evaluations:[],totalHours:0,evaluationRefresh:0,maintenanceRefresh:0,job:"",autoJob:true,game:"",autoGame:true,viewer:null,raw:null,
  jobs:[],manifests:new Map(),fileData:null,games:[],gamesScope:null,archivedJobs:new Set(),archivesLoaded:false,activeJobId:null};
const pieces={K:"♔",Q:"♕",R:"♖",B:"♗",N:"♘",P:"♙",k:"♚",q:"♛",r:"♜",b:"♝",n:"♞",p:"♟"};
const labels={BASELINE:"固定多难度基线",WAITING:"等待计算任务",PLANNING:"强Agent自主研究",EXPERIMENT:"准备研究任务",
  EVALUATING:"固定评测",NEEDS_INPUT:"等待缺失项处理"};
const show=v=>v==null?"未提供":String(v);
const date=v=>{if(v==null)return "未记录";const d=new Date(typeof v==="number"?v*1000:v);return Number.isNaN(d.getTime())?show(v):d.toLocaleString();};
const apiRoot=new URL("api/rsi/campaigns",document.baseURI||"http://localhost/").pathname;
const api=(suffix="")=>apiRoot+"/"+encodeURIComponent(state.campaign)+suffix;
async function json(url){
  const c=new AbortController();state.requests.add(c);const timer=setTimeout(()=>c.abort(),15000);
  try{const r=await fetch(url,{signal:c.signal,cache:"no-store"});if(!r.ok)throw Error(r.status+": "+(await r.text()).slice(0,500));return await r.json();}
  finally{clearTimeout(timer);state.requests.delete(c);}
}
function clearFeed(target){el(target).replaceChildren();for(const [key,card] of state.cards)if(card.target===target)state.cards.delete(key);}
function filterFeed(target){
  if(!["coach","executor-events"].includes(target))return;
  const category=el(target==="coach"?"coach-filter":"executor-filter").value||"all";
  for(const node of el(target).children)node.hidden=category!=="all"&&node.dataset.category!==category;
}
function detail(target,title,value,raw=null,key=null){
  const d=document.createElement("details"),s=document.createElement("summary"),p=document.createElement("pre");
  const card={node:d,title:s,pre:p,entry:value,raw,target,count:1,text:value?.text||""};
  d.dataset.category=value?.category||"event";s.textContent=title;d.append(s,p);
  d.ontoggle=()=>{p.textContent=d.open?JSON.stringify({...card.entry,text:card.text},null,2).slice(0,24000):"";};
  if(raw){const b=document.createElement("button");b.textContent="读取原始事件";b.onclick=()=>openRaw(card.raw);d.append(b);}
  el(target).prepend(d);if(key)state.cards.set(key,card);
  while(el(target).children.length>80){const last=el(target).lastChild;for(const [k,c] of state.cards)if(c.node===last)state.cards.delete(k);last.remove();}
  return card;
}
function usage(value){const u=value.total||{},last=value.last||{};
    el("meta-usage").textContent="线程累计：输入 "+show(u.inputTokens)+" · 缓存命中 "+show(u.cachedInputTokens)+" · 输出 "+show(u.outputTokens)+
      " · 推理 "+show(u.reasoningOutputTokens)+" tokens；最近请求总计 "+show(last.totalTokens)+"。仅使用服务报告值。";
}
function consume(row,source){
  const e=row.event,target=source==="executor"?"executor-events":e.channel==="coach"?"coach":"memory";
  if(e.kind==="evaluation")state.evaluationRefresh=0;
  if(e.usage&&state.streams[source].follow)usage(e.usage);
  const merged=e.stream_key&&e.item_id||["heartbeat","usage"].includes(e.category);
  const key=merged?source+":"+(["heartbeat","usage"].includes(e.category)?e.category+e.thread_id:e.stream_key):null;
  let card=key&&state.cards.get(key);
  if(card){
    card.count++;card.entry=e;
    if(e.mode==="append")card.text=(card.text+(e.text||"")).slice(-20000);
    else if(e.text&&(!e.text_truncated||!card.text))card.text=e.text.slice(-20000);
    if(card.raw?.path===row.path){const start=Math.min(card.raw.offset,row.offset),end=Math.max(card.raw.offset+card.raw.length,row.offset+row.length);card.raw={...row,offset:start,length:end-start};}
    else card.raw=row;
    card.title.textContent=date(e.ts)+" · "+e.title+" · "+card.count+"条相关事件";
    card.node.dataset.category=e.category;if(card.node.open)card.node.ontoggle();
  }else card=detail(target,date(e.ts)+" · "+(e.title||e.kind),e,row,key);
  if(source==="research"&&e.channel==="coach"&&e.stream&&e.stream!=="tool"&&card.text)el("coach-live").textContent=card.text.slice(-20000);
  filterFeed(target);
}
async function readActivity(source,older=false){
  const generation=state.generation,s=state.streams[source],query=new URLSearchParams({source,limit:"80"});
  if(older&&!s.before)return;
  const request=++s.request;if(older)s.follow=false;
  if(older&&s.before)query.set("before",s.before);else if(!older&&s.cursor!==null)query.set("cursor",s.cursor);
  const data=await json(api("/activity?"+query));if(generation!==state.generation||s!==state.streams[source]||request!==s.request)return;
  const status=source==="research"?"coach-window":"executor-window";
  if(!data.available){el(status).textContent=data.note;return;}
  if(older){s.follow=false;for(const target of source==="research"?["coach","memory"]:["executor-events"])clearFeed(target);}
  for(const row of data.events)consume(row,source);
  if(older)s.before=data.older_cursor;
  else{if(data.events.length){s.recent=s.recent.concat(data.events.map(row=>row.ordinal)).slice(-80);s.before=s.recent[0];}s.cursor=data.cursor;}
  el(status).textContent=(s.follow?"实时":"历史页，自动追加已停")+" · 最近最多80行 · "+(data.more_new&&s.follow?"正在追赶新增事件 · ":"")+"原文完整保留";
  if(older||data.events.length)el(source==="research"?"coach-older":"executor-older").disabled=older?!data.has_older:!(s.before>1);
}
async function returnLive(source){
  state.streams[source]=freshStream();for(const target of source==="research"?["coach","memory"]:["executor-events"])clearFeed(target);
  try{await readActivity(source);}catch(e){el("connection").textContent=e.message;}
}
function board(fen,move){
  el("board").replaceChildren();fen.split(" ")[0].split("/").forEach((rank,row)=>{let col=0;
    for(const char of rank)for(let i=0,n=/[1-8]/.test(char)?Number(char):1;i<n;i++,col++){
      const d=document.createElement("div"),square="abcdefgh"[col]+(8-row);d.className="square "+((row+col)%2?"dark":"");d.textContent=pieces[char]||"";
      if(pieces[char])d.className+=char===char.toUpperCase()?" white-piece":" black-piece";
      if(move?.slice(0,2)===square)d.className+=" move-from";if(move?.slice(2,4)===square)d.className+=" move-to";
      d.title=square;el("board").append(d);}});
}
function curve(){
  const svg=el("curve"),ns="http://www.w3.org/2000/svg";svg.replaceChildren();
  const draw=(tag,attrs)=>{const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));svg.append(e);return e;};
  [0,.5,1].forEach(y=>{draw("line",{x1:50,x2:620,y1:225-y*200,y2:225-y*200,stroke:"#d4dfd7"});draw("text",{x:4,y:230-y*200,"font-size":12}).textContent=y*100+"%";});
  state.evaluations.forEach((e,i)=>{const x=state.totalHours?50+550*(e.worker_gpu_hours||0)/state.totalHours:50+550*i/Math.max(1,state.evaluations.length-1);
    const dot=draw("circle",{cx:x,cy:225-e.statistics.score*200,r:5,fill:"hsl("+((e.level*67)%360)+" 55% 40%)"}),t=document.createElementNS(ns,"title");
    t.textContent=e.candidate_id+" · "+(e.track||"player")+" L"+e.level+": "+e.statistics.score;dot.append(t);});
  el("cost").textContent="已收集成功任务的Worker分配时长估算 "+state.totalHours.toFixed(3)+" GPU小时；不含排队/启动，不等于活跃计算或FLOPs。";
}
function results(rows){
  const signature=JSON.stringify(rows);if(el("results").dataset.signature===signature)return;
  el("results").dataset.signature=signature;clearFeed("results");
  matrix(rows);
  for(const e of rows.slice(-80))detail("results",(e.candidate_id||"候选")+" · "+(e.track=== "model"?"模型对照":"完整玩家")+" · L"+show(e.level)+" · 得分率 "+
    (typeof e.statistics?.score==="number"?(e.statistics.score*100).toFixed(1)+"%":"未提供"),e);
}
function matrix(rows){
  const target=el("matrix");target.replaceChildren();
  const groups=new Map();
  for(const e of rows){const key=e.candidate_id+":"+(e.track||"player");if(!groups.has(key))groups.set(key,{id:e.candidate_id,track:e.track||"player",levels:{}});groups.get(key).levels[e.level]=e;}
  const table=document.createElement("table"),head=document.createElement("tr");
  for(const title of ["候选 / 评测",...Array.from({length:8},(_,i)=>"L"+(i+1)),"完整均分"]){const th=document.createElement("th");th.textContent=title;head.append(th);}table.append(head);
  for(const group of [...groups.values()].slice(-20)){
    const row=document.createElement("tr"),name=document.createElement("td");name.textContent=group.id+" / "+(group.track==="model"?"模型对照":"完整玩家");row.append(name);
    let total=0,count=0;
    for(let level=1;level<=8;level++){const cell=document.createElement("td"),r=group.levels[level],v=r?.statistics?.score;
      cell.textContent=typeof v==="number"?(v*100).toFixed(1)+"%":"—";
      if(typeof v==="number"){total+=v;count++;cell.title="胜/和/负 "+[r.statistics.wins,r.statistics.draws,r.statistics.losses].map(show).join("/")+" · "+(r.statistics.games??20)+"局";}row.append(cell);}
    const score=document.createElement("td");score.textContent=count===8?(total/8*100).toFixed(1)+"%":"部分测试 "+count+"/8";row.append(score);table.append(row);
  }target.append(table);
}
function renderStatus(s,x){
  const m=s.meta||{},j=s.active_job||{},r=x.report||{},h=x.heartbeat||{},recovery=s.controller_recovery;
  if(m.token_usage)usage(m.token_usage);
  el("overview").textContent=(labels[s.phase]||s.phase)+" · 工作版本 "+(s.working?.id||s.incumbent?.id||"—")+" · 实测最佳 "+(s.best?.candidate_id||"—")+
    (s.round===0?" · 初始基线（研究轮次0，不是难度0）":" · 第"+s.round+"轮研究")+(j.level!=null?" · 当前评测 L"+j.level+" / "+j.id:"");
  el("meta-status").textContent="最近记录状态："+show(m.status)+"\n线程："+show(m.thread_id)+"\n轮次："+show(m.turn_id)+"\n工作目录："+show(m.cwd)+
    "\n最近状态时间："+date(m.updated)+"\n记录的进程："+(m.process?m.process.host+" / PID "+m.process.pid:"当前无已记录子进程")+
    (s.research_control?.paused?"\n研究已暂停 · "+show(s.research_control.reason)+"\n既有任务继续监控；解除需 --resume-research":"")+
    (recovery?"\n控制器："+(recovery.status==="waiting"?"等待自动恢复":"暂停，需要处理")+" · "+recovery.stage+" / "+recovery.code+
      "\n失败次数："+recovery.attempt+" · 最多自动重试 "+recovery.max_retries+"次\n下一次重试："+date(recovery.next_retry_at)+"\n原因："+show(recovery.error):"");
  const left=r.next_check_at?Math.ceil(r.next_check_at-(s.server_time||Date.now()/1000)):null;
  el("executor-state").textContent="模式："+(x.mode||"embedded")+"\n任务："+(j.id||r.job_id||"—")+" / 平台ID "+show(r.task_id)+
    "\n平台："+show(r.platform_status)+" / 交接："+show(r.status)+"\n已检查："+show(r.checks)+"次 · 间隔 "+show(r.interval_seconds)+"秒\n下次检查："+date(r.next_check_at)+
    (left===null?"":"（"+(left>0?"约"+left+"秒后":"已到计划时间，等执行器处理")+"）")+"\n"+(r.summary||"等待Executor报告")+"\n最近报告："+date(r.updated)+"\n独立服务心跳："+date(h.updated);
  el("research-state").textContent="阶段："+(labels[s.phase]||s.phase)+"\n任务数："+s.job_count+" · 尝试数："+s.attempt_count+" · 完整评测新最佳："+s.accepted_count+
    "\n研究目标："+(s.candidate?.hypothesis||"由Meta-Agent自主决定")+"\n最佳跨难度得分："+(typeof s.best?.score==="number"?(s.best.score*100).toFixed(1)+"%":"尚无完整测试")+"\n推荐版本："+(s.recommended?.candidate_id||"尚未指定")+"\n当前任务职责："+(j.role||"—")+" / 候选 "+(j.candidate_id||"—")+"\n更新："+date(s.updated)+(s.failure?"\n失败："+JSON.stringify(s.failure):"")+(s.needs_input?"\n待处理："+s.needs_input:"");
}
function options(id,rows,selected,label,value){
  const select=el(id),signature=JSON.stringify(rows.map(r=>[value(r),label(r)]));
  if(select.dataset.signature!==signature){select.replaceChildren();for(const row of rows)select.add(new Option(label(row),value(row)));select.dataset.signature=signature;}
  select.value=selected;
}
const percent=value=>typeof value==="number"&&Number.isFinite(value)?(value*100).toFixed(1)+"%":"—";
const gameJob=game=>game.folder?.split("/")[1];
const roleName=role=>role==="baseline"?"初始基线":role==="candidate"?"研究候选":show(role);
function jobInfo(id){return state.jobs.find(j=>j.id===id)||{};}
function visibleJobs(){return state.jobs.filter(j=>!state.archivedJobs.has(j.id)||el("show-archived-jobs").checked||j.id===state.activeJobId);}
async function loadArchivedJobs(generation){
  if(state.archivesLoaded)return;
  try{
    const query=new URLSearchParams({path:"archived-jobs.json",offset:"0",length:"65536"});
    const data=await json(api("/artifact?"+query));
    if(generation!==state.generation)return;
    const index=JSON.parse(data.text);
    if(data.eof&&Array.isArray(index.jobs))state.archivedJobs=new Set(index.jobs.filter(id=>typeof id==="string"&&/^j\d+$/.test(id)));
    state.archivesLoaded=true;
  }catch(e){if(generation===state.generation&&e.name!=="AbortError")state.archivesLoaded=true;}
}
function jobLevel(id){return state.manifests.get(id)?.payload?.protocol?.opponent?.level??jobInfo(id).level;}
function difficulty(id){const level=jobLevel(id);return level==null?"难度未登记":"L"+level;}
function opponentText(id){
  const o=state.manifests.get(id)?.payload?.protocol?.opponent;
  return o?difficulty(id)+" · Fairy-Stockfish Skill "+show(o.skill)+" · depth "+show(o.depth)+" · 时间上限 "+show(o.movetime_sec)+"秒":difficulty(id);
}
async function loadJobManifest(id){
  if(!id||!/^[A-Za-z0-9_-]+$/.test(id))return;
  const cache=state.manifests,old=cache.get(id);
  if(old?.payload||old?.promise||old?.error&&Date.now()-old.time<30000)return old?.promise;
  const entry={time:Date.now()};cache.set(id,entry);
  if(cache.size>32)cache.delete(cache.keys().next().value);
  entry.promise=(async()=>{
    try{
      // Existing read-only endpoint; at most 4 x 64KiB, once per immutable job.
      let text="",offset=0,complete=false;
      for(let i=0;i<4;i++){
        const query=new URLSearchParams({path:"jobs/"+id+"/manifest.json",offset:String(offset),length:"65536"});
        const data=await json(api("/artifact?"+query));text+=data.text||"";
        if(data.eof){complete=true;break;}
        if(!(data.next_offset>offset))break;offset=data.next_offset;
      }
      if(!complete)throw Error("任务参数超过显示上限或未完整返回；可按需读取manifest");
      const wrapper=JSON.parse(text);
      if(!wrapper.payload||wrapper.payload.job_id!==id)throw Error("任务参数身份不匹配");
      entry.payload=wrapper.payload;
    }catch(e){if(e.name!=="AbortError")entry.error=e.message;}
    finally{entry.promise=null;}
  })();
  return entry.promise;
}
function studentResult(game){
  const s=game.summary;
  if(!s)return "进行中";
  if(["harness_error","engine_error","cancelled","aborted"].includes(s.result_reason)||s.result==="aborted")return "程序异常，不计胜负";
  if(!["white","black"].includes(game.model_seat))return "结果身份不完整";
  return s.result==="draw"?"和棋":s.result===game.model_seat+"_win"?"学生胜":s.result===(game.model_seat==="white"?"black_win":"white_win")?"学生负":"未完成";
}
function jobStats(id){
  const formal=state.evaluations.findLast(e=>e.ref?.job_id===id)?.statistics;
  const games=state.games.filter(g=>gameJob(g)===id);
  const covered=state.gamesScope==="campaign"||state.gamesScope===id;
  const counts={wins:0,draws:0,losses:0,errors:0,running:0,random:0,length:0};
  for(const game of games){
    const result=studentResult(game),s=game.summary;
    if(result==="学生胜")counts.wins++;else if(result==="和棋")counts.draws++;else if(result==="学生负")counts.losses++;
    else if(result==="进行中")counts.running++;else counts.errors++;
    counts.random+=s?.random_moves||0;counts.length+=s?.length_count||0;
  }
  const completed=formal?formal.games:counts.wins+counts.draws+counts.losses;
  return {...counts,...(formal?{wins:formal.wins,draws:formal.draws,losses:formal.losses}:{}),completed,
    available:Boolean(formal)||covered&&games.length>0,formal:Boolean(formal),
    winRate:completed?(formal?formal.wins:counts.wins)/completed:null,
    score:completed?(formal?formal.score:(counts.wins+.5*counts.draws)/completed):null};
}
function renderEvaluationJobs(){
  const target=el("evaluation-jobs"),rows=visibleJobs().slice().reverse().filter(j=>j.level!=null),table=document.createElement("table"),header=document.createElement("tr");
  const hidden=state.jobs.length-visibleJobs().length;el("archived-job-count").textContent=hidden?" 已隐藏"+hidden+"项旧任务；编号与提交凭据保留。":"";
  for(const title of ["任务 / 平台ID","难度","用途 / 候选","状态","有效完成 / 计划","胜 / 和 / 负","胜率","得分率","结果来源"]){const th=document.createElement("th");th.textContent=title;header.append(th);}table.append(header);
  for(const job of rows){
    const s=jobStats(job.id),manifest=state.manifests.get(job.id)?.payload,expected=manifest?.protocol?.games;
    const collected=state.evaluations.findLast(e=>e.ref?.job_id===job.id);
    const expectedText=expected??collected?.statistics?.games??(state.fileData?.job===job.id?state.fileData.data.formal_games:"?");
    const values=[job.id+" / "+show(job.task_id),difficulty(job.id),roleName(job.role)+" / "+show(job.candidate_id),show(job.status),
      (s.available?s.completed:"—")+" / "+expectedText,s.available?[s.wins,s.draws,s.losses].join(" / "):"—",s.available?percent(s.winRate):"—",s.available?percent(s.score):"—",
      s.formal?"正式已收集":s.available?"暂时统计" :"无可用对局记录"];
    const row=document.createElement("tr");if(job.id===state.job)row.className="selected";
    values.forEach((value,index)=>{const cell=document.createElement("td");if(index===0){const button=document.createElement("button");button.textContent=value;button.onclick=()=>selectJob(job.id);cell.append(button);}else cell.textContent=value;row.append(cell);});table.append(row);
  }
  const signature=JSON.stringify(rows.map(j=>[j,jobStats(j.id),state.manifests.get(j.id)?.payload?.protocol?.games]))+state.job+state.fileData?.data.formal_games;
  if(target.dataset.signature!==signature){target.dataset.signature=signature;target.replaceChildren(table);}
  el("evaluation-scope").textContent="最近最多30项评测；胜负以学生视角统计。"+(state.gamesScope==="campaign"?"对局范围：本研究最近1000局。":"对局范围：选中任务，其他任务只显示已收集的正式结果。")+"程序异常不计入分母；旧记录已清理时显示—，不表示胜率0%。";
  renderJobContext();
}
function renderJobContext(){
  const job=jobInfo(state.job),entry=state.manifests.get(state.job),p=entry?.payload,s=jobStats(state.job),g=p?.protocol;
  const expected=g?.games??(state.fileData?.job===state.job?state.fileData.data.formal_games:null);
  el("job-context").textContent="选中任务："+(state.job||"—")+" · 平台ID "+show(job.task_id)+" · "+roleName(job.role)+"\n"+
    opponentText(state.job)+"\n候选："+show(p?.candidate_id??job.candidate_id)+" · "+((p?.track??job.track)==="model"?"模型对照":"完整玩家（模型+Harness+skills）")+
    "\n计划："+show(expected)+"局"+(typeof expected==="number"?"，白黑各"+expected/2+"局":"")+" · 最多 "+show(g?.max_plies??200)+"个半回合（双方合计）"+
    "\n有效完成："+(s.available?s.completed:"—")+" · 胜/和/负："+(s.available?[s.wins,s.draws,s.losses].join("/"):"—")+" · 胜率 "+percent(s.winRate)+" · 得分率 "+percent(s.score)+
    "\n"+(s.formal?"结果已正式收集。":"暂时统计，整批完成并收集后才有正式成绩。")+
    "已读取记录中的程序异常："+(s.available?s.errors:"—")+" · 进行中："+(s.available?s.running:"—")+" · 随机后备："+(s.available?s.random:"—")+" · 输出达长度上限："+(s.available?s.length:"—")+
    (entry?.error?"\n对手参数读取失败："+entry.error:!p?"\n正在读取正式任务参数…":"\n参数来自本任务manifest；Skill和时间上限不是校准Elo，时间上限不是固定每步用时。");
}
function selectJob(id){state.job=id;state.autoJob=false;state.autoGame=true;el("game-scope").value="job";resetGame();refresh();}
async function files(generation){
  if(!state.job)return;const job=state.job,data=await json(api("/job-files?job_id="+encodeURIComponent(job)));
  if(generation!==state.generation||job!==state.job)return;
  state.fileData={job,data};await loadJobManifest(job);if(generation!==state.generation||job!==state.job)return;
  const p=data.progress;el("worker-state").textContent="查看任务："+job+" · "+opponentText(job)+"\n阶段："+(p?.stage||"尚无阶段记录")+"\n"+
    (p?.message||"旧版本/尚未启动的Worker可能没有progress.json，按需查看已有日志。")+"\n课程局面："+show(p?.curriculum_records)+
    (p?.backend?"\n训练："+p.backend+" · 批次："+show(p?.batch)+" · 采样："+show(p?.sampled)+" · 评分："+show(p?.labeled)+" · 更新："+show(p?.updates):"")+
    (p?.parallelism?"\n评测GPU副本："+p.parallelism+"\n"+(p.slots||[]).map(s=>"GPU "+s.gpu+" · "+s.stage+" · 棋局 "+show(s.game_index)+" · 尝试 "+s.attempt).join("\n"):"")+
    (p?.total_steps?"\nverl PPO："+p.step+"/"+p.total_steps+" · Actor GPU："+p.actor_gpus:"")+
    "\n评测已落盘："+data.game_results+"/"+data.formal_games+"局\n阶段时间："+date(p?.updated);
  el("file-status").textContent=data.files.length+"个已存在文件 · 只在点击时读正文";
  const signature=JSON.stringify(data.files.map(f=>[f.path,f.bytes]));if(el("job-files").dataset.signature===signature)return;
  el("job-files").dataset.signature=signature;el("job-files").replaceChildren();
  for(const file of data.files){const b=document.createElement("button");b.textContent=file.path.split("/").slice(-2).join("/")+" · "+(file.bytes/1024).toFixed(1)+"KiB";
    b.title=file.path+"\n更新："+date(file.modified);b.onclick=()=>openRaw({path:file.path,offset:0,length:file.bytes},true);el("job-files").append(b);}
}
const seatName=seat=>seat==="white"?"白方":seat==="black"?"黑方":"学生";
const blankTurn=()=>({prompt:"",reasoning:"",content:"",events:[],eventBytes:0,request:null});
function stopPlayback(r=state.viewer){if(r?.timer)clearInterval(r.timer);if(r)r.timer=null;el("replay-play").textContent="播放";}
function resetGame(){
  stopPlayback();state.viewer=null;state.game="";
  for(const id of ["board","thinking","student-output","student-prompt","moves"])el(id).replaceChildren();
  clearFeed("student-events");el("student-seat").textContent="学生";el("game-summary").textContent="尚未选择对局";
  el("game-state").textContent="此范围尚无对局，可选择历史任务查看";el("game-loading").textContent="";
  playbackControls();
}
function openGame(game){
  resetGame();state.game=game.folder;
  const r={folder:game.folder,generation:state.generation,info:game,frames:[],index:0,follow:true,timer:null,
    cursor:0,source:"compact",rawCursor:0,reading:false,loading:true,latest:blankTurn(),pending:null,summary:game.summary};
  state.viewer=r;el("game-loading").textContent="正在载入对局…";return r;
}
function excerpt(value){
  if(!value)return "";
  return (value.head||"")+(value.omitted?"\n[…省略 "+value.omitted+" "+(value.unit==="tokens"?"tokens":"字符")+"；可读取原文…]\n":"")+(value.tail||"");
}
function traceRow(r,row,source){
  const ref=row.event.raw_ref;
  return {...row,path:r.folder+"/"+(ref?"events.jsonl":source),offset:ref?.offset??row.offset,length:ref?.length??row.length};
}
function keepTrace(turn,row){
  if(turn.events.some(old=>old.path===row.path&&old.offset===row.offset))return;
  const cost=JSON.stringify(row.event).length;
  turn.events.push(row);turn.eventBytes+=cost;
  while(turn.events.length>40||turn.eventBytes>96000&&turn.events.length>1)turn.eventBytes-=JSON.stringify(turn.events.shift().event).length;
}
function ingestGame(r,row,source){
  const e=row.event,type=e.type,student=e.seat==null||e.seat===r.info.model_seat;
  if(type==="position"){
    if(!e.fen)return;
    const frame={position:e,view:r.pending||r.latest,label:e.last_move?seatName(e.last_move.seat)+" "+e.last_move.uci:"开局"};
    // Freeze the student's turn at its resulting position; engine moves reuse it.
    r.latest=frame.view;r.pending=null;r.frames.push(frame);
    const button=document.createElement("button"),index=r.frames.length-1;
    button.textContent=index===0?"开局":index+". "+frame.label;button.onclick=()=>showFrame(index);
    el("moves").append(button);return;
  }
  if(type==="game_finished"){r.summary=e.summary||r.summary;r.finished=e;return;}
  if(type==="engine_search")return;
  // The referee's raw stream is authoritative. Candidate-forwarded copies
  // remain on disk, but must not replace requests/tools in the bounded feed.
  if(type==="candidate_event"&&e.payload?.type==="token")return;
  if(type==="turn_started"&&student)r.pending=blankTurn();
  if(!student)return;
  if(!r.pending)r.pending=blankTurn();const turn=r.pending;
  if(e.prompt)turn.prompt=e.prompt.slice(0,20000);
  if(["model_request","request"].includes(type)){
    if(e.status==="started"){
      turn.request=e.request_id;turn.reasoning="";turn.content="";turn.compact=false;
      if(source==="short_events.jsonl")turn.inflightOffset=e.raw_ref?.offset;
    }else if(["finished","streaming"].includes(e.status)){
      if(e.status==="finished")turn.inflightOffset=null;
      if(e.reasoning)turn.reasoning=e.reasoning.slice(-20000);
      else if(e.output_excerpt)turn.reasoning=excerpt(e.output_excerpt).slice(-20000);
      else if(e.output&&!turn.reasoning)turn.reasoning=e.output.slice(-20000);
      if(e.text)turn.content=e.text.slice(-20000);
      turn.compact=source==="short_events.jsonl";
    }
  }
  if(type==="token"){
    if(turn.compact){turn.reasoning="";turn.content="";turn.compact=false;}
    const channel=e.channel==="reasoning"?"reasoning":"content";
    turn[channel]=(e.replace?e.text:turn[channel]+(e.text||"")).slice(-20000);
  }
  if(type==="choice"&&e.status==="answered"&&e.uci)turn.content=e.uci;
  if(type==="attempt"){
    if(e.reasoning&&!turn.reasoning)turn.reasoning=e.reasoning.slice(-20000);
    if(e.content)turn.content=e.content.slice(-20000);
  }
  if(["turn_started","choice","tool_call","model_request","request","fallback","attempt","candidate_event"].includes(type)||e.display_truncated)
    keepTrace(turn,traceRow(r,row,source));
}
function playbackControls(){
  const r=state.viewer,last=(r?.frames.length||0)-1;
  el("replay-prev").disabled=!r||r.index<=0;el("replay-next").disabled=!r||r.index>=last;
  el("replay-play").disabled=last<1;el("replay-restart").disabled=last<1;
  const range=el("replay-range");range.max=Math.max(0,last);range.value=r?.index||0;range.disabled=last<1;
  el("replay-label").textContent=last<0?"尚无可回放的走子":(r.follow?"直播":"复盘")+" · "+r.index+" / "+last+" 半步"+(r.loading?" · 历史仍在载入":"");
  for(const [index,node] of Array.from(el("moves").children).entries())node.setAttribute("aria-current",index===r?.index?"step":"false");
}
function renderGame(r){
  if(state.viewer!==r)return;
  const frame=r.frames[r.index],view=r.follow?(r.pending||r.latest):(frame?.view||blankTurn());
  if(frame){board(frame.position.fen,frame.position.last_move?.uci);
    el("game-state").textContent=(r.follow?"直播":"复盘")+" · "+frame.label+" · 轮到"+seatName(frame.position.side)+(frame.position.in_check?"，正在被将军":"");}
  el("student-seat").textContent="学生 · 执"+seatName(r.info.model_seat);
  el("student-prompt").textContent=view.prompt;el("thinking").textContent=view.reasoning;el("student-output").textContent=view.content;
  const signature=JSON.stringify(view.events.map(row=>[row.path,row.offset,row.length,row.event.status]));
  if(r.traceSignature!==signature){r.traceSignature=signature;clearFeed("student-events");
    for(const row of view.events){const e=row.event;detail("student-events",e.type+" "+(e.status||"")+" "+(e.tool||e.reason||e.finish_reason||""),e,row);}}
  const s=r.summary;
  el("game-summary").textContent="对局："+r.info.id+"\n任务："+gameJob(r.info)+" · "+opponentText(gameJob(r.info))+"\n学生执"+seatName(r.info.model_seat)+" · "+studentResult(r.info)+"\n"+(s?
    "结果："+show(s.result)+" / "+show(s.result_reason)+"\n模型选步："+show(s.model_decisions)+" · 强制步："+show(s.forced_moves)+
    "\n随机后备："+show(s.random_moves)+" · 请求："+show(s.request_count)+"\n生成："+show(s.completion_tokens)+" tokens"+"\n后备原因："+JSON.stringify(s.fallback_reasons||{}):"尚无最终结果；显示已记录的局面与请求");
  playbackControls();
}
function showFrame(index){
  const r=state.viewer;if(!r?.frames.length)return;stopPlayback(r);
  r.follow=false;state.autoGame=false;state.autoJob=false;r.index=Math.max(0,Math.min(index,r.frames.length-1));renderGame(r);
}
function playGame(fromStart=false){
  const r=state.viewer;if(!r||r.frames.length<2)return;if(r.timer&&!fromStart){stopPlayback(r);return;}
  showFrame(fromStart||r.index===r.frames.length-1?0:r.index);el("replay-play").textContent="暂停播放";
  r.timer=setInterval(()=>{
    if(state.viewer!==r||document.hidden||state.paused||r.index>=r.frames.length-1){stopPlayback(r);return;}
    r.index++;renderGame(r);if(r.index>=r.frames.length-1)stopPlayback(r);
  },700);
}
async function readGame(r){
  if(r.reading||state.viewer!==r||state.paused||document.hidden)return;r.reading=true;let more=false;
  try{
    for(let i=0;i<8;i++){
      const cursor=r.cursor,view=r.source==="compact"?"compact":"full";
      const data=await json(api("/game-events?folder="+encodeURIComponent(r.folder)+"&view="+view+"&cursor="+cursor+"&limit=200"));
      if(state.viewer!==r||r.generation!==state.generation)return;
      const source=data.source||"events.jsonl";
      for(const row of data.events)ingestGame(r,row,source);
      r.cursor=data.cursor;if(data.raw_cursor!=null)r.rawCursor=Math.max(r.rawCursor,data.raw_cursor);
      more=Boolean(data.more)&&r.cursor>cursor;
      if(r.follow)r.index=Math.max(0,r.frames.length-1);
      if(!more&&r.source==="compact"&&source==="short_events.jsonl"){
        // Compact logs can advance through candidate-forwarded token copies
        // during an unfinished request. Replay only that request's raw tail,
        // so opening the page mid-generation still recovers its visible text.
        r.source="events.jsonl";r.cursor=r.pending?.inflightOffset??r.rawCursor;more=true;
      }else if(source==="events.jsonl")r.source=source;
      if(!more){r.loading=false;break;}
      if(state.paused||document.hidden)break;
    }
    el("game-loading").textContent=r.loading?"正在载入历史…":r.finished?"对局已结束":"每3秒读取新事件";renderGame(r);
  }catch(e){more=false;if(state.viewer===r&&e.name!=="AbortError")el("game-loading").textContent="对局读取失败："+e.message;}
  finally{r.reading=false;}
  if(more&&state.viewer===r&&!state.paused&&!document.hidden)setTimeout(()=>readGame(r),0);
}
async function student(generation){
  const job=state.job,scope=el("game-scope").value||"campaign";
  const games=await json(api("/games?limit=1000"+(scope==="job"&&job?"&job_id="+encodeURIComponent(job):"")));
  if(generation!==state.generation||job!==state.job||scope!==(el("game-scope").value||"campaign"))return;
  const latest=games.filter(g=>g.folder.startsWith("jobs/"+job+"/")).at(-1)||games.at(-1);
  state.games=games;
  state.gamesScope=scope==="job"?job:"campaign";
  renderEvaluationJobs();
  if(state.autoGame&&latest&&state.game!==latest.folder||!state.viewer&&latest)openGame(latest);
  options("game",games,state.game,g=>difficulty(gameJob(g))+" · "+seatName(g.model_seat)+" · "+studentResult(g)+" · "+g.id,g=>g.folder);
  const r=state.viewer;if(!r)return;const info=games.find(g=>g.folder===r.folder);
  if(info){r.info=info;r.summary=info.summary||r.summary;}
  await loadJobManifest(gameJob(r.info));if(generation!==state.generation||state.viewer!==r)return;
  await readGame(r);
}
async function refresh(){
  if(state.busy||state.paused||document.hidden||!state.campaign)return;state.busy=true;const generation=state.generation;
  try{const s=await json(api("/overview"));if(generation!==state.generation)return;
    // Older Python Web services can serve this small JSON artifact without a
    // restart. Never read the full campaign or raw research journal on refresh.
    if(s.controller_status_version!==1){
      try{const page=await json(api("/artifact?path=controller-status.json&offset=0&length=16384"));
        const c=page.eof?JSON.parse(page.text):null;
        if(c?.schema_version===1&&c.campaign_id===state.campaign){
          s.latest_job??=s.active_job;s.active_job=s.active_job?.id===c.active_job?s.active_job:null;
          s.phase=c.phase;s.round=c.round;s.controller_recovery=c.controller_recovery;s.active_job_id=c.active_job;
          if(c.research_control)s.research_control=c.research_control;
        }
      }catch(e){/* Legacy campaigns have no compatibility artifact yet. */}
      if(!s.research_control){
        try{
          const page=await json(api("/artifact?"+new URLSearchParams({path:"research-control.json",offset:"0",length:"16384"})));
          if(page.eof)s.research_control=JSON.parse(page.text);
        }catch(e){/* An absent file means no operator hold. */}
      }
      if(generation!==state.generation)return;
    }
    const x=await json(api("/executor"));if(generation!==state.generation)return;state.jobs=s.recent_jobs||[];state.activeJobId=s.active_job?.id;await loadArchivedJobs(generation);if(generation!==state.generation)return;renderStatus(s,x);state.totalHours=s.worker_gpu_hours||0;
    const latest=s.active_job?.id||s.latest_job?.id;if(state.autoJob&&latest&&state.job!==latest){state.job=latest;resetGame();state.autoGame=true;}
    options("job",visibleJobs().slice().reverse(),state.job,j=>j.id+" · "+difficulty(j.id)+" · "+roleName(j.role)+" · "+j.status+" · "+j.candidate_id,j=>j.id);
    for(const source of ["research","executor"])if(state.streams[source].follow)await readActivity(source);
    if(generation!==state.generation)return;
    if(Date.now()-state.evaluationRefresh>20000){const rows=await json(api("/evaluations"));if(generation!==state.generation)return;state.evaluations=rows;state.evaluationRefresh=Date.now();curve();results(rows);}
    await files(generation);await student(generation);await maintenance(generation);
    if(generation===state.generation){el("connection").textContent="最近读取 "+new Date().toLocaleTimeString()+" · 3秒增量刷新";el("connection").className="";}
  }catch(e){if(generation===state.generation&&e.name!=="AbortError"){el("connection").textContent=e.message;el("connection").className="error";}}
  finally{state.busy=false;}
}
async function maintenance(generation){
  if(Date.now()-state.maintenanceRefresh<20000)return;state.maintenanceRefresh=Date.now();
  const read=async path=>{
    const page=await json(api("/artifact?"+new URLSearchParams({path,offset:"0",length:"32768"})));
    if(!page.eof)throw Error("维护摘要缺失或超过32KiB");return JSON.parse(page.text);
  };
  try{
    const plan=await read("maintenance/active.json");
    if(plan.schema_version!==1||!Array.isArray(plan.jobs)||plan.jobs.length>8)throw Error("维护任务清单格式无效");
    const rows=await Promise.all(plan.jobs.map(async job=>{
      if(!job.output?.startsWith("maintenance/")||job.output.split("/").includes(".."))throw Error("维护任务路径无效");
      let progress,health,gameProgress,evaluation=[];
      try{progress=await read(job.output+"/progress.json");}catch(e){return {job,pending:true};}
      try{health=await read(job.output+"/training/health.json");}catch(e){/* Evaluation/queued tasks have no PPO health. */}
      if(progress?.progress_path?.endsWith("game-progress.json")) {try{gameProgress=await read(job.output+"/training/game-progress.json");}catch(e){/* Bounded summary may still be pending. */}}
      if(job.id.startsWith("eval-"))evaluation=(await Promise.all((job.levels||[2,3]).filter(n=>Number.isInteger(n)&&n>=1&&n<=8).slice(0,8).map(async level=>{
        try{return {level,...await read(job.output+"/evaluation/L"+level+"/progress.json")};}catch(e){return null;}
      }))).filter(Boolean);
      return {job,progress,health,gameProgress,evaluation};
    }));
    if(generation!==state.generation)return;renderMaintenance(plan,rows);
  }catch(e){if(generation===state.generation)el("maintenance-state").textContent="当前未读取到有效维护清单："+e.message;}
}
function renderMaintenance(plan,rows){
  const lines=["验证："+show(plan.run_id)+" · 主研究暂停状态见上方面板"];
  if(plan.summary)lines.push(plan.summary);
  el("maintenance-files").replaceChildren();
  if(plan.comparison_path?.startsWith("maintenance/")&&!plan.comparison_path.split("/").includes("..")){
    const button=document.createElement("button");button.textContent="完整前后对照报告";button.onclick=()=>openRaw({path:plan.comparison_path,offset:0,length:65536,dynamic:true});el("maintenance-files").append(button);
  }
  for(const {job,progress:p,health:h,gameProgress:g,evaluation=[],pending} of rows){
    const v=h?.validation,q=h?.quality,d=h?.diagnostics;
    lines.push("\n"+job.id+" · "+show(job.label)+" · 平台ID "+show(job.task_id)+"\n阶段："+
      (pending?"排队/尚无Worker记录":show(p?.stage))+" / "+show(h?.stage)+" · 更新："+date(h?.updated??p?.updated));
    if(typeof p?.completed_games==="number")lines.push("完整对局："+p.completed_games+"/"+p.total_games+" · "+show(p.split));
    if(g?.games)for(const game of g.games.slice(0,32)){
      lines.push(game.episode_id+" · "+game.student_color+" · 半回合 "+game.plies+" · 学生决策 "+game.decisions+" · 随机后备 "+(game.random_moves||0)+" · "+(game.completed?"终局 "+game.result+" / "+game.reason:"进行中"));
      if(/^(train|validation)-[0-9]{8}-[0-9]{4}$/.test(game.episode_id))for(const name of ["summary.json","events.jsonl","trajectory.json"]){
        const button=document.createElement("button"),path=job.output+"/training/episodes/"+game.episode_id+"/"+name;
        button.textContent=game.episode_id+" · "+name;button.onclick=()=>openRaw({path,offset:0,length:65536,dynamic:true});el("maintenance-files").append(button);
      }
    }
    if(g?.omitted_games)lines.push("其余 "+g.omitted_games+" 局摘要保存在episode目录；此处仅显示最近32局。");
    if(p?.scenario)lines.push("采样探针场景："+p.scenario+" · 局面数 "+show(p.samples)+" · PPO更新 "+show(p.updates));
    if(typeof p?.formal_valid_rate==="number")lines.push("探针固定题集合法率："+percent(p.formal_valid_rate));
    for(const result of evaluation)lines.push("L"+result.level+"实战："+result.completed_games+"/"+result.total_games+"局 · "+show(result.stage)+" · 异常 "+(result.failures?.length||0));
    if(q)lines.push("本批请求合法率："+percent(q.valid_rate)+"（"+q.valid+"/"+q.samples+"）");
    if(d)lines.push("reference KL（每手）："+show(d.reference_kl_move_mean)+" · 警戒："+show(d.reference_kl_warning));
    if(v)lines.push(typeof v.score==="number"?"固定完整局验证 step "+v.step+"：合法率 "+percent(v.valid_rate)+" · 得分率 "+percent(v.score)+" · 胜/和/负 "+show(v.wins)+"/"+show(v.draws)+"/"+show(v.losses)+" · "+(v.passed?"通过":"未通过"):"固定验证 step "+v.step+"："+percent(v.valid_rate)+" · 历史单步TD均值 "+show(v.legal_td_mean)+" · "+(v.passed?"通过":"未通过"));
    if(typeof v?.random_moves==="number")lines.push("验证随机后备："+v.random_moves+"手 · 占比 "+percent(v.random_rate));
    if(typeof h?.metrics?.["game/random_moves"]==="number")lines.push("训练随机后备："+h.metrics["game/random_moves"]+"手 · 占比 "+percent(h.metrics["game/random_rate"]));
    if(h?.selected_checkpoint_step)lines.push("选用checkpoint：step "+h.selected_checkpoint_step+" · 提前停止："+show(h.stopped_early));
    if(h?.reason)lines.push("停止原因："+h.reason);
    for(const name of (job.id.startsWith("eval-")?["progress.json","research.log","evaluation/comparison.json"]:job.id.startsWith("pilot")?["progress.json","research.log","probe-summary.json","probe-result.json"]:["progress.json","research.log","training/health.json","training/stability-stop.json","training/checkpoint-selection.json","training/validation-latest.json","training/game-progress.json","training/identity.json"])){
      const button=document.createElement("button"),path=job.output+"/"+name;
      button.textContent=job.id+" · "+name;button.onclick=()=>openRaw({path,offset:0,length:65536,dynamic:true});el("maintenance-files").append(button);
    }
  }
  el("maintenance-state").textContent=lines.join("\n");
}
async function openRaw(row,tail=false){
  state.raw={...row,relative:tail?Math.max(0,row.length-65536):0,generation:state.generation,request:0};el("raw").showModal();await rawPage();
}
async function rawPage(){
  const raw=state.raw;if(!raw)return;const request=++raw.request;
  try{const length=Math.max(1,Math.min(65536,raw.length-raw.relative)),offset=raw.offset+raw.relative;
    const data=await json(api("/artifact?path="+encodeURIComponent(raw.path||"events.jsonl")+"&offset="+offset+"&length="+length));
    if(state.raw!==raw||raw.generation!==state.generation||request!==raw.request)return;
    if(raw.dynamic&&typeof data.next_offset==="number")raw.length=data.eof?data.next_offset-raw.offset:data.next_offset-raw.offset+65536;
    el("raw-content").textContent=data.text;el("raw-position").textContent=(raw.path||"events.jsonl")+" · 字节 "+raw.relative+"–"+Math.min(raw.length,raw.relative+length)+" / "+raw.length+"（原文区间，可含交错事件）";
    el("raw-prev").disabled=raw.relative===0;el("raw-next").disabled=raw.relative+length>=raw.length;
  }catch(e){if(state.raw===raw&&raw.generation===state.generation)el("raw-content").textContent=e.message;}
}
function selectCampaign(){
  for(const request of state.requests)request.abort();state.generation++;state.campaign=el("campaign").value;
  state.maintenanceRefresh=0;
  resetGame();state.games=[];state.jobs=[];state.manifests=new Map();state.fileData=null;state.gamesScope=null;state.streams={research:freshStream(),executor:freshStream()};state.job="";state.autoJob=true;state.autoGame=true;state.evaluations=[];state.evaluationRefresh=0;state.raw=null;
  state.archivedJobs=new Set();state.archivesLoaded=false;state.activeJobId=null;el("show-archived-jobs").checked=false;
  el("evaluation-jobs").replaceChildren();el("evaluation-jobs").dataset.signature="";el("job-context").textContent="正在读取任务参数…";el("evaluation-scope").textContent="";
  for(const id of ["coach","memory","results","student-events","executor-events"])clearFeed(id);
  el("results").dataset.signature="";el("job-files").dataset.signature="";el("job-files").replaceChildren();
  for(const id of ["thinking","coach-live","board","meta-usage"])el(id).replaceChildren();el("raw").close();refresh();
}
el("campaign").onchange=selectCampaign;
el("pause").onclick=()=>{state.paused=!state.paused;el("pause").textContent=state.paused?"继续刷新":"暂停刷新";if(state.paused)stopPlayback();else refresh();};
el("coach-filter").onchange=()=>filterFeed("coach");el("executor-filter").onchange=()=>filterFeed("executor-events");
el("coach-older").onclick=()=>readActivity("research",true).catch(e=>el("connection").textContent=e.message);
el("executor-older").onclick=()=>readActivity("executor",true).catch(e=>el("connection").textContent=e.message);
el("coach-live-button").onclick=()=>returnLive("research");el("executor-live").onclick=()=>returnLive("executor");
el("job").onchange=()=>selectJob(el("job").value);
el("show-archived-jobs").onchange=()=>{
  if(!el("show-archived-jobs").checked&&state.archivedJobs.has(state.job)&&state.job!==state.activeJobId){state.autoJob=true;state.job=state.activeJobId||visibleJobs().at(-1)?.id||"";resetGame();state.autoGame=true;}
  renderEvaluationJobs();refresh();
};
el("game-scope").onchange=()=>{state.autoJob=false;state.autoGame=false;resetGame();refresh();};
el("game").onchange=async()=>{const game=state.games?.find(g=>g.folder===el("game").value);if(!game)return;state.autoGame=false;state.autoJob=false;const r=openGame(game);await loadJobManifest(gameJob(game));if(state.viewer===r)readGame(r);};
el("follow-game").onclick=()=>{state.autoJob=true;state.autoGame=true;resetGame();refresh();};
el("replay-prev").onclick=()=>showFrame((state.viewer?.index||0)-1);
el("replay-next").onclick=()=>showFrame((state.viewer?.index||0)+1);
el("replay-play").onclick=()=>playGame();el("replay-restart").onclick=()=>playGame(true);
el("replay-range").oninput=()=>showFrame(Number(el("replay-range").value));
el("close-raw").onclick=()=>{state.raw=null;el("raw").close();};
el("raw-prev").onclick=()=>{if(state.raw){state.raw.relative=Math.max(0,state.raw.relative-65536);rawPage();}};
el("raw-next").onclick=()=>{if(state.raw){state.raw.relative=Math.min(Math.max(0,state.raw.length-1),state.raw.relative+65536);rawPage();}};
async function list(){
  try{const rows=await json(apiRoot);options("campaign",rows,state.campaign,c=>c.campaign_id,c=>c.campaign_id);
    if(!state.campaign&&rows.length){el("campaign").value=rows.at(-1).campaign_id;selectCampaign();}else if(!rows.length)el("connection").textContent="尚无campaign";
  }catch(e){if(e.name!=="AbortError")el("connection").textContent=e.message;}
}
list();setInterval(refresh,3000);setInterval(()=>{if(!document.hidden&&!state.paused)list();},30000);
