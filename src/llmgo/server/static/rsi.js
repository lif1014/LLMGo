"use strict";
const el=id=>document.getElementById(id), freshStream=()=>({cursor:null,before:null,follow:true,request:0,recent:[]});
const state={campaign:"",generation:0,busy:false,paused:false,requests:new Set(),cards:new Map(),streams:{research:freshStream(),executor:freshStream()},
  evaluations:[],totalHours:0,evaluationRefresh:0,job:"",autoJob:true,game:"",autoGame:true,gameCursor:0,channels:{reasoning:"",content:""},raw:null};
const pieces={K:"♔",Q:"♕",R:"♖",B:"♗",N:"♘",P:"♙",k:"♚",q:"♛",r:"♜",b:"♝",n:"♞",p:"♟"};
const labels={BASELINE:"测量现任基线",WAITING:"等待计算任务",LEVEL_CHECK:"检查难度晋级",PLANNING:"强Agent研究与规划",TRAIN:"准备训练",
  CANDIDATE_EVAL:"准备候选评测",REVIEW:"比较候选与现任",FAILED:"任务失败，保留证据",NEEDS_INPUT:"等待缺失项处理"};
const show=v=>v==null?"未提供":String(v);
const date=v=>{if(v==null)return "未记录";const d=new Date(typeof v==="number"?v*1000:v);return Number.isNaN(d.getTime())?show(v):d.toLocaleString();};
const api=(suffix="")=>"/api/rsi/campaigns/"+encodeURIComponent(state.campaign)+suffix;
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
function board(fen){
  el("board").replaceChildren();fen.split(" ")[0].split("/").forEach((rank,row)=>{let col=0;
    for(const char of rank)for(let i=0,n=/[1-8]/.test(char)?Number(char):1;i<n;i++,col++){
      const d=document.createElement("div"),c=document.createElement("span");d.className="square "+((row+col)%2?"dark":"");d.textContent=pieces[char]||"";
      c.className="coordinate";c.textContent="abcdefgh"[col]+(8-row);d.append(c);el("board").append(d);}});
}
function curve(){
  const svg=el("curve"),ns="http://www.w3.org/2000/svg";svg.replaceChildren();
  const draw=(tag,attrs)=>{const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));svg.append(e);return e;};
  [0,.5,1].forEach(y=>{draw("line",{x1:50,x2:620,y1:225-y*200,y2:225-y*200,stroke:"#d4dfd7"});draw("text",{x:4,y:230-y*200,"font-size":12}).textContent=y*100+"%";});
  state.evaluations.forEach((e,i)=>{const x=state.totalHours?50+550*(e.worker_gpu_hours||0)/state.totalHours:50+550*i/Math.max(1,state.evaluations.length-1);
    const dot=draw("circle",{cx:x,cy:225-e.statistics.score*200,r:5,fill:"hsl("+((e.level*67)%360)+" 55% 40%)"}),t=document.createElementNS(ns,"title");
    t.textContent=e.candidate_id+" L"+e.level+": "+e.statistics.score;dot.append(t);});
  el("cost").textContent="已收集成功任务的Worker分配时长估算 "+state.totalHours.toFixed(3)+" GPU小时；不含排队/启动，不等于活跃计算或FLOPs。";
}
function results(rows){
  const signature=JSON.stringify(rows);if(el("results").dataset.signature===signature)return;
  el("results").dataset.signature=signature;clearFeed("results");
  for(const e of rows.slice(-80))detail("results",(e.candidate_id||"候选")+" · L"+show(e.level)+" · 得分率 "+
    (typeof e.statistics?.score==="number"?(e.statistics.score*100).toFixed(1)+"%":"未提供"),e);
}
function renderStatus(s,x){
  const m=s.meta||{},j=s.active_job||{},r=x.report||{},h=x.heartbeat||{};
  if(m.token_usage)usage(m.token_usage);
  el("overview").textContent=(labels[s.phase]||s.phase)+" · incumbent "+(s.incumbent?.id||"—")+" · 当前L"+s.current_level+" / 已通关L"+s.highest_cleared_level+" · 第"+s.round+"轮";
  el("meta-status").textContent="最近记录状态："+show(m.status)+"\n线程："+show(m.thread_id)+"\n轮次："+show(m.turn_id)+"\n工作目录："+show(m.cwd)+
    "\n最近状态时间："+date(m.updated)+"\n记录的进程："+(m.process?m.process.host+" / PID "+m.process.pid:"当前无已记录子进程");
  const left=r.next_check_at?Math.ceil(r.next_check_at-(s.server_time||Date.now()/1000)):null;
  el("executor-state").textContent="模式："+(x.mode||"embedded")+"\n任务："+(j.id||r.job_id||"—")+" / 平台ID "+show(r.task_id)+
    "\n平台："+show(r.platform_status)+" / 交接："+show(r.status)+"\n已检查："+show(r.checks)+"次 · 间隔 "+show(r.interval_seconds)+"秒\n下次检查："+date(r.next_check_at)+
    (left===null?"":"（"+(left>0?"约"+left+"秒后":"已到计划时间，等执行器处理")+"）")+"\n"+(r.summary||"等待Executor报告")+"\n最近报告："+date(r.updated)+"\n独立服务心跳："+date(h.updated);
  el("research-state").textContent="阶段："+(labels[s.phase]||s.phase)+"\n任务数："+s.job_count+" · 尝试数："+s.attempt_count+" · 扶正数："+s.accepted_count+
    "\n当前任务职责："+(j.role||"—")+" / 候选 "+(j.candidate_id||"—")+"\n更新："+date(s.updated)+(s.failure?"\n失败："+JSON.stringify(s.failure):"")+(s.needs_input?"\n待处理："+s.needs_input:"");
}
function options(id,rows,selected,label,value){
  const select=el(id),signature=JSON.stringify(rows.map(r=>[value(r),label(r)]));
  if(select.dataset.signature!==signature){select.replaceChildren();for(const row of rows)select.add(new Option(label(row),value(row)));select.dataset.signature=signature;}
  select.value=selected;
}
async function files(generation){
  if(!state.job)return;const job=state.job,data=await json(api("/job-files?job_id="+encodeURIComponent(job)));
  if(generation!==state.generation||job!==state.job)return;
  const p=data.progress;el("worker-state").textContent="查看任务："+job+"\n阶段："+(p?.stage||"尚无阶段记录")+"\n"+
    (p?.message||"旧版本/尚未启动的Worker可能没有progress.json，按需查看已有日志。")+"\n课程局面："+show(p?.curriculum_records)+
    (p?.backend?"\n训练："+p.backend+" · 批次："+show(p?.batch)+" · 采样："+show(p?.sampled)+" · 评分："+show(p?.labeled)+" · 更新："+show(p?.updates):"")+
    "\n评测已落盘："+data.game_results+"/"+data.formal_games+"局\n阶段时间："+date(p?.updated);
  el("file-status").textContent=data.files.length+"个已存在文件 · 只在点击时读正文";
  const signature=JSON.stringify(data.files.map(f=>[f.path,f.bytes]));if(el("job-files").dataset.signature===signature)return;
  el("job-files").dataset.signature=signature;el("job-files").replaceChildren();
  for(const file of data.files){const b=document.createElement("button");b.textContent=file.path.split("/").slice(-2).join("/")+" · "+(file.bytes/1024).toFixed(1)+"KiB";
    b.title=file.path+"\n更新："+date(file.modified);b.onclick=()=>openRaw({path:file.path,offset:0,length:file.bytes},true);el("job-files").append(b);}
}
async function student(generation){
  if(!state.job)return;const job=state.job,games=await json(api("/games?job_id="+encodeURIComponent(job)+"&limit=20"));
  if(generation!==state.generation||job!==state.job)return;
  if(state.autoGame&&games.length&&state.game!==games.at(-1).folder){state.game=games.at(-1).folder;state.gameCursor=0;state.channels={reasoning:"",content:""};el("thinking").textContent="";clearFeed("student-events");}
  options("game",games,state.game,g=>g.model_seat+" · "+(g.summary?.result||"进行中")+" · "+g.id,g=>g.folder);
  if(!state.game)return;const chosen=state.game,data=await json(api("/game-events?folder="+encodeURIComponent(chosen)+"&cursor="+state.gameCursor+"&limit=200"));
  if(generation!==state.generation||chosen!==state.game)return;state.gameCursor=data.cursor;
  for(const row of data.events){const e=row.event;
    if(e.type==="position"){board(e.fen);el("game-state").textContent="轮到"+e.side+" · "+(e.last_move?.uci||"初始局面");}
    if(e.type==="model_request"&&e.status==="started"){state.channels={reasoning:"",content:""};el("thinking").textContent="";}
    if(e.type==="token"){const c=e.channel==="reasoning"?"reasoning":"content";state.channels[c]=(e.replace?e.text:state.channels[c]+e.text).slice(-20000);el("thinking").textContent=state.channels.reasoning+"\n"+state.channels.content;}
    if(e.type==="model_request"&&e.status==="finished")el("thinking").textContent=(e.output||(e.reasoning||"")+"\n"+(e.text||"")).slice(-20000);
    if(["turn_started","choice","tool_call","model_request","fallback","game_finished","engine_search"].includes(e.type)||e.display_truncated)
      detail("student-events",e.type+" "+(e.status||"")+" "+(e.tool||e.reason||e.finish_reason||""),e,{...row,path:chosen+"/events.jsonl"});}
}
async function refresh(){
  if(state.busy||state.paused||document.hidden||!state.campaign)return;state.busy=true;const generation=state.generation;
  try{const s=await json(api("/overview"));if(generation!==state.generation)return;
    const x=await json(api("/executor"));if(generation!==state.generation)return;renderStatus(s,x);state.totalHours=s.worker_gpu_hours||0;
    const latest=s.active_job?.id;if(state.autoJob&&latest&&state.job!==latest){state.job=latest;state.game="";state.gameCursor=0;state.autoGame=true;}
    options("job",(s.recent_jobs||[]).slice().reverse(),state.job,j=>j.id+" · "+j.role+" · "+j.status+" · "+j.candidate_id,j=>j.id);
    for(const source of ["research","executor"])if(state.streams[source].follow)await readActivity(source);
    if(generation!==state.generation)return;
    if(Date.now()-state.evaluationRefresh>20000){const rows=await json(api("/evaluations"));if(generation!==state.generation)return;state.evaluations=rows;state.evaluationRefresh=Date.now();curve();results(rows);}
    await files(generation);await student(generation);
    if(generation===state.generation){el("connection").textContent="最近读取 "+new Date().toLocaleTimeString()+" · 3秒增量刷新";el("connection").className="";}
  }catch(e){if(generation===state.generation&&e.name!=="AbortError"){el("connection").textContent=e.message;el("connection").className="error";}}
  finally{state.busy=false;}
}
async function openRaw(row,tail=false){
  state.raw={...row,relative:tail?Math.max(0,row.length-65536):0,generation:state.generation,request:0};el("raw").showModal();await rawPage();
}
async function rawPage(){
  const raw=state.raw;if(!raw)return;const request=++raw.request;
  try{const length=Math.max(1,Math.min(65536,raw.length-raw.relative)),offset=raw.offset+raw.relative;
    const data=await json(api("/artifact?path="+encodeURIComponent(raw.path||"events.jsonl")+"&offset="+offset+"&length="+length));
    if(state.raw!==raw||raw.generation!==state.generation||request!==raw.request)return;
    el("raw-content").textContent=data.text;el("raw-position").textContent=(raw.path||"events.jsonl")+" · 字节 "+raw.relative+"–"+Math.min(raw.length,raw.relative+length)+" / "+raw.length+"（原文区间，可含交错事件）";
    el("raw-prev").disabled=raw.relative===0;el("raw-next").disabled=raw.relative+length>=raw.length;
  }catch(e){if(state.raw===raw&&raw.generation===state.generation)el("raw-content").textContent=e.message;}
}
function selectCampaign(){
  for(const request of state.requests)request.abort();state.generation++;state.campaign=el("campaign").value;
  state.streams={research:freshStream(),executor:freshStream()};state.job="";state.autoJob=true;state.game="";state.autoGame=true;state.gameCursor=0;state.evaluations=[];state.evaluationRefresh=0;state.raw=null;
  for(const id of ["coach","memory","results","student-events","executor-events"])clearFeed(id);
  el("results").dataset.signature="";el("job-files").dataset.signature="";el("job-files").replaceChildren();
  for(const id of ["thinking","coach-live","board","meta-usage"])el(id).replaceChildren();el("raw").close();refresh();
}
el("campaign").onchange=selectCampaign;
el("pause").onclick=()=>{state.paused=!state.paused;el("pause").textContent=state.paused?"继续刷新":"暂停刷新";if(!state.paused)refresh();};
el("coach-filter").onchange=()=>filterFeed("coach");el("executor-filter").onchange=()=>filterFeed("executor-events");
el("coach-older").onclick=()=>readActivity("research",true).catch(e=>el("connection").textContent=e.message);
el("executor-older").onclick=()=>readActivity("executor",true).catch(e=>el("connection").textContent=e.message);
el("coach-live-button").onclick=()=>returnLive("research");el("executor-live").onclick=()=>returnLive("executor");
el("job").onchange=()=>{state.job=el("job").value;state.autoJob=false;state.game="";state.gameCursor=0;state.autoGame=true;refresh();};
el("game").onchange=()=>{state.game=el("game").value;state.gameCursor=0;state.autoGame=false;el("thinking").textContent="";clearFeed("student-events");refresh();};
el("follow-game").onclick=()=>{state.autoJob=true;state.autoGame=true;refresh();};
el("close-raw").onclick=()=>{state.raw=null;el("raw").close();};
el("raw-prev").onclick=()=>{if(state.raw){state.raw.relative=Math.max(0,state.raw.relative-65536);rawPage();}};
el("raw-next").onclick=()=>{if(state.raw){state.raw.relative=Math.min(Math.max(0,state.raw.length-1),state.raw.relative+65536);rawPage();}};
async function list(){
  try{const rows=await json("/api/rsi/campaigns");options("campaign",rows,state.campaign,c=>c.campaign_id,c=>c.campaign_id);
    if(!state.campaign&&rows.length){el("campaign").value=rows.at(-1).campaign_id;selectCampaign();}else if(!rows.length)el("connection").textContent="尚无campaign";
  }catch(e){if(e.name!=="AbortError")el("connection").textContent=e.message;}
}
list();setInterval(refresh,3000);setInterval(()=>{if(!document.hidden&&!state.paused)list();},30000);
