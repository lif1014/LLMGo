/* Fixed-weight harness research: completed candidates and live development games. */
(() => {
  'use strict';
  const percent=v=>typeof v==='number'&&Number.isFinite(v)?(100*v).toFixed(1)+'%':'—';
  const levelOf=game=>game.id?.match(/-L(2|3)-/)?.[1];
  function levelStatistics(games,level,formal,planned=40){
    if(formal&&formal.games===planned&&formal.wins+formal.draws+formal.losses===planned){
      return {...formal,completed:formal.games,planned,formal:true};
    }
    let wins=0,draws=0,losses=0,errors=0;
    const seen=new Set();
    for(const game of games){
      if(levelOf(game)!==String(level)||seen.has(game.id))continue;seen.add(game.id);
      const s=game.summary;if(!s)continue;
      if(['aborted','harness_error','engine_error','cancelled'].includes(s.result)||
         ['aborted','harness_error','engine_error','cancelled'].includes(s.result_reason)){errors++;continue;}
      if(s.result==='draw')draws++;
      else if(s.result===game.model_seat+'_win')wins++;
      else if(['white_win','black_win'].includes(s.result))losses++;
    }
    const completed=wins+draws+losses;
    return {wins,draws,losses,completed,planned,errors,formal:false,
      score:completed?(wins+.5*draws)/completed:null};
  }
  function candidateRows(points,jobs){
    const rows=points.map(p=>({...p,job:jobs.filter(j=>j.candidate_id===p.candidate_id).at(-1)}));
    for(const job of jobs){
      const row=rows.find(p=>p.candidate_id===job.candidate_id);
      if(row)row.job=job;
      else rows.push({candidate_id:job.candidate_id,job,
        round:Number(job.candidate_id?.replace(/^r/,''))||0,levels:{},status:job.status});
    }
    return rows.sort((a,b)=>a.round-b.round);
  }
  const helpers={percent,levelStatistics,candidateRows};
  if(typeof module!=='undefined'&&module.exports)module.exports=helpers;
  if(typeof document==='undefined')return;
  const el=id=>document.getElementById(id),panel=el('phase2-panel');if(!panel)return;
  const make=(tag,cls='',text='')=>{const n=document.createElement(tag);n.className=cls;n.textContent=text;return n;};
  let studies=[],selected='',selectedJob='',busy=false,discovered=0,revision=0;
  const immutable=new Map();
  async function json(url,optional=false){
    for(let attempt=0;attempt<2;attempt++){
      const r=await fetch(url,{cache:'no-store',signal:AbortSignal.timeout(15000)});
      if(optional&&r.status===404)return null;
      if([502,503,504].includes(r.status)&&!attempt)continue;
      if(!r.ok)throw Error('HTTP '+r.status);return r.json();
    }
  }
  async function object(base,path,optional=false,cached=false){
    const key=base+'/'+path;if(cached&&immutable.has(key))return immutable.get(key);
    const page=await json(base+'/artifact?'+new URLSearchParams({path,length:524288}),optional);
    if(!page)return null;if(!page.eof)throw Error(path+' 超过显示窗口，请读取完整原始文件');
    const data=JSON.parse(page.text);if(cached)immutable.set(key,data);return data;
  }
  async function discover(){
    if(Date.now()-discovered<30000)return;
    const all=await json('api/rsi/campaigns');
    const found=await Promise.all(all.map(async c=>{
      const base='api/rsi/campaigns/'+encodeURIComponent(c.campaign_id),o=await json(base+'/overview');
      // Original RL web servers can load this UI without restarting Python.
      const curve=o.research_scope?null:await object(base,'prompt-curve.json',true);
      return o.research_scope==='phase2_prompt_only'||curve?.research_scope==='phase2_prompt_only'?{...c,overview:o}:null;
    }));
    studies=found.filter(Boolean).sort((a,b)=>(!a.overview.research_control?.paused?0:1)-(!b.overview.research_control?.paused?0:1)||
      String(b.updated).localeCompare(String(a.updated)));discovered=Date.now();
    if(!studies.some(s=>s.campaign_id===selected))selected=studies[0]?.campaign_id||'';
    const select=el('phase2-study');select.replaceChildren();
    for(const s of studies){const option=make('option','',s.campaign_id+(s.overview.research_control?.paused?' · 已暂停':''));option.value=s.campaign_id;select.append(option);}
    select.value=selected;
  }
  function draw(points){
    const svg=el('phase2-curve'),ns='http://www.w3.org/2000/svg';svg.replaceChildren();
    const max=Math.max(1,...points.map(p=>p.round)),x=p=>40+570*p.round/max,y=v=>205-180*v;
    const node=(tag,attrs)=>{const n=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);return n;};
    for(const value of [0,.5,1]){svg.append(node('line',{x1:40,x2:610,y1:y(value),y2:y(value),stroke:'#dce6df'}));
      const label=node('text',{x:0,y:y(value)+4});label.textContent=percent(value);svg.append(label);}
    for(const p of points.filter((p,i)=>i===0||i===points.length-1||i%Math.max(1,Math.ceil(points.length/6))===0)){
      const label=node('text',{x:x(p),y:225,'text-anchor':'middle'});label.textContent=p.round;svg.append(label);}
    for(const [color,get] of [['#26836a',p=>p.levels?.['2']?.score],['#7357a5',p=>p.levels?.['3']?.score],['#c18a33',p=>p.best_observed]]){
      const valid=points.filter(p=>typeof get(p)==='number');
      svg.append(node('polyline',{points:valid.map(p=>`${x(p)},${y(get(p))}`).join(' '),fill:'none',stroke:color,'stroke-width':2}));
      for(const p of valid){const dot=node('circle',{cx:x(p),cy:y(get(p)),r:4,fill:color}),title=node('title',{});
        title.textContent=p.candidate_id+' · '+percent(get(p));dot.append(title);svg.append(dot);}
    }
  }
  function replay(campaign,job,game){
    if(!globalThis.llmgoReplay?.openCampaignJob){el('phase2-status').textContent='对局面板正在载入，请稍后再试。';return;}
    globalThis.llmgoReplay.openCampaignJob(campaign,job,game).catch(e=>el('phase2-status').textContent=e.message);
  }
  function renderCards(stats,baseline){
    const box=el('phase2-levels');box.replaceChildren();
    for(const level of ['2','3']){
      const s=stats[level],card=make('div','harness-level');card.style.setProperty('--harness-color',level==='2'?'#26836a':'#7357a5');
      card.append(make('h3','',`L${level} · ${s.completed} / ${s.planned} 局`),make('strong','',percent(s.score)),
        make('p','',`${s.wins} 胜 · ${s.draws} 和 · ${s.losses} 负`),
        make('span','muted',s.formal?'完整评测结果':'已完成局的暂时得分率'));
      const old=baseline?.[level];if(old)card.append(make('p','harness-baseline',`原始 prompt 基线：${old.wins} 胜 / ${old.draws} 和 / ${old.losses} 负 · ${percent(old.score)}`));
      box.append(card);
    }
  }
  function renderRows(rows,stats,job){
    const table=make('table'),head=make('tr');
    for(const text of ['版本','状态','L2 胜 / 和 / 负','L2 得分率','L3 胜 / 和 / 负','L3 得分率','假设与查看'])head.append(make('th','',text));table.append(head);
    for(const p of rows){const row=make('tr'),isSelected=p.job?.id===job?.id;if(isSelected)row.className='harness-selected';
      const values=[p.candidate_id==='initial'?'0 · 原始 prompt':`${p.round} · ${p.candidate_id}`,p.job?.status||p.status||'完整评测'];
      for(const level of ['2','3']){const s=isSelected?stats[level]:p.levels?.[level];
        values.push(s?`${s.wins} / ${s.draws} / ${s.losses}`:'—',s?percent(s.score)+(s.formal===false?'（暂时）':''):'—');}
      for(const text of values)row.append(make('td','',text));
      const detail=make('td','harness-hypothesis',p.hypothesis||'候选评测中，方案见下方详情。');
      if(p.job){const button=make('button','',isSelected?'正在查看':'查看版本');button.onclick=()=>{
        selectedJob=p.job.id;el('phase2-follow').checked=false;revision++;refresh();};detail.append(button);}
      row.append(detail);table.append(row);
    }
    el('phase2-candidates').replaceChildren(table);
  }
  function renderGames(games,campaign,job){
    const table=make('table'),head=make('tr');for(const text of ['对局','学生执方','结果','随机后备','复盘'])head.append(make('th','',text));table.append(head);
    for(const game of games){const s=game.summary,result=!s?'进行中':s.result==='draw'?'和棋':s.result===game.model_seat+'_win'?'学生胜':
      ['white_win','black_win'].includes(s.result)?'学生负':'程序异常';
      const row=make('tr');for(const text of [game.id,game.model_seat==='white'?'白':'黑',result,s?.random_moves??'—'])row.append(make('td','',text));
      const cell=make('td'),button=make('button','','复盘');button.onclick=()=>replay(campaign,job.id,game);cell.append(button);row.append(cell);table.append(row);}
    el('phase2-games').replaceChildren(games.length?table:make('p','muted','任务开跑并写入对局后，这里自动显示。'));
  }
  async function refresh(){
    if(busy||document.hidden||globalThis.llmgoReplay?.isPaused())return;busy=true;const version=revision;
    try{
      await discover();panel.hidden=false;
      if(!selected){el('phase2-status').textContent='尚无 harness 研究；新建 Phase 2 后自动显示。';return;}
      const campaign=selected,base='api/rsi/campaigns/'+encodeURIComponent(campaign);
      const [o,curve]=await Promise.all([json(base+'/overview'),object(base,'prompt-curve.json',true)]);
      if(version!==revision||campaign!==selected)return;
      const points=curve?.points||[],jobs=(o.recent_jobs||[]).filter(j=>['baseline','candidate'].includes(j.role));
      for(const j of [o.active_job,o.latest_job])if(j&&!jobs.some(x=>x.id===j.id))jobs.push(j);
      const latest=o.active_job||o.latest_job;if(el('phase2-follow').checked||!jobs.some(j=>j.id===selectedJob))selectedJob=latest?.id||jobs.at(-1)?.id||'';
      const job=jobs.find(j=>j.id===selectedJob),point=points.find(p=>p.candidate_id===job?.candidate_id);
      el('phase2-origin').textContent='固定 checkpoint：'+(o.model_origin?.model_path||curve?.initial_model||o.working?.model_path||'—');
      el('phase2-badge').textContent=o.research_control?.paused?'研究已暂停':job?`${job.candidate_id} · ${job.status}`:o.phase;
      if(!job){el('phase2-status').textContent='正在准备初始基线；完成后开始 harness / skills 研究。';draw(points);return;}
      const [manifest,progress,games,formal2,formal3,proposal]=await Promise.all([
        object(base,`jobs/${job.id}/manifest.json`,false,true),object(base,`jobs/${job.id}/output/progress.json`,true),
        json(base+'/games?'+new URLSearchParams({job_id:job.id,limit:1000})),
        object(base,`jobs/${job.id}/output/evaluations/L2/metrics.json`,true,true),
        object(base,`jobs/${job.id}/output/evaluations/L3/metrics.json`,true,true),
        job.candidate_id==='initial'?null:object(base,`candidates/${job.candidate_id}/proposal.json`,true,true)]);
      if(version!==revision||campaign!==selected)return;
      const p=manifest.payload,stats={},levels=p.evaluation_panel||[];
      for(const level of ['2','3']){const planned=levels.find(x=>String(x.protocol?.opponent?.level)===level)?.protocol?.games||
        levels.find(x=>x.comparison_id?.includes('L'+level+':'))?.protocol?.games||point?.levels?.[level]?.games||40;
        stats[level]=levelStatistics(games,level,(level==='2'?formal2:formal3)?.statistics||point?.levels?.[level],planned);}
      const completed=stats['2'].completed+stats['3'].completed,total=stats['2'].planned+stats['3'].planned;
      el('phase2-status').textContent=`任务 ${job.task_id??'—'} · ${job.id} · 完成 ${completed} / ${total} 局 · ${p.resources?.gpu_count??1} GPU`+
        (o.research_control?.paused?' · 研究已暂停':'')+(progress?.failures?.length?` · ${progress.failures.length} 条异常记录`:'');
      el('phase2-progress').style.width=Math.min(100,100*completed/total)+'%';
      el('phase2-updated').textContent=(progress?.updated?'对局进度更新：'+new Date(progress.updated).toLocaleString('zh-CN',{timeZone:'Asia/Hong_Kong'})+'（香港时间） · ':'')+
        (stats['2'].formal&&stats['3'].formal?'两档完整结果已落盘。':'整批完成前仅显示暂时统计。');
      renderCards(stats,points.find(x=>x.candidate_id==='initial')?.levels||(o.best?.candidate_id==='initial'?o.best.panel?.player?.levels:null));
      const rows=candidateRows(points,jobs);for(const row of rows)if(row.candidate_id===job.candidate_id&&proposal)row.hypothesis=proposal.hypothesis;
      renderRows(rows,stats,job);draw(points);
      const files=Object.keys(p.candidate_source?.files||{}).filter(x=>x==='harness.py'||x==='skills.json'||x.startsWith('skills/'));
      el('phase2-plan').textContent=job.candidate_id==='initial'?'原始 Chess-R1 prompt；尚未加入 harness 或 skills。':
        [proposal?.hypothesis||point?.hypothesis||o.candidate?.hypothesis||'方案说明尚未落盘。',
          proposal?.research_notes?.intervention,files.length?'候选文件：\n'+files.join('\n'):'',
          proposal?.research_notes?'研究记录：\n'+JSON.stringify(proposal.research_notes,null,2):''].filter(Boolean).join('\n\n');
      el('phase2-replay').onclick=()=>replay(campaign,job.id);renderGames(games,campaign,job);
      const final=o.final_candidate;el('phase2-delivery').textContent=final?'已交付 '+final.candidate_id+' · '+final.model_path:
        '横轴为候选版本。绿=L2，紫=L3，金=两档均分的观测最佳；只绘制已收集结果。';
    }catch(error){panel.hidden=false;el('phase2-status').textContent='Harness 数据读取失败：'+error.message;}
    finally{busy=false;if(version!==revision)refresh();}
  }
  function reset(){selectedJob='';revision++;el('phase2-levels').replaceChildren();el('phase2-candidates').replaceChildren();
    el('phase2-games').replaceChildren();el('phase2-plan').textContent='正在读取方案…';el('phase2-progress').style.width='0%';draw([]);refresh();}
  el('phase2-study').onchange=()=>{selected=el('phase2-study').value;el('phase2-follow').checked=true;reset();};
  el('phase2-follow').onchange=()=>{revision++;refresh();};
  el('campaign')?.addEventListener('change',()=>{const value=el('campaign').value;if(studies.some(s=>s.campaign_id===value)){
    selected=value;el('phase2-study').value=value;reset();}});
  setInterval(refresh,15000);refresh();
})();
