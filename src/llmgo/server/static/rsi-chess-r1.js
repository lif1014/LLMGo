/* Read current GRPO experiments and their logs from LLMGo!/var/runs. */
(() => {
  'use strict';
  const section=document.createElement('section');
  section.id='chess-r1-panel';
  section.setAttribute('aria-label','Chess-R1 GRPO');
  const bar=document.createElement('div');bar.className='toolbar';section.append(bar);
  const title=document.createElement('h2');title.textContent='Chess-R1 · GRPO / 验证 / 最终测试';bar.append(title);
  const select=document.createElement('select');select.setAttribute('aria-label','选择实验');bar.append(select);
  const status=document.createElement('p');section.append(status);
  const metrics=document.createElement('div');metrics.className='matrix';section.append(metrics);
  const scaling=document.createElement('div');scaling.className='matrix';section.append(scaling);
  const gameLinks=document.createElement('div');section.append(gameLinks);
  const fileSelect=document.createElement('select');fileSelect.setAttribute('aria-label','选择实验文件');section.append(fileSelect);
  const content=document.createElement('pre');content.className='stream';section.append(content);
  const next=document.createElement('button');next.textContent='读取下一段';section.append(next);
  document.querySelector('main')?.append(section);
  let selected='',offset=0,loading=false;
  async function get(path){
    const reply=await fetch('api/grpo'+path,{cache:'no-store',signal:AbortSignal.timeout(15000)});
    if(!reply.ok)throw Error('HTTP '+reply.status);
    return reply.json();
  }
  async function loadFile(reset=true){
    if(!selected||!fileSelect.value||loading)return;
    loading=true;if(reset){offset=0;content.textContent='';}
    try{
      const path='/runs/'+encodeURIComponent(selected)+'/artifact?'+new URLSearchParams({path:fileSelect.value,offset,length:131072});
      const page=await get(path);content.textContent+=page.text;offset=page.next_offset;next.disabled=page.eof;
    }catch(error){status.textContent='文件读取失败：'+error.message;}
    finally{loading=false;}
  }
  async function loadRun(){
    if(selected!==select.value)gameLinks.replaceChildren();
    selected=select.value;
    if(!selected)return;
    try{
      const files=await get('/runs/'+encodeURIComponent(selected)+'/files');
      const previous=fileSelect.value;
      // Artifact access also works with an already-running older web process.
      try{
        const page=await get('/runs/'+encodeURIComponent(selected)+'/artifact?'+new URLSearchParams({path:'scaling/curve.json',length:524288}));
        if(!page.eof)throw Error('曲线超过单次读取上限，请读取完整 curve.json');
        const curve=JSON.parse(page.text);renderCurve([...(curve.ancestry||[]),...curve.points]);
        if(!files.some(row=>row.path==='scaling/curve.json'))files.push({path:'scaling/curve.json',bytes:page.next_offset});
      }catch(error){scaling.replaceChildren();if(!error.message.includes('404'))scaling.textContent=error.message;}
      fileSelect.replaceChildren(...files.map(row=>new Option(row.path+' · '+row.bytes+' bytes',row.path)));
      if(files.some(row=>row.path===previous))fileSelect.value=previous;
      else if(files.some(row=>row.path==='grpo/progress.json'))fileSelect.value='grpo/progress.json';
      metrics.replaceChildren();
      for(const path of ['grpo/progress.json','grpo/result.json','submission.json','final-test/result.json']){
        if(!files.some(row=>row.path===path))continue;
        try{
          const page=await get('/runs/'+encodeURIComponent(selected)+'/artifact?'+new URLSearchParams({path,length:32768}));
          if(!page.eof)continue;
          const card=document.createElement('div');const heading=document.createElement('strong');
          heading.textContent=path;card.append(heading);
          const pre=document.createElement('pre');pre.textContent=JSON.stringify(JSON.parse(page.text),null,2);card.append(pre);
          metrics.append(card);
        }catch(error){status.textContent='摘要读取失败：'+error.message;}
      }
      await loadFile();
    }catch(error){status.textContent='实验读取失败：'+error.message;}
  }
  function renderCurve(points){
    scaling.replaceChildren();if(!points.length)return;
    const note=document.createElement('p');note.textContent='L2 / L3 开发验证 · 按各档实际局数计算，保留历史批次 · 原始分数与观测最优（不表示能力单调提升）';scaling.append(note);
    const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');
    svg.setAttribute('viewBox','0 0 680 220');svg.setAttribute('role','img');svg.setAttribute('aria-label','各checkpoint的L2、L3和平均得分，横轴为累计optimizer step');svg.style.width='100%';svg.style.maxWidth='850px';
    const max=Math.max(1,...points.map(p=>p.lineage_step));
    const x=p=>40+620*p.lineage_step/max,y=v=>185-160*v;
    for(const v of [0,.5,1]){
      const line=document.createElementNS(ns,'path');line.setAttribute('d',`M40 ${y(v)}H660`);line.setAttribute('stroke','#888');line.setAttribute('opacity','.3');svg.append(line);
      const label=document.createElementNS(ns,'text');label.setAttribute('x','4');label.setAttribute('y',String(y(v)+4));label.setAttribute('fill','currentColor');label.textContent=(100*v)+'%';svg.append(label);
    }
    for(const [label,color,value,dash] of [['L2','#52c6a1',p=>p.levels.find(l=>l.level===2).statistics.score,false],['L3','#b398ff',p=>p.levels.find(l=>l.level===3).statistics.score,false],['平均','#66aaff',p=>p.score,false],['观测最优','#eeae55',p=>p.best_observed,true]]){
      const path=document.createElementNS(ns,'polyline');path.setAttribute('points',points.map(p=>`${x(p)},${y(value(p))}`).join(' '));path.setAttribute('fill','none');path.setAttribute('stroke',color);path.setAttribute('stroke-width','2');if(dash)path.setAttribute('stroke-dasharray','5 4');svg.append(path);
      const key=document.createElement('span');key.textContent=label+'  ';key.style.color=color;scaling.append(key);
    }
    scaling.append(svg);
    const table=document.createElement('table'),head=document.createElement('tr');
    for(const name of ['累计 step','L2 局数','L2 胜/和/负','L2 胜率','L2 得分率','L3 局数','L3 胜/和/负','L3 胜率','L3 得分率','平均得分率','观测最优','阶段 GPU h','checkpoint / 对局']){const th=document.createElement('th');th.textContent=name;head.append(th);}table.append(head);
    for(const p of points){
      const row=document.createElement('tr');
      const levelValues=[2,3].flatMap(level=>{
        const s=p.levels.find(r=>r.level===level)?.statistics;
        return s?[s.games,`${s.wins} / ${s.draws} / ${s.losses}`,s.games>0?(100*s.wins/s.games).toFixed(1)+'%':'—',(100*s.score).toFixed(1)+'%']:['—','—','—','—'];
      });
      for(const v of [p.lineage_step,...levelValues,(100*p.score).toFixed(1)+'%',(100*p.best_observed).toFixed(1)+'%',Number(p.lineage_gpu_hours||0).toFixed(2)]){const td=document.createElement('td');td.textContent=String(v);row.append(td);}
      const td=document.createElement('td');const model=document.createElement('code');model.textContent=p.model_path;td.append(model);
      for(const level of p.levels){
        const match=level.ref.path.match(/\/var\/runs\/([^/]+)\/(scaling\/.+)$/);if(!match)continue;
        const [,run,rel]=match,link=document.createElement('button');link.textContent='L'+level.level+' 对局记录';
        link.onclick=async()=>{
          await openArtifact(run,rel);gameLinks.replaceChildren();
          const page=await get('/runs/'+encodeURIComponent(run)+'/artifact?'+new URLSearchParams({path:rel,length:524288}));
          if(!page.eof)return;
          for(const game of JSON.parse(page.text).games||[]){
            const raw=game.raw_events?.match(/\/var\/runs\/([^/]+)\/(.+)$/);if(!raw)continue;
            const button=document.createElement('button');button.textContent=game.game_id+' · '+game.model_seat+' · '+game.result;
            button.onclick=()=>openArtifact(raw[1],raw[2]);gameLinks.append(button);
          }
        };td.append(link);
      }
      row.append(td);table.append(row);
    }scaling.append(table);
  }
  async function openArtifact(run,rel){
    if(select.value!==run){
      if(!Array.from(select.options).some(o=>o.value===run))select.add(new Option(run,run));
      select.value=run;await loadRun();
    }
    if(!Array.from(fileSelect.options).some(o=>o.value===rel))fileSelect.add(new Option(rel,rel));
    fileSelect.value=rel;await loadFile();
  }
  async function refresh(){
    try{
      const runs=await get('/runs');const previous=select.value;
      select.replaceChildren(...runs.map(row=>new Option(row.id+' · '+(row.final_tested?'DeepSeek 测试完成':row.stage||'等待记录'),row.id)));
      if(runs.some(row=>row.id===previous))select.value=previous;
      status.textContent=runs.length?'实验 '+runs.length+' 个 · '+new Date().toLocaleTimeString()+' 更新':'暂无实验；日志将写入 var/runs/<实验ID>/';
      if(select.value!==selected||select.value&&runs[0]?.id===select.value)await loadRun();
    }catch(error){status.textContent='实验索引暂不可读：'+error.message;}
  }
  select.onchange=loadRun;fileSelect.onchange=()=>loadFile();next.onclick=()=>loadFile(false);
  refresh();setInterval(()=>{if(!document.hidden)refresh();},15000);
})();
