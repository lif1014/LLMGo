/* Read current GRPO experiments and their logs from LLMGo!/var/runs. */
(() => {
  'use strict';
  const section=document.createElement('section');
  section.id='chess-r1-panel';
  section.setAttribute('aria-label','Chess-R1 GRPO');
  const bar=document.createElement('div');bar.className='toolbar';section.append(bar);
  const title=document.createElement('h2');title.textContent='Chess-R1 · verl GRPO';bar.append(title);
  const select=document.createElement('select');select.setAttribute('aria-label','选择实验');bar.append(select);
  const status=document.createElement('p');section.append(status);
  const metrics=document.createElement('div');metrics.className='matrix';section.append(metrics);
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
    selected=select.value;
    if(!selected)return;
    try{
      const files=await get('/runs/'+encodeURIComponent(selected)+'/files');
      const previous=fileSelect.value;
      fileSelect.replaceChildren(...files.map(row=>new Option(row.path+' · '+row.bytes+' bytes',row.path)));
      if(files.some(row=>row.path===previous))fileSelect.value=previous;
      else if(files.some(row=>row.path==='grpo/progress.json'))fileSelect.value='grpo/progress.json';
      metrics.replaceChildren();
      for(const path of ['grpo/progress.json','grpo/result.json','submission.json']){
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
  async function refresh(){
    try{
      const runs=await get('/runs');const previous=select.value;
      select.replaceChildren(...runs.map(row=>new Option(row.id+' · '+(row.stage||'等待记录'),row.id)));
      if(runs.some(row=>row.id===previous))select.value=previous;
      status.textContent=runs.length?'实验 '+runs.length+' 个 · '+new Date().toLocaleTimeString()+' 更新':'暂无实验；日志将写入 var/runs/<实验ID>/';
      if(select.value!==selected||select.value&&runs[0]?.id===select.value)await loadRun();
    }catch(error){status.textContent='实验索引暂不可读：'+error.message;}
  }
  select.onchange=loadRun;fileSelect.onchange=()=>loadFile();next.onclick=()=>loadFile(false);
  refresh();setInterval(()=>{if(!document.hidden)refresh();},15000);
})();
