/* Checkpoint comparisons use a derived read-only catalog, never scheduler polling. */
(() => {
  'use strict';
  const labels={prepared:'待提交',Queued:'排队',Pending:'排队',Starting:'启动中',Running:'运行中',
    starting:'加载模型',playing:'对局中',completed:'已完成',failed:'失败',Failed:'失败',Killed:'已终止',
    Succeeded:'平台已完成',invalid_result:'结果身份不符'};
  function display(row){
    return {state:labels[row.stage]||row.stage||'待提交',
      progress:(row.completed_games||0)+'/'+(row.total_games||40),
      wdl:[row.wins??0,row.draws??0,row.losses??0].join(' / '),
      score:typeof row.score_rate==='number'?(100*row.score_rate).toFixed(1)+'%':'—',
      rating:row.formal&&typeof row.elo==='number'?String(row.elo)+(row.nominal_endpoint?' *':''):'—'};
  }
  if(typeof module!=='undefined'&&module.exports)module.exports={display};
  if(typeof document==='undefined')return;
  const el=id=>document.getElementById(id),section=el('elo-panel');if(!section)return;
  let busy=false;
  async function refresh(){
    if(busy||document.hidden||globalThis.llmgoReplay?.isPaused())return;
    busy=true;
    try{
      const url='api/grpo/runs/chess-checkpoint-benchmark/artifact?'+new URLSearchParams({path:'benchmark.json',length:524288});
      let response;
      for(let attempt=0;attempt<2;attempt++){
        response=await fetch(url,{cache:'no-store',signal:AbortSignal.timeout(15000)});
        if(![502,503,504].includes(response.status)||attempt)break;
      }
      if(response.status===404){el('elo-status').textContent='六项评测正在准备，等待远端任务回执。';return;}
      if(!response.ok)throw Error('HTTP '+response.status);
      const artifact=await response.json();if(!artifact.eof)throw Error('评测目录未完整返回');
      const data=JSON.parse(artifact.text);
      el('elo-status').textContent=(data.complete?'全部完成':'评测进行中')+' · '+data.formal_checkpoints+'/'+data.total_checkpoints+
        ' 个 checkpoint 已有正式结果 · 最后更新 '+new Date(data.updated).toLocaleString();
      el('elo-status').className='grpo-message';
      el('elo-protocol').textContent='Teacher：'+data.teacher.model+' · non-thinking · 固定 '+data.teacher.rating+
        '；每模型 '+data.games_per_checkpoint+' 局，白黑各半，每卡 '+data.parallel_games_per_gpu+' 局并行。'+
        '最后一个 boxed SAN；300 半步、五次重复判和；推论请求无客户端逾时。';
      const table=document.createElement('table'),head=document.createElement('tr');
      for(const title of ['Checkpoint','累计 RL step','任务','状态','有效局数','胜 / 和 / 负','得分率','表現分','棋局']){
        const th=document.createElement('th');th.textContent=title;head.append(th);
      }table.append(head);
      for(const row of data.entries){
        const shown=display(row),tr=document.createElement('tr');
        for(const value of [row.label,row.cumulative_steps,row.task_id||'—',shown.state,shown.progress,shown.wdl,shown.score,shown.rating]){
          const td=document.createElement('td');td.textContent=value;
          if(value===row.label)td.title=row.model+'\nSHA: '+row.expected_sha256;
          if(value===shown.state&&row.error)td.title=row.error;
          tr.append(td);
        }
        const td=document.createElement('td'),button=document.createElement('button');button.textContent='查看对局';
        button.disabled=!row.protocol_available;
        button.onclick=()=>{
          const select=el('final-run');
          if(!Array.from(select.options).some(o=>o.value===row.run_id))select.add(new Option(row.label,row.run_id));
          el('final-latest').checked=false;select.value=row.run_id;select.dispatchEvent(new Event('change'));
          el('deepseek-panel').scrollIntoView({behavior:'smooth'});
        };td.append(button);tr.append(td);table.append(tr);
      }
      el('elo-table').replaceChildren(table);
    }catch(error){el('elo-status').textContent='评测目录暂不可读：'+error.message;el('elo-status').className='grpo-message error';}
    finally{busy=false;}
  }
  refresh();setInterval(refresh,15000);
})();
