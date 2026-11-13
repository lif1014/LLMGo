/* One Meta panel for API and scheduled GPU execution. All data is read-only. */
(()=>{
'use strict';
const names={prepared:'准备请求',queued:'等待GPU排程',running:'执行中',loading_model:'加载模型',long_context_check:'长上下文验证',
 thinking:'思考中',generating:'生成回答',tool_execution:'执行工具',releasing_gpu:'交接完成，释放GPU',
 waiting_for_terminal:'交接已落盘，等待任务结束',waiting_for_artifact:'等待结果落盘',invalid_pending_restore:'隔离无效尝试的修改',completed:'工作完成',invalid:'无效尝试',failed:'执行失败',switching_pool:'切换资源池',idle:'等待研究／验证结果',paused:'研究暂停'};
const fmt=v=>typeof v==='number'?v.toFixed(3):'未记录';
function stage(row,a){
 if(!a)return names[row.status]||row.status||'等待结果';
 if(['completed','invalid','failed','switching_pool','invalid_pending_restore'].includes(a.status))return names[a.status];
 if(a.worker_stage==='completed'||a.worker_stage==='releasing_gpu')return names.waiting_for_terminal;
 if(a.worker_stage==='loading_model'||a.worker_stage==='long_context_check'||a.worker_stage==='failed')return names[a.worker_stage];
 if(a.worker_stage==='running')return names[a.activity]||names.running;
 if(a.status==='queued')return names.queued;
 return names[a.activity]||names[a.worker_stage]||names[a.status]||a.status;
}
function view(data){
 const current=data.current||{},a=current.attempts?.at(-1),gpu=current.execution==='gpu_job',config=current.inference||{};
 const settings=[config.meta_precision,config.meta_context_length?'Context '+config.meta_context_length:undefined,config.meta_max_output_tokens?'输出 '+config.meta_max_output_tokens:undefined,config.meta_reasoning_effort].filter(Boolean).join(' · ');
 return {title:(current.model||'模型未记录')+' · '+(gpu?'按需GPU':'API')+' · Thinking '+(current.thinking==null?'未记录':current.thinking?'开启':'关闭'),
   status:stage(current,a),gpu,attempt:a,settings,
   stages:gpu?['发布请求','排队','加载模型','思考／工具','完成交接','GPU任务结束']:['发布请求','模型请求','思考／工具','完成交接'],
   metaGpuHours:gpu?fmt(data.meta_gpu_hours)+' GPU-hours':'不适用（API推理）'};
}
if(typeof module!=='undefined'&&module.exports)module.exports={view,stage};
if(typeof document==='undefined')return;
const el=id=>document.getElementById(id);let campaign='',version=0,selected='',pending=false,latest=null;
function reset(){campaign='';version++;selected='';pending=false;latest=null;
 for(const id of ['meta-jobs-table','meta-job-cost','meta-job-stages'])el(id)?.replaceChildren();
 if(el('meta-jobs-status'))el('meta-jobs-status').textContent='正在读取所选研究…';
 if(el('meta-job-detail'))el('meta-job-detail').textContent='选择一项工作查看详情。';}
async function get(url){const r=await fetch(url,{cache:'no-store',signal:AbortSignal.timeout(15000)});
 if(!r.ok){const e=Error('HTTP '+r.status);e.status=r.status;throw e;}return r.json();}
function render(data){
 latest=data;
 const v=view(data);
 if(el('meta-status'))el('meta-status').textContent=v.title+'\n'+v.status+(v.attempt?.task_id?'\n平台任务：'+v.attempt.task_id+' / '+v.attempt.pool+' / '+v.attempt.priority:'');el('meta-jobs-status').textContent=v.title+' · '+v.status+(v.settings?' · '+v.settings:'');
 el('meta-job-stages').replaceChildren(...v.stages.map(s=>{const li=document.createElement('li');li.textContent=s;return li;}));
 const cost=document.createElement('p');cost.textContent='Meta佔卡：'+v.metaGpuHours+' · 最近最多30次工作（包含无效尝试）';
 const note=document.createElement('p');note.className='muted';note.textContent='学生训练与验证的GPU时间在GRPO／评测栏分别查看。API Token仅显示服务报告值；没有可核实数据时不估算费用。';
 el('meta-job-cost').replaceChildren(cost,note);
 const table=document.createElement('table'),head=document.createElement('tr');
 for(const label of ['工作／尝试','用途','模型／Thinking','执行方式','状态','平台任务／资源池','Meta GPU-hours','查看']){const th=document.createElement('th');th.textContent=label;head.append(th);}table.append(head);
 for(const row of data.requests||[]){
  for(const [i,a] of (row.attempts?.length?row.attempts:[null]).entries()){
   const tr=document.createElement('tr');if(a?.status==='invalid')tr.className='meta-invalid';
   const gpu=row.execution==='gpu_job';
   for(const value of [row.request_id.slice(0,10)+' / '+(i+1),row.kind==='review'?'Checkpoint审查':'研究方案',
     (row.model||'—')+' / '+(row.thinking==null?'未记录':row.thinking?'开启':'关闭'),gpu?'按需GPU':'API',stage(row,a),
     a?.task_id?a.task_id+' / '+a.pool+' / '+a.priority:'不适用',gpu?fmt(a?.gpu_hours):'不适用']){
    const td=document.createElement('td');td.textContent=value;tr.append(td);}
   const td=document.createElement('td'),button=document.createElement('button');button.textContent='详情';
   const key=row.request_id+'/'+i;
   button.onclick=()=>{selected=key;el('meta-job-detail').textContent=JSON.stringify({request:row,attempt:a},null,2);el('meta-job-detail').parentElement.open=true;};td.append(button);tr.append(td);table.append(tr);
   if(selected===key)el('meta-job-detail').textContent=JSON.stringify({request:row,attempt:a},null,2);
  }
 }
 el('meta-jobs-table').replaceChildren(table);
}
async function refresh(id,legacy){
 if(campaign!==id){reset();campaign=id;}if(pending)return;pending=true;const tag=version;
 const base='api/rsi/campaigns/'+encodeURIComponent(id);
 try{
  let data;
  try{data=await get(base+'/meta');}
  catch(e){if(e.status!==404)throw e;
   try{const p=await get(base+'/artifact?path=meta-overview.json&length=524288');if(!p.eof)throw Error('Meta索引过大');data=JSON.parse(p.text);}
   catch(f){if(f.status!==404)throw f;data={current:{execution:'api',status:legacy?.status,model:legacy?.model,thinking:legacy?.thinking,legacy:true},requests:[]};}}
  if(tag===version)render(data);
 }catch(e){if(tag===version)el('meta-jobs-status').textContent='Meta记录暂不可读：'+e.message;}
 finally{if(tag===version)pending=false;}
}
function session(){
 if(!latest)return {};
 const r=latest.current,a=r.attempts?.at(-1);
 return {model:r.model,thinking:r.thinking,execution:r.execution,status:stage(r,a),
 thread_id:a?.thread_id,updated:a?.last_activity,token_usage:a?.token_usage||r.token_usage,cwd:r.workspace};
}
globalThis.llmgoMetaPanel={reset,refresh,session};
})();
