/* Independent engineering acceptance; never a formal study or a scheduler client. */
(()=>{
'use strict';
function rows(data){
 return Object.entries(data.models||{}).map(([key,r])=>{
  const b=r.benchmark||{},s=b.summary||{},n=s.schema_measured_cases;
  return {model:key==='qwen38'?'Qwen3.8-27B':'Qwen3.5-4B',
    task:r.task_id?`${r.task_id} / ${r.pool} / ${r.priority}`:'等待提交',
    stage:r.worker_stage||r.platform_status||r.status||'等待',
    progress:b.total?`${b.completed} / ${b.total}`:'尚未开始案例',
    passed:s.cases!=null?`${s.passed} / ${s.cases}`:'—',
    schema:n?`${s.first_check_schema_passes} / ${n}`:'—',
    context:r.long_context?.status||'未记录'};
 });
}
if(typeof module!=='undefined'&&module.exports)module.exports={rows};
if(typeof document==='undefined')return;
const el=id=>document.getElementById(id),base=new URL('api/grpo/runs/qwen38-meta-20261113/artifact',document.baseURI);
let pending=false;
async function read(path){
 const url=new URL(base);url.search=new URLSearchParams({path,length:'131072'});
 const response=await fetch(url,{cache:'no-store',signal:AbortSignal.timeout(15000)});
 if(!response.ok)throw Error('HTTP '+response.status);
 const page=await response.json();if(!page.eof)throw Error('报告过大');return JSON.parse(page.text);
}
async function refresh(){
 if(pending||document.hidden||!el('meta-acceptance-status'))return;pending=true;
 try{
  const data=await read('acceptance-controller.json');
  el('meta-acceptance-status').textContent='验收状态：'+data.stage+(data.error?' · '+data.error:'');
  const table=document.createElement('table'),head=document.createElement('tr');
  for(const text of ['模型','平台任务／卡池','阶段','案例进度','事实／任务通过','首次schema检查','长上下文']){const th=document.createElement('th');th.textContent=text;head.append(th);}table.append(head);
  for(const row of rows(data)){const tr=document.createElement('tr');for(const value of Object.values(row)){const td=document.createElement('td');td.textContent=value;tr.append(td);}table.append(tr);}
  el('meta-acceptance-table').replaceChildren(table);
  if(data.comparison){
   const c=data.comparison;
   el('meta-acceptance-result').textContent=`按案例配对：27B单独成功 ${c.right_only_success}，4B单独成功 ${c.left_only_success}；双侧 p=${c.exact_mcnemar_p_two_sided.toPrecision(3)}。`+
    (c.right_improved_at_p05?'本组工程案例有显著改善。':'本组结果尚未证明显著改善。');
  }else el('meta-acceptance-result').textContent='完整有效结果尚未齐备，暂不判断模型优劣。';
 }catch(error){el('meta-acceptance-status').textContent='验收记录暂不可读：'+error.message;}
 finally{pending=false;}
}
refresh();setInterval(refresh,15000);
})();
