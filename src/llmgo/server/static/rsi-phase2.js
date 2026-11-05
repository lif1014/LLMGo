/* Phase 2 prompt candidates use candidate versions, not training steps. */
(() => {
  'use strict';
  const byId=id=>document.getElementById(id),panel=byId('phase2-panel');
  let busy=false;
  async function json(url){const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);return r.json();}
  const percent=value=>typeof value==='number'?(value*100).toFixed(1)+'%':'—';
  function draw(points){
    const svg=byId('phase2-curve'),ns='http://www.w3.org/2000/svg';svg.replaceChildren();
    const max=Math.max(1,...points.map(p=>p.round));
    const x=p=>40+570*p.round/max,y=v=>205-180*v;
    for(const value of [0,.5,1]){
      const line=document.createElementNS(ns,'line');
      Object.entries({x1:40,x2:610,y1:y(value),y2:y(value),stroke:'#cbd9d0'}).forEach(([k,v])=>line.setAttribute(k,v));svg.append(line);
      const label=document.createElementNS(ns,'text');label.setAttribute('x',0);label.setAttribute('y',y(value)+4);label.textContent=percent(value);svg.append(label);
    }
    for(const point of points.filter((p,i)=>i===0||i===points.length-1||i%Math.max(1,Math.ceil(points.length/6))===0)){
      const label=document.createElementNS(ns,'text');label.setAttribute('x',x(point));label.setAttribute('y',225);label.setAttribute('text-anchor','middle');label.textContent=point.round;svg.append(label);
    }
    for(const [color,get] of [['#26836a',p=>p.levels?.['2']?.score],['#7357a5',p=>p.levels?.['3']?.score],['#c18a33',p=>p.best_observed]]){
      const valid=points.filter(p=>typeof get(p)==='number');
      const line=document.createElementNS(ns,'polyline');line.setAttribute('points',valid.map(p=>`${x(p)},${y(get(p))}`).join(' '));line.setAttribute('fill','none');line.setAttribute('stroke',color);line.setAttribute('stroke-width',2);svg.append(line);
      for(const point of valid){const dot=document.createElementNS(ns,'circle');dot.setAttribute('cx',x(point));dot.setAttribute('cy',y(get(point)));dot.setAttribute('r',4);dot.setAttribute('fill',color);const title=document.createElementNS(ns,'title');title.textContent=point.candidate_id+' · '+percent(get(point));dot.append(title);svg.append(dot);}
    }
  }
  async function refresh(){
    const campaign=byId('campaign')?.value;if(!campaign||busy)return;busy=true;
    try{
      const base='api/rsi/campaigns/'+encodeURIComponent(campaign),overview=await json(base+'/overview');
      if(byId('campaign').value!==campaign)return;
      panel.hidden=overview.research_scope!=='phase2_prompt_only';if(panel.hidden)return;
      byId('phase2-origin').textContent='起始 checkpoint：'+(overview.model_origin?.model_path||'—')+' · 横轴为提示候选版本 · 绿=L2，紫=L3，金=观测最优';
      const final=overview.final_candidate;
      byId('phase2-delivery').textContent=final?'已交付 '+final.candidate_id+' · harness/skills：'+final.source.root+' · 模型：'+final.model_path:'尚未交付最终候选。原始分数可下降，观测最优不代表能力单调提升。';
      const page=await json(base+'/artifact?'+new URLSearchParams({path:'prompt-curve.json',length:524288}));
      if(byId('campaign').value!==campaign)return;
      if(!page.eof)throw Error('曲线超过显示窗口，请读取完整 prompt-curve.json');
      const data=JSON.parse(page.text);draw(data.points);
      const table=document.createElement('table'),head=document.createElement('tr');
      for(const title of ['候选','L2 胜/和/负','L2 得分率','L3 胜/和/负','L3 得分率','均分','观测最优','假设']){const cell=document.createElement('th');cell.textContent=title;head.append(cell);}table.append(head);
      for(const p of data.points){const row=document.createElement('tr'),l2=p.levels?.['2'],l3=p.levels?.['3'];
        for(const value of [p.candidate_id,l2?[l2.wins,l2.draws,l2.losses].join('/'):'—',percent(l2?.score),l3?[l3.wins,l3.draws,l3.losses].join('/'):'—',percent(l3?.score),percent(p.score),percent(p.best_observed),p.hypothesis]){const cell=document.createElement('td');cell.textContent=value;row.append(cell);}table.append(row);}
      byId('phase2-candidates').replaceChildren(table);
    }catch(error){if(!panel.hidden)byId('phase2-candidates').textContent=error.message.includes('404')?'等待初始基线。':error.message;}
    finally{busy=false;if(byId('campaign').value!==campaign)refresh();}
  }
  byId('campaign')?.addEventListener('change',()=>{panel.hidden=true;byId('phase2-curve').replaceChildren();byId('phase2-candidates').replaceChildren();refresh();});setInterval(refresh,15000);refresh();
})();
