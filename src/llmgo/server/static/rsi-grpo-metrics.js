/* GRPO metrics from shared-disk artifacts. No model calls or scheduler queries. */
(() => {
  'use strict';
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const valueOf = (row, keys) => {
    for (const key of keys) if (finite(row?.metrics?.[key])) return row.metrics[key];
    return null;
  };
  const number = value => !finite(value) ? '—' : value !== 0 && Math.abs(value) < .001 ? value.toExponential(2) : new Intl.NumberFormat('zh-CN', {maximumFractionDigits: 4}).format(value);
  const percent = value => finite(value) ? (value * 100).toFixed(1) + '%' : '—';
  const metric = (label, keys, color, format = number) => ({label, keys, color, format});
  const specs = [
    metric('平均 Reward', ['quality/reward_mean', 'critic/rewards/mean', 'critic/score/mean'], '#267451'),
    metric('270M 动作 Q 值', ['quality/q'], '#487da0'),
    metric('Reference KL', ['reference/kl_before_update', 'actor/kl_loss'], '#ad7842'),
    metric('策略 Entropy', ['entropy/token_mean_nats', 'actor/entropy'], '#8265a6'),
    metric('合法回答比例', ['quality/legal'], '#2b847b', percent),
    metric('格式正确比例', ['quality/format'], '#658348', percent),
    metric('平均回答 Tokens', ['quality/response_tokens_mean', 'response_length/mean'], '#5275a2'),
    metric('梯度范数', ['actor/grad_norm'], '#a86670'),
    metric('Policy Loss', ['actor/pg_loss'], '#526a92'),
    metric('零优势组比例', ['groups/zero_advantage_fraction'], '#a58143', percent)
  ];
  function cleanRow(raw) {
    if (!raw || !Number.isInteger(raw.step) || raw.step < 1 || !raw.metrics || typeof raw.metrics !== 'object') return null;
    const metrics = Object.fromEntries(Object.entries(raw.metrics).filter(([, value]) => finite(value)));
    return {step: raw.step, time: finite(raw.time) ? raw.time : 0, metrics};
  }
  function newCache() { return {offset: 0, carry: '', rows: new Map(), attempt: -1, checkpoint: 0, warnings: [], caughtUp: false}; }
  function activateAttempt(cache, attempt, checkpoint) {
    if (attempt <= cache.attempt) return;
    cache.attempt = attempt; cache.checkpoint = checkpoint;
    if (attempt > 0) for (const [step, old] of cache.rows) {
      if (step > checkpoint && (old.metrics['resume/attempt'] ?? 0) < attempt) cache.rows.delete(step);
    }
  }
  function upsert(cache, raw) {
    const row = cleanRow(raw); if (!row) return;
    const attempt = row.metrics['resume/attempt'] ?? 0;
    activateAttempt(cache, attempt, row.metrics['resume/checkpoint_step'] ?? 0);
    if (attempt < cache.attempt && row.step > cache.checkpoint) return;
    const previous = cache.rows.get(row.step);
    const previousAttempt = previous?.metrics['resume/attempt'] ?? 0;
    if (!previous || attempt > previousAttempt || attempt === previousAttempt && row.time >= previous.time) cache.rows.set(row.step, row);
  }
  function appendPage(cache, page) {
    if (page.offset !== cache.offset || !Number.isInteger(page.next_offset) || page.next_offset < cache.offset) throw Error('指标读取游标不一致');
    const parts = (cache.carry + page.text).split('\n'); cache.carry = parts.pop();
    for (const line of parts) {
      if (!line.trim()) continue;
      try { upsert(cache, JSON.parse(line)); }
      catch { if (!cache.warnings.includes('部分指标行无法解析，已保留其它记录')) cache.warnings.push('部分指标行无法解析，已保留其它记录'); }
    }
    cache.offset = page.next_offset; cache.caughtUp = page.eof;
  }
  function completedStep(rows, latest, receipts = []) {
    const verified = rows.filter(row => (row.metrics['update/mean_abs_logprob_change'] ?? 0) > 1e-9).map(row => row.step);
    const receiptsSteps = receipts.filter(row => row.optimizer_completed === true && row.verification_completed !== false && row.logprob_change > 1e-9).map(row => row.step);
    if (latest && (latest.metrics['update/mean_abs_logprob_change'] ?? 0) > 1e-9) verified.push(latest.step);
    return Math.max(0, ...verified, ...receiptsSteps);
  }
  function statusOf(job = {}, outer = {}, progress = {}, completed = 0, final = null) {
    if (job.status === 'Failed' || outer.stage === 'failed' || progress.stage === 'failed') return {label:'运行失败', tone:'error'};
    if (final?.training_loop_action === 'finish') return {label:'已交付模型', tone:'done'};
    if (job.status === 'Succeeded' || outer.stage === 'completed') return {label:'任务已完成', tone:'done'};
    const stages = {sampling:'生成采样', resampling_overlong_groups:'丢弃过长组并补样', sampling_completed:'采样完成', updating:'更新策略', verifying_update:'验证权重更新', saving_checkpoint:'保存 checkpoint', awaiting_meta_review:'等待 Meta-Agent 决策', validating_games:'L2 / L3 验证', exporting:'导出 checkpoint', selected:'已选定模型', starting:'初始化训练', loading:'加载模型', validating:'验证', update_completed:'本步更新完成'};
    if (stages[progress.stage]) return {label:stages[progress.stage], tone:'active'};
    if (progress.stage === 'training') return {label:'准备训练至 step ' + (progress.step ?? '—'), tone:'active'};
    if (job.status === 'Queued') return {label:'等待 GPU 排程', tone:'waiting'};
    if (job.status === 'PREPARED' || job.status === 'PREPARING') return {label:'等待 Executor 提交', tone:'waiting'};
    return {label:progress.stage ? '训练阶段 · ' + progress.stage : completed ? '已记录更新' : job.status === 'Running' ? '初始化 / 验证' : '等待训练记录', tone:'waiting'};
  }
  globalThis.LLMGoGrpoMetrics = {cleanRow, newCache, activateAttempt, upsert, appendPage, completedStep, statusOf, valueOf, specs};
  if (typeof document === 'undefined') return;

  const make = (tag, className, text) => { const item = document.createElement(tag); if (className) item.className = className; if (text != null) item.textContent = text; return item; };
  const section = make('section', 'grpo-dashboard'); section.id = 'grpo-metrics-panel'; section.setAttribute('aria-label', 'GRPO 训练指标');
  const heading = make('div', 'grpo-heading');
  const headingText = make('div'); headingText.append(make('p', 'grpo-eyebrow', 'REINFORCEMENT LEARNING'), make('h2', '', 'GRPO · 训练指标'), make('p', 'grpo-subtitle', '每次 RL 实验的真实更新、训练曲线与验证进度'));
  const toolbar = make('div', 'grpo-controls');
  const selectLabel = make('label', '', 'RL 实验 '), select = make('select'); select.setAttribute('aria-label', '选择 RL 实验'); selectLabel.append(select);
  const autoLabel = make('label', 'grpo-auto'), auto = make('input'); auto.type = 'checkbox'; auto.checked = true; autoLabel.append(auto, document.createTextNode('每 15 秒更新'));
  const refreshButton = make('button', '', '刷新'); refreshButton.type = 'button'; toolbar.append(selectLabel, autoLabel, refreshButton); heading.append(headingText, toolbar); section.append(heading);
  const banner = make('div', 'grpo-run-banner'), badge = make('span', 'grpo-badge', '正在读取'), runName = make('strong'), detail = make('span', 'grpo-run-detail'); banner.append(badge, runName, detail); section.append(banner);
  const progressTrack = make('div', 'grpo-progress-track'), progressFill = make('div', 'grpo-progress-fill'); progressTrack.append(progressFill); section.append(progressTrack);
  const progressText = make('p', 'grpo-progress-note'), message = make('p', 'grpo-message'); message.setAttribute('role', 'status'); section.append(progressText, message);
  const cards = make('div', 'grpo-kpis'), charts = make('div', 'grpo-charts'); section.append(cards, charts);
  const explorer = make('details', 'grpo-explorer'), explorerTitle = make('summary', '', '全部训练指标 · 选择任意标量');
  const explorerSelect = make('select'); explorerSelect.setAttribute('aria-label', '选择训练标量'); const explorerChart = make('div'); explorer.append(explorerTitle, explorerSelect, explorerChart); section.append(explorer);
  const history = make('details', 'grpo-history'), historyTitle = make('summary', '', '逐步记录'), historyWrap = make('div', 'matrix'), historyControls = make('div', 'grpo-history-controls');
  const previous = make('button', '', '较早 25 步'), next = make('button', '', '较新 25 步'), pageLabel = make('span', 'muted'); previous.type = next.type = 'button'; historyControls.append(previous, pageLabel, next); history.append(historyTitle, historyWrap, historyControls); section.append(history);
  const footer = make('p', 'grpo-footnote', '指标来自远端实验原始记录。训练 Reward 与实战得分分别展示；缺失值显示 —。'); section.append(footer);
  const main = document.querySelector('main'); if (!main) return; main.prepend(section);
  let campaign = '', index = new Map(), selected = '', epoch = 0, refreshing = false, historyPage = 0, lastRender = null;
  const caches = new Map(), runReads = new Map();
  const campaignId = () => document.getElementById('campaign')?.value || 'chess-grpo-v1';
  const base = new URL('api/grpo/', document.baseURI);
  async function get(url, optional = false) {
    const response = await fetch(url, {cache:'no-store', signal:AbortSignal.timeout(15000)});
    if (optional && response.status === 404) return null;
    if (!response.ok) throw Error('HTTP ' + response.status);
    return response.json();
  }
  const endpoint = (run, suffix) => new URL('runs/' + encodeURIComponent(run) + suffix, base);
  async function artifact(run, path, offset = 0, length = 131072) {
    return get(endpoint(run, '/artifact?' + new URLSearchParams({path, offset, length})), true);
  }
  async function object(run, path) {
    const page = await artifact(run, path, 0, 524288);
    if (!page) return null;
    if (!page.eof) throw Error(path + ' 超过显示读取上限');
    return JSON.parse(page.text);
  }
  async function readData(run) {
    if (runReads.has(run)) return runReads.get(run);
    const promise = (async () => {
      const cache = caches.get(run) || newCache(); caches.set(run, cache);
      const responses = await Promise.allSettled([
        object(run, 'grpo/recipe.json'), object(run, 'grpo/progress.json'), object(run, 'progress.json'),
        object(run, 'grpo/latest.json'), object(run, 'scaling/final.json'), get(endpoint(run, '/files'), true), object(run, 'grpo/resume.json')
      ]);
      const values = responses.map(result => result.status === 'fulfilled' ? result.value : null);
      const errors = responses.filter(result => result.status === 'rejected').map(result => result.reason.message);
      if (Number.isInteger(values[6]?.attempt) && Number.isInteger(values[6]?.step)) activateAttempt(cache, values[6].attempt, values[6].step);
      let metricsExists = false;
      try {
        for (let i = 0; i < 4; i++) {
          const page = await artifact(run, 'grpo/metrics.jsonl', cache.offset);
          if (!page) break;
          metricsExists = true; appendPage(cache, page);
          if (page.eof || !page.text) break;
        }
      } catch (error) { errors.push('指标流：' + error.message); }
      const latest = cleanRow(values[3]); if (latest) upsert(cache, latest);
      const receiptFiles = (values[5] || []).filter(file => /^grpo\/updates\/step-\d+\.json$/.test(file.path)).sort((a,b) => a.path.localeCompare(b.path));
      const receipts = [];
      // Older dashboards do not index metrics.jsonl/latest.json. Direct artifact
      // access above still works; committed update receipts are another fallback.
      if (!metricsExists && !latest && receiptFiles.length) {
        const found = await Promise.allSettled(receiptFiles.slice(-30).map(file => object(run, file.path)));
        for (const result of found) if (result.status === 'fulfilled' && result.value) {
          const row = result.value; receipts.push(row);
          if (row.optimizer_completed && row.logprob_change > 1e-9) upsert(cache, {...row, metrics:{...row.metrics, 'update/mean_abs_logprob_change':row.logprob_change}});
        }
      }
      return {recipe:values[0], progress:values[1] || {}, outer:values[2] || {}, latest, final:values[4],
        rows:[...cache.rows.values()].sort((a,b) => a.step-b.step), receipts, errors:[...errors, ...cache.warnings], cache};
    })();
    runReads.set(run, promise);
    try { return await promise; } finally { runReads.delete(run); }
  }
  const svgElement = (tag, attributes = {}) => { const item = document.createElementNS('http://www.w3.org/2000/svg', tag); for (const [key,value] of Object.entries(attributes)) item.setAttribute(key, value); return item; };
  function drawChart(target, definition, rows, compact = false) {
    const points = rows.map(row => ({step:row.step, value:valueOf(row, definition.keys)})).filter(point => finite(point.value));
    target.replaceChildren();
    const head = make('div', 'grpo-chart-heading'); head.append(make('h3', '', definition.label), make('strong', '', definition.format(points.at(-1)?.value))); target.append(head);
    const key = definition.keys.find(key => rows.some(row => finite(row.metrics[key]))); if (key) target.title = key;
    if (!points.length) { target.append(make('div', 'grpo-chart-empty', '等待首个已记录的训练 step')); return; }
    const width = 520, height = compact ? 150 : 184, left = 59, right = 16, top = 16, bottom = 29;
    const loStep = Math.min(...rows.map(row => row.step)), hiStep = Math.max(loStep + 1, ...rows.map(row => row.step));
    let lo = Math.min(...points.map(point => point.value)), hi = Math.max(...points.map(point => point.value));
    const pad = (hi - lo) * .14 || Math.max(Math.abs(hi) * .1, .01); lo -= pad; hi += pad;
    if (definition.format === percent) { lo = Math.max(0,lo); hi = Math.min(1,hi); }
    const x = step => left + (width-left-right) * (step-loStep)/(hiStep-loStep), y = value => top + (height-top-bottom) * (hi-value)/(hi-lo);
    const svg = svgElement('svg', {viewBox:`0 0 ${width} ${height}`, role:'img', 'aria-label':definition.label + '，横轴为 GRPO step'});
    for (const fraction of [0,.5,1]) {
      const value = lo+(hi-lo)*fraction, py = y(value);
      svg.append(svgElement('line', {x1:left, x2:width-right, y1:py, y2:py, stroke:'#dfe7e1', 'stroke-dasharray':'3 4'}));
      const label = svgElement('text', {x:left-8, y:py+4, 'text-anchor':'end'}); label.textContent = definition.format(value); svg.append(label);
    }
    const groups = []; let group = [];
    for (const row of rows) { const value = valueOf(row, definition.keys); if (finite(value)) group.push([x(row.step), y(value)]); else if (group.length) { groups.push(group); group = []; } }
    if (group.length) groups.push(group);
    for (const segment of groups) svg.append(svgElement('polyline', {points:segment.map(point => point.join(',')).join(' '), fill:'none', stroke:definition.color, 'stroke-width':2.2, 'stroke-linejoin':'round'}));
    const tooltip = make('p', 'grpo-chart-tooltip', '悬停或聚焦曲线点查看数值');
    for (const point of points) {
      const circle = svgElement('circle', {cx:x(point.step), cy:y(point.value), r:3.5, fill:definition.color, tabindex:0, 'aria-label':`step ${point.step}: ${definition.format(point.value)}`});
      const title = svgElement('title'); title.textContent = `step ${point.step} · ${definition.format(point.value)}`; circle.append(title);
      const show = () => { tooltip.textContent = title.textContent; }; circle.addEventListener('pointerenter', show); circle.addEventListener('focus', show); svg.append(circle);
    }
    for (const step of [loStep, hiStep]) { const label = svgElement('text', {x:x(step), y:height-8, 'text-anchor':step===loStep?'start':'end'}); label.textContent = 'step ' + step; svg.append(label); }
    target.append(svg, tooltip);
  }
  function renderHistory(rows) {
    const pages = Math.max(1, Math.ceil(rows.length / 25)); historyPage = Math.min(historyPage, pages-1);
    const end = rows.length-historyPage*25, shown = rows.slice(Math.max(0,end-25), end).reverse();
    const table = make('table'), header = make('tr');
    for (const label of ['GRPO step', 'Reward', 'Q', 'KL', 'Entropy', '合法率', '格式率', '回答 Tokens', 'Policy Loss', '梯度范数', '学习率']) header.append(make('th', '', label)); table.append(header);
    for (const row of shown) {
      const tr = make('tr'); const values = [row.step, ...[0,1,2,3,4,5,6,8,7].map(i => specs[i].format(valueOf(row,specs[i].keys))), number(valueOf(row,['actor/lr','actor/learning_rate']))];
      for (const value of values) tr.append(make('td', '', String(value))); table.append(tr);
    }
    historyWrap.replaceChildren(table); historyTitle.textContent = `逐步记录 · ${rows.length} 个 step`; pageLabel.textContent = `第 ${historyPage+1} / ${pages} 页`; previous.disabled = historyPage >= pages-1; next.disabled = historyPage === 0;
  }
  function render(run, data) {
    const meta = index.get(run) || {}, completed = completedStep(data.rows, data.latest, data.receipts), status = statusOf(meta.job, data.outer, data.progress, completed, data.final);
    badge.textContent = status.label; badge.className = 'grpo-badge ' + status.tone; runName.textContent = run;
    detail.textContent = meta.job?.task_id ? '平台任务 ' + meta.job.task_id : meta.job?.status === 'PREPARED' ? '尚无平台任务 ID' : '远端实验记录';
    const planned = data.recipe?.steps, phaseStep = data.progress.step, latest = data.rows.at(-1), total = finite(planned) ? planned : null;
    progressFill.style.width = total ? Math.min(100,completed/total*100)+'%' : '0%';
    progressText.textContent = `已核实完成 ${completed} 个 GRPO step` + (total ? ` / 配方上限 ${total}` : '') + (finite(phaseStep) ? ` · 当前阶段标记 step ${phaseStep}` : '') + ' · 完成数依据更新指标，不把准备阶段的目标 step 当作已完成';
    const update = latest?.time ? new Date(latest.time*1000).toLocaleString() : null;
    message.className = 'grpo-message' + (data.errors.length ? ' error' : '');
    message.textContent = data.errors.length ? '部分记录暂不可读；保留已载入数据。' + data.errors.join('；') : !data.rows.length ? status.tone === 'error' ? '此实验尚无已完成训练更新的指标。可在下方原始日志区查看错误。' : '尚无完成更新的指标；采样、初始化或排程期间显示 —，首步落盘后自动更新。' : `${data.rows.length} 个 step 已载入 · 最新训练记录 ${update || '未提供时间'}` + (!data.cache.caughtUp ? ' · 正在增量载入历史指标' : '');
    cards.replaceChildren();
    const cardDefs = [specs[0],specs[1],specs[2],specs[3],specs[4],specs[6]];
    for (const definition of cardDefs) {
      const card = make('div', 'grpo-kpi'), key = definition.keys.find(key => finite(latest?.metrics?.[key]));
      card.style.setProperty('--metric-color',definition.color); card.append(make('span', 'grpo-kpi-label',definition.label), make('strong','',definition.format(valueOf(latest,definition.keys))), make('span','grpo-kpi-foot',latest ? 'step ' + latest.step : '等待更新'));
      if (key) card.title = key; cards.append(card);
    }
    charts.replaceChildren();
    if (!data.rows.length) {
      const empty = make('article','grpo-chart grpo-empty-state');
      empty.append(make('div','grpo-empty-mark','↗'),make('h3','','等待训练更新'),make('p','','完成首个 GRPO step 后，这里会自动显示训练曲线。'),make('p','muted','初始化、采样与 GPU 排程不会被计为已完成更新。'));
      charts.append(empty);
    } else for (const definition of specs) {
      const chart = make('article','grpo-chart'); drawChart(chart,definition,data.rows,true); charts.append(chart);
    }
    const metricKeys = [...new Set(data.rows.flatMap(row => Object.keys(row.metrics)))].sort(), selectedKey = explorerSelect.value;
    explorerSelect.replaceChildren(...metricKeys.map(key => new Option(key,key)));
    if (metricKeys.includes(selectedKey)) explorerSelect.value = selectedKey;
    explorer.hidden = !metricKeys.length; history.hidden = !data.rows.length;
    renderHistory(data.rows); lastRender = {run,data}; renderExplorer();
    footer.textContent = '来源：var/runs/' + run + '/grpo/metrics.jsonl、latest.json 与更新收据。每一步采用最新记录；续训回退后的旧分支指标不冒充当前更新。Entropy 系数为 0 时仍可记录策略熵。L2/L3 实战结果见验证区。';
  }
  function renderExplorer() {
    if (!lastRender || !explorerSelect.value) return;
    drawChart(explorerChart, metric(explorerSelect.value,[explorerSelect.value],'#267451'), lastRender.data.rows);
  }
  async function loadSelected() {
    const run = selected, ticket = ++epoch;
    if (!run) { runName.textContent = ''; badge.textContent = '暂无 RL 实验'; cards.replaceChildren(); charts.replaceChildren(); history.hidden = explorer.hidden = true; progressFill.style.width = '0%'; progressText.textContent = ''; detail.textContent = ''; message.textContent = '训练提案登记后，这里会列出对应实验及其指标。'; return; }
    runName.textContent = run; badge.textContent = '读取指标'; badge.className = 'grpo-badge waiting'; message.textContent = '正在读取所选实验…';
    cards.replaceChildren(); charts.replaceChildren(); history.hidden = explorer.hidden = true; progressText.textContent = ''; progressFill.style.width = '0%'; detail.textContent = '';
    try { const data = await readData(run); if (ticket === epoch && selected === run) render(run,data); }
    catch (error) { if (ticket === epoch) { badge.textContent = '读取失败'; message.className = 'grpo-message error'; message.textContent = error.message; } }
  }
  async function refresh() {
    if (refreshing || document.hidden) return; refreshing = true;
    try {
      const current = campaignId();
      if (current !== campaign) { campaign = current; selected = ''; epoch++; lastRender = null; }
      const [runs, overview] = await Promise.all([get(new URL('runs',base)),get(new URL('api/rsi/campaigns/'+encodeURIComponent(campaign)+'/overview',document.baseURI))]);
      const entries = new Map();
      for (const run of runs) if (run.id.startsWith(campaign+'-') && (run.stage || run.trained || run.exported)) entries.set(run.id,{...run});
      for (const job of overview.recent_jobs || []) if (job.role === 'research') {
        const id = campaign+'-'+job.id; entries.set(id,{...entries.get(id),id,job});
      }
      const active = overview.active_job;
      if (active?.role === 'research') { const id = campaign+'-'+active.id; entries.set(id,{...entries.get(id),id,job:active}); }
      index = entries; const ids = [...entries.keys()].sort().reverse();
      const previous = selected;
      select.replaceChildren(...ids.map(id => { const row = entries.get(id); return new Option(id + (row.job?.status ? ' · ' + row.job.status : row.stage ? ' · ' + row.stage : ''),id); }));
      selected = ids.includes(previous) ? previous : ids[0] || ''; select.value = selected;
      if (previous !== selected) historyPage = 0;
      const ticket = ++epoch, run = selected;
      if (!run) await loadSelected();
      else { const data = await readData(run); if (ticket === epoch && selected === run) render(run,data); }
    } catch (error) { message.className = 'grpo-message error'; message.textContent = '读取暂时失败，保留上次数据：' + error.message; }
    finally { refreshing = false; }
  }
  select.addEventListener('change', () => { selected = select.value; historyPage = 0; loadSelected(); });
  refreshButton.addEventListener('click', refresh); explorerSelect.addEventListener('change',renderExplorer);
  previous.addEventListener('click',() => { historyPage++; if(lastRender) renderHistory(lastRender.data.rows); });
  next.addEventListener('click',() => { historyPage--; if(lastRender) renderHistory(lastRender.data.rows); });
  document.getElementById('campaign')?.addEventListener('change',refresh);
  refresh(); setInterval(() => { if(auto.checked) refresh(); },15000);
})();
