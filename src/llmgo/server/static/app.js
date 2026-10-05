const FILES = "abcdefghi";
const RED_NUM = "九八七六五四三二一";
const CHAR = {
  red: { king: "帅", advisor: "仕", elephant: "相", horse: "马", rook: "车", cannon: "炮", pawn: "兵" },
  black: { king: "将", advisor: "士", elephant: "象", horse: "马", rook: "车", cannon: "炮", pawn: "卒" },
};
const ERRORS = {
  parse_fail: "无法解析",
  unchanged: "起点终点相同",
  empty_origin: "起点无子",
  not_own_piece: "不是己方棋子",
  illegal_geometry: "走法不合规则",
  king_face: "将帅照面",
  leaves_check: "走后被将军",
  request_error: "请求失败",
};
const SEAT = { red: "红方", black: "黑方" };

function initialPieces() {
  const back = ["rook", "horse", "elephant", "advisor", "king", "advisor", "elephant", "horse", "rook"];
  const pieces = [];
  back.forEach((kind, file) => {
    pieces.push({ file, rank: 0, color: "red", kind });
    pieces.push({ file, rank: 9, color: "black", kind });
  });
  [1, 7].forEach((file) => {
    pieces.push({ file, rank: 2, color: "red", kind: "cannon" });
    pieces.push({ file, rank: 7, color: "black", kind: "cannon" });
  });
  [0, 2, 4, 6, 8].forEach((file) => {
    pieces.push({ file, rank: 3, color: "red", kind: "pawn" });
    pieces.push({ file, rank: 6, color: "black", kind: "pawn" });
  });
  return pieces;
}

const state = {
  id: null,
  socket: null,
  pieces: initialPieces(),
  last: null,
  side: "red",
  attempts: [],
  live: {
    red: { prompt: "", reasoning: "", content: "" },
    black: { prompt: "", reasoning: "", content: "" },
  },
};

const board = document.getElementById("board");
const piecesEl = document.getElementById("pieces");
const marksEl = document.getElementById("marks");

function place(el, file, rank) {
  el.style.left = `${(file / 8) * 100}%`;
  el.style.top = `${((9 - rank) / 9) * 100}%`;
}

function drawLines() {
  const svg = document.getElementById("lines");
  const seg = [];
  const add = (x1, y1, x2, y2) => seg.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" />`);
  for (let rank = 0; rank < 10; rank += 1) add(0, 9 - rank, 8, 9 - rank);
  for (let file = 0; file < 9; file += 1) {
    add(file, 0, file, 4);
    add(file, 5, file, 9);
  }
  add(0, 4, 0, 5);
  add(8, 4, 8, 5);
  add(3, 9, 5, 7);
  add(5, 9, 3, 7);
  add(3, 2, 5, 0);
  add(5, 2, 3, 0);
  svg.innerHTML = seg.join("");
  const files = (html) => [...FILES].map((ch, i) => `<span style="left:${(i / 8) * 100}%">${html(ch, i)}</span>`).join("");
  document.getElementById("files-top").innerHTML = files((ch, i) => `${i + 1}<small>${ch}</small>`);
  document.getElementById("files-bottom").innerHTML = files((ch, i) => `${RED_NUM[i]}<small>${ch}</small>`);
  document.getElementById("ranks").innerHTML = Array.from({ length: 10 }, (_, i) => `<span style="top:${(i / 9) * 100}%">${9 - i}</span>`).join("");
}

function renderBoard() {
  piecesEl.innerHTML = "";
  marksEl.innerHTML = "";
  if (state.last) {
    for (const [cls, uci] of [["from", state.last.uci.slice(0, 2)], ["to", state.last.uci.slice(2)]]) {
      const mark = document.createElement("div");
      mark.className = `mark ${cls}`;
      place(mark, FILES.indexOf(uci[0]), Number(uci[1]));
      marksEl.appendChild(mark);
    }
  }
  for (const piece of state.pieces) {
    const el = document.createElement("div");
    el.className = `piece ${piece.color}`;
    el.innerHTML = `<span>${CHAR[piece.color][piece.kind]}</span>`;
    place(el, piece.file, piece.rank);
    piecesEl.appendChild(el);
  }
}

function inputText(event) {
  if (!event) return "";
  return event.llm_input || event.prompt || "";
}

function archive(seat) {
  const live = state.live[seat];
  if (!live.prompt && !live.reasoning && !live.content) return;
  const box = document.createElement("details");
  const title = document.createElement("summary");
  title.textContent = `${SEAT[seat]}先前一手的提示和输出`;
  const pre = document.createElement("pre");
  pre.textContent = `送给模型的提示\n${live.prompt}\n\n思考\n${live.reasoning}\n\n正式输出\n${live.content}`;
  box.append(title, pre);
  document.getElementById(`${seat}-archive`).prepend(box);
  live.prompt = "";
  live.reasoning = "";
  live.content = "";
  document.getElementById(`${seat}-prompt`).textContent = "";
  document.getElementById(`${seat}-reasoning`).textContent = "";
  document.getElementById(`${seat}-content`).textContent = "";
}

function setLive(seat) {
  document.getElementById(`${seat}-prompt`).textContent = state.live[seat].prompt;
  document.getElementById(`${seat}-reasoning`).textContent = state.live[seat].reasoning;
  document.getElementById(`${seat}-content`).textContent = state.live[seat].content;
}

const dirtyChannels = new Set();
let paintScheduled = false;

function schedulePaint(seat, channel) {
  dirtyChannels.add(`${seat}:${channel}`);
  if (paintScheduled) return;
  paintScheduled = true;
  requestAnimationFrame(() => {
    paintScheduled = false;
    for (const key of dirtyChannels) {
      const [paintedSeat, paintedChannel] = key.split(":");
      const node = document.getElementById(`${paintedSeat}-${paintedChannel}`);
      node.textContent = state.live[paintedSeat][paintedChannel];
      node.scrollTop = node.scrollHeight;
    }
    dirtyChannels.clear();
  });
}

function watchLive(gameId) {
  if (state.socket && state.id === gameId && state.socket.readyState <= 1) return;
  state.id = gameId;
  document.getElementById("start").disabled = true;
  document.getElementById("stop").disabled = false;
  document.getElementById("status").textContent = "正在接上进行中的对局…";
  if (state.socket) state.socket.close();
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/ws/games/${gameId}`);
  state.socket = socket;
  socket.onmessage = (message) => onEvent(JSON.parse(message.data));
  for (const button of document.querySelectorAll("#game-list button")) {
    button.classList.toggle("active", button.dataset.id === gameId);
  }
}

function addMove(attempt) {
  const row = document.createElement("div");
  row.className = attempt.ok ? "move" : "move bad";
  const label = attempt.ok
    ? `${SEAT[attempt.seat]} ${attempt.uci} ${attempt.zh || ""}`
    : `${SEAT[attempt.seat]} 第 ${attempt.attempt_index + 1} 次 ${ERRORS[attempt.error_type] || attempt.error_type}：${attempt.error_detail || ""}`;
  row.textContent = label;
  document.getElementById("moves").prepend(row);
}

function percent(rate) {
  if (rate === null || rate === undefined) return "还没有样本";
  return `${Math.round(rate * 1000) / 10}%`;
}

function bars(target, errors) {
  const entries = Object.entries(errors || {});
  if (!entries.length) {
    target.insertAdjacentHTML("beforeend", "<p>没有错误记录。</p>");
    return;
  }
  const max = Math.max(...entries.map(([, n]) => n));
  for (const [name, count] of entries) {
    const row = document.createElement("div");
    row.className = "bar";
    row.innerHTML = `<span>${ERRORS[name] || name}</span><i style="width:${(count / max) * 100}%"></i><b>${count}</b>`;
    target.appendChild(row);
  }
}

function renderSummary(node, summary, title) {
  node.innerHTML = "";
  if (!summary) {
    node.textContent = "还没有数据。";
    return;
  }
  const head = document.createElement("div");
  head.innerHTML = `<div>${title}</div><div class="rate">${percent(summary.first_try_legal_rate)}</div><div>首步合法 ${summary.first_try_legal || 0} / ${summary.first_try_total || 0}</div>`;
  if (summary.result) head.insertAdjacentHTML("beforeend", `<div>结果：${summary.result}（${summary.result_reason || ""}）</div>`);
  node.appendChild(head);
  for (const seat of ["red", "black"]) {
    const item = (summary.by_seat || {})[seat];
    if (!item) continue;
    const line = document.createElement("p");
    const rate = item.first_try_total ? item.first_try_legal / item.first_try_total : null;
    line.textContent = `${SEAT[seat]}首步合法率 ${percent(rate)}（${item.first_try_legal}/${item.first_try_total}）`;
    node.appendChild(line);
  }
  bars(node, summary.error_types);
  if (summary.finish_length_count) {
    const note = document.createElement("p");
    note.textContent = `有 ${summary.finish_length_count} 手碰到模型长度上限。页面和记录里的文字没有再被裁短。`;
    node.appendChild(note);
  }
}

function resetView() {
  state.attempts = [];
  state.pieces = initialPieces();
  state.last = null;
  state.side = "red";
  state.live = {
    red: { prompt: "", reasoning: "", content: "" },
    black: { prompt: "", reasoning: "", content: "" },
  };
  document.getElementById("moves").innerHTML = "";
  document.getElementById("red-archive").innerHTML = "";
  document.getElementById("black-archive").innerHTML = "";
  setLive("red");
  setLive("black");
  renderBoard();
}

function onEvent(event) {
  if (event.type === "replay_begin") {
    resetView();
    return;
  }
  if (event.type === "turn_started") {
    archive(event.seat);
    state.live[event.seat].prompt = inputText(event);
    setLive(event.seat);
    document.getElementById("status").textContent = `${SEAT[event.seat]}正在思考，第 ${event.attempt_index + 1} 次`;
    return;
  }
  if (event.type === "token") {
    state.live[event.seat][event.channel] += event.text;
    schedulePaint(event.seat, event.channel);
    return;
  }
  if (event.type === "attempt") {
    state.attempts.push(event);
    state.live[event.seat].prompt = inputText(event);
    setLive(event.seat);
    addMove(event);
    return;
  }
  if (event.type === "position") {
    state.pieces = event.pieces;
    state.side = event.side;
    state.last = event.last_move;
    renderBoard();
    document.getElementById("status").textContent = event.in_check ? `轮到${SEAT[event.side]}，正在被将军` : `轮到${SEAT[event.side]}`;
    return;
  }
  if (event.type === "summary") {
    renderSummary(document.getElementById("game-report"), event.game, "本局首步合法率");
    return;
  }
  if (event.type === "game_finished") {
    document.getElementById("status").textContent = `对局结束：${event.result || ""} ${event.result_reason || ""}`;
    document.getElementById("start").disabled = false;
    document.getElementById("stop").disabled = true;
    if (event.summary) renderSummary(document.getElementById("game-report"), event.summary, "本局首步合法率");
    loadReports();
  }
}

const replay = { frames: [], index: 0, timer: null, id: null };

function showAttempt(attempt) {
  if (!attempt) return;
  const seat = attempt.seat;
  state.live[seat].prompt = inputText(attempt);
  state.live[seat].reasoning = attempt.reasoning || "";
  state.live[seat].content = attempt.content || "";
  setLive(seat);
}

function showFrame(index) {
  if (!replay.frames.length) return;
  replay.index = Math.max(0, Math.min(index, replay.frames.length - 1));
  const frame = replay.frames[replay.index];
  state.pieces = frame.position.pieces;
  state.side = frame.position.side;
  state.last = frame.position.last_move;
  renderBoard();
  state.live.red = { prompt: "", reasoning: "", content: "" };
  state.live.black = { prompt: "", reasoning: "", content: "" };
  for (let i = 0; i <= replay.index; i += 1) {
    const attempt = replay.frames[i].attempt;
    if (!attempt) continue;
    state.live[attempt.seat] = {
      prompt: inputText(attempt),
      reasoning: attempt.reasoning || "",
      content: attempt.content || "",
    };
  }
  setLive("red");
  setLive("black");
  const label = document.getElementById("replay-label");
  label.textContent = `${replay.id}  ${replay.index + 1}/${replay.frames.length}  ${frame.label}`;
  document.getElementById("status").textContent = frame.position.in_check
    ? `回放：轮到${SEAT[frame.position.side]}，正在被将军`
    : `回放：轮到${SEAT[frame.position.side]}`;
}

function buildFrames(events) {
  const frames = [];
  let position = null;
  let accepted = null;
  for (const event of events) {
    if (event.type === "position" && !event.last_move) {
      position = event;
      frames.push({ position, attempt: null, label: "开局" });
    } else if (event.type === "attempt" && !event.ok && position) {
      frames.push({
        position,
        attempt: event,
        label: `${SEAT[event.seat]}未走成：${ERRORS[event.error_type] || event.error_type || ""}`,
      });
    } else if (event.type === "attempt" && event.ok) {
      accepted = event;
    } else if (event.type === "position" && event.last_move) {
      position = event;
      const move = event.last_move;
      frames.push({
        position,
        attempt: accepted,
        label: `${SEAT[move.seat]} ${move.uci} ${move.zh || ""}`,
      });
      accepted = null;
    }
  }
  return frames;
}

async function openReplay(gameId) {
  if (replay.timer) {
    clearInterval(replay.timer);
    replay.timer = null;
    document.getElementById("replay-play").textContent = "播放";
  }
  const response = await fetch(`/api/games/${gameId}/events`);
  const data = await response.json();
  replay.id = gameId;
  replay.frames = buildFrames(data.events || []);
  document.getElementById("moves").innerHTML = "";
  for (const event of data.events || []) {
    if (event.type === "attempt") addMove(event);
  }
  const summary = [...(data.events || [])].reverse().find((event) => event.type === "summary" || event.type === "game_finished");
  const report = summary && (summary.game || summary.summary);
  if (report) renderSummary(document.getElementById("game-report"), report, "本局首步合法率");
  showFrame(Math.max(0, replay.frames.length - 1));
  for (const button of document.querySelectorAll("#game-list button")) {
    button.classList.toggle("active", button.dataset.id === gameId);
  }
}

async function loadReports() {
  const response = await fetch("/api/reports");
  const data = await response.json();
  renderSummary(document.getElementById("all-report"), data.aggregate, `共 ${data.aggregate.games} 局`);
  const list = document.getElementById("game-list");
  list.innerHTML = "";
  const rows = [];
  for (const game of data.games || []) {
    let stateLabel = game.result || "已中断";
    let live = false;
    if (!game.result) {
      try {
        const info = await (await fetch(`/api/games/${game.game_id}`)).json();
        live = Boolean(info.live);
        stateLabel = live ? "进行中" : "已中断";
      } catch (error) {
        stateLabel = "已中断";
      }
    }
    rows.push({ game, stateLabel, live });
  }
  rows.sort((a, b) => Number(b.live) - Number(a.live) || (a.game.game_id < b.game.game_id ? 1 : -1));
  let liveId = null;
  for (const row of rows) {
    if (row.live && !liveId) liveId = row.game.game_id;
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.id = row.game.game_id;
    const rate = percent(row.game.first_try_legal_rate);
    button.textContent = `${row.game.game_id}  ${row.stateLabel}  首步合法率 ${rate}`;
    button.addEventListener("click", () => {
      if (row.live) watchLive(row.game.game_id);
      else openReplay(row.game.game_id);
    });
    list.appendChild(button);
  }
  if (liveId && !state.socket) watchLive(liveId);
  for (const button of document.querySelectorAll("#game-list button")) {
    button.classList.toggle("active", button.dataset.id === state.id);
  }
}

async function checkHealth() {
  const node = document.getElementById("health");
  try {
    const data = await (await fetch("/api/vllm/health")).json();
    node.textContent = data.ok ? "vLLM 已就绪" : "vLLM 未启动，可以先用脚本试下";
  } catch (error) {
    node.textContent = "检查 vLLM 失败";
  }
}

document.getElementById("controls").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const body = {
    backend: form.get("backend"),
    prompt_policy: form.get("prompt_policy"),
    red_temperature: Number(form.get("red_temperature")),
    black_temperature: Number(form.get("black_temperature")),
    max_plies: Number(form.get("max_plies")),
    max_tokens: 16384,
    record_video: form.get("record_video") === "on",
    temperature: Number(form.get("red_temperature")),
  };
  const response = await fetch("/api/games", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    document.getElementById("status").textContent = data.detail || "无法开局";
    return;
  }
  watchLive(data.id);
  loadReports();
});

document.getElementById("stop").addEventListener("click", async () => {
  if (!state.id) return;
  await fetch(`/api/games/${state.id}/stop`, { method: "POST" });
});

document.getElementById("replay-prev").addEventListener("click", () => showFrame(replay.index - 1));
document.getElementById("replay-next").addEventListener("click", () => showFrame(replay.index + 1));
document.getElementById("replay-play").addEventListener("click", () => {
  const button = document.getElementById("replay-play");
  if (replay.timer) {
    clearInterval(replay.timer);
    replay.timer = null;
    button.textContent = "播放";
    return;
  }
  button.textContent = "暂停";
  replay.timer = setInterval(() => {
    if (replay.index >= replay.frames.length - 1) {
      clearInterval(replay.timer);
      replay.timer = null;
      button.textContent = "播放";
      return;
    }
    showFrame(replay.index + 1);
  }, 700);
});

drawLines();
renderBoard();
loadReports();
checkHealth();
