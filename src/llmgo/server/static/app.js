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
  occupied: "落点已有子",
  off_board: "超出棋盘",
  illegal_move: "走法不合法",
};
const SEAT = { red: "红方", black: "黑方", white: "白方" };
const GAMES = {
  xiangqi: { name: "xiangqi", first: "red", second: "black", files: 9, ranks: 10, labels: { red: "红方", black: "黑方" } },
  chess: { name: "chess", first: "white", second: "black", files: 8, ranks: 8, labels: { white: "White", black: "Black" } },
  gomoku: { name: "gomoku", first: "black", second: "white", files: 15, ranks: 15, labels: { black: "Black", white: "White" } },
};
const CHESS_SYMBOL = { king: "♚", queen: "♛", rook: "♜", bishop: "♝", knight: "♞", pawn: "♟" };

function gameSpec(name) {
  return GAMES[name] || GAMES.xiangqi;
}

function seatName(seat, rulesName) {
  const spec = gameSpec(rulesName || state.game);
  return spec.labels[seat] || SEAT[seat] || seat;
}

function column(seat) {
  return seat === gameSpec(state.game).second ? "black" : "red";
}

function ensureLive(seat) {
  if (!state.live[seat]) state.live[seat] = { prompt: "", reasoning: "", tools: "", content: "" };
  return state.live[seat];
}

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

function initialChessPieces() {
  const back = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"];
  const pieces = [];
  back.forEach((kind, file) => {
    pieces.push({ file, rank: 0, color: "white", kind });
    pieces.push({ file, rank: 7, color: "black", kind });
    pieces.push({ file, rank: 1, color: "white", kind: "pawn" });
    pieces.push({ file, rank: 6, color: "black", kind: "pawn" });
  });
  return pieces;
}

const state = {
  id: null,
  socket: null,
  pieces: initialChessPieces(),
  last: null,
  side: "white",
  game: "chess",
  files: 8,
  ranks: 8,
  attempts: [],
  choiceMarks: [],
  modelSeat: null,
  modelName: "Qwen3.5-4B",
  externalTimer: null,
  live: {
    red: { prompt: "", reasoning: "", tools: "", content: "" },
    black: { prompt: "", reasoning: "", tools: "", content: "" },
  },
};

const board = document.getElementById("board");
const piecesEl = document.getElementById("pieces");
const marksEl = document.getElementById("marks");

function place(el, file, rank) {
  const spec = gameSpec(state.game);
  if (spec.name === "chess") {
    el.style.left = `${((file + 0.5) / spec.files) * 100}%`;
    el.style.top = `${((spec.ranks - rank - 0.5) / spec.ranks) * 100}%`;
    return;
  }
  const xDenom = Math.max(spec.files - 1, 1);
  const yDenom = Math.max(spec.ranks - 1, 1);
  el.style.left = `${(file / xDenom) * 100}%`;
  el.style.top = `${((spec.ranks - 1 - rank) / yDenom) * 100}%`;
}

function drawLines() {
  const spec = gameSpec(state.game);
  if (spec.name === "chess") {
    drawChessBoard();
    return;
  }
  if (spec.name === "gomoku") {
    drawGomokuLines();
    return;
  }
  drawXiangqiLines();
}

function drawXiangqiLines() {
  const svg = document.getElementById("lines");
  svg.setAttribute("viewBox", "0 0 8 9");
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

function drawChessBoard() {
  const svg = document.getElementById("lines");
  svg.setAttribute("viewBox", "0 0 8 8");
  const rects = [];
  for (let rank = 0; rank < 8; rank += 1) {
    for (let file = 0; file < 8; file += 1) {
      const dark = (file + rank) % 2 === 1;
      rects.push(`<rect x="${file}" y="${7 - rank}" width="1" height="1" fill="${dark ? "#b58863" : "#f0d9b5"}" />`);
    }
  }
  svg.innerHTML = rects.join("");
  const files = "abcdefgh";
  const mark = (html) => [...files].map((ch, i) => `<span style="left:${((i + 0.5) / 8) * 100}%">${html(ch, i)}</span>`).join("");
  document.getElementById("files-top").innerHTML = mark((ch) => ch);
  document.getElementById("files-bottom").innerHTML = mark((ch) => ch);
  document.getElementById("ranks").innerHTML = Array.from({ length: 8 }, (_, i) => `<span style="top:${((i + 0.5) / 8) * 100}%">${8 - i}</span>`).join("");
}

function drawGomokuLines() {
  const svg = document.getElementById("lines");
  svg.setAttribute("viewBox", "0 0 14 14");
  const seg = [];
  const add = (x1, y1, x2, y2) => seg.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" />`);
  for (let i = 0; i < 15; i += 1) {
    add(0, i, 14, i);
    add(i, 0, i, 14);
  }
  svg.innerHTML = seg.join("");
  const files = "abcdefghijklmno";
  const mark = (html) => [...files].map((ch, i) => `<span style="left:${(i / 14) * 100}%">${html(ch, i)}</span>`).join("");
  document.getElementById("files-top").innerHTML = mark((ch) => ch);
  document.getElementById("files-bottom").innerHTML = mark((ch) => ch);
  document.getElementById("ranks").innerHTML = Array.from({ length: 15 }, (_, i) => `<span style="top:${(i / 14) * 100}%">${15 - i}</span>`).join("");
}

function setGame(name, files, ranks) {
  const spec = gameSpec(name);
  state.game = spec.name;
  state.files = files || spec.files;
  state.ranks = ranks || spec.ranks;
  board.className = `board ${spec.name}`;
  document.querySelector(".river").style.display = spec.name === "xiangqi" ? "" : "none";
  document.querySelector("#red-seat h2").textContent = spec.labels[spec.first];
  document.querySelector("#black-seat h2").textContent = spec.labels[spec.second];
  document.querySelector("#red-seat header p").textContent = "只显示这一方自己的输出";
  document.querySelector("#black-seat header p").textContent = "只显示这一方自己的输出";
  drawLines();
  markSeats();
}

function renderBoard() {
  piecesEl.innerHTML = "";
  marksEl.innerHTML = "";
  const marks = [];
  if (state.last) marks.push(...movePoints(state.last.uci));
  marks.push(...(state.choiceMarks || []));
  for (const point of marks) {
    const mark = document.createElement("div");
    mark.className = `mark ${point.cls}`;
    place(mark, point.file, point.rank);
    marksEl.appendChild(mark);
  }
  for (const piece of state.pieces) {
    const el = document.createElement("div");
    el.className = `piece ${piece.color}${piece.kind === "stone" ? " stone" : ""}${state.game === "chess" ? " chess-piece" : ""}`;
    const text = pieceText(piece);
    if (text) el.innerHTML = `<span>${text}</span>`;
    if (state.game === "gomoku") el.style.width = "5.2%";
    place(el, piece.file, piece.rank);
    piecesEl.appendChild(el);
  }
}

function pieceText(piece) {
  if (piece.kind === "stone") return "";
  if (state.game === "chess") return CHESS_SYMBOL[piece.kind] || "";
  return CHAR[piece.color][piece.kind];
}

function squareMark(sq, cls) {
  if (!sq || sq.length < 2) return null;
  const file = sq.charCodeAt(0) - 97;
  const rank = Number(sq[1]) - 1;
  if (file < 0 || file > 7 || rank < 0 || rank > 7) return null;
  return { cls, file, rank };
}

function markSeats() {
  const spec = gameSpec(state.game);
  const sides = [
    ["red", spec.first],
    ["black", spec.second],
  ];
  for (const [node, seat] of sides) {
    const note = document.querySelector(`#${node}-seat header p`);
    if (!note) continue;
    if (!state.modelSeat) continue;
    note.textContent = seat === state.modelSeat
      ? `模型 ${state.modelName || "Qwen3.5-4B"}`
      : "Fairy-Stockfish";
  }
}

function choiceTrace(choices, previousStart) {
  const lines = [];
  let previous = previousStart ? previousStart.slice() : [];
  for (const event of choices || []) {
    if (event.type === "tool_call" && event.status === "finished") {
      lines.push(`${event.tool} ${JSON.stringify(event.parameters || {})}\n${(event.result || []).join("\n")}`);
      continue;
    }
    if (event.type === "route_recovery") {
      lines.push(`路由恢复 ${event.stage}: ${event.reason} → ${event.action}`);
      continue;
    }
    if (event.source === "random") {
      lines.push(`未取得有效最终答案，随机走 ${event.uci || ""}`);
      continue;
    }
    if (event.status !== "answered") continue;
    const stage = { tool: "选工具", piece: "选子", move: "选走法" }[event.stage] || "选择";
    const option = (event.options || []).find((item) => event.uci ? item.uci === event.uci : event.letter && item.letter === event.letter);
    const name = option ? option.label : (event.tool || "");
    lines.push(`${stage} ${event.uci || event.letter || ""} ${name}${event.source === "forced" ? "（唯一合法步）" : ""}`.trim());
    const added = (event.blackboard || []).filter((line) => !previous.includes(line));
    previous = event.blackboard || previous;
    if (event.stage === "tool" && added.length) lines.push(added.join("\n"));
  }
  return lines.join("\n\n");
}

function displayText(text) {
  return text.length <= 20000 ? text : text.slice(0, 4000) + "\n\n[… 网页省略中段，完整原文保存在本机 events.jsonl …]\n\n" + text.slice(-12000);
}

function excerptText(excerpt) {
  if (!excerpt) return "";
  const gap = excerpt.omitted ? `\n\n[… 省略 ${excerpt.omitted} ${excerpt.unit} …]\n\n` : "";
  const warning = excerpt.exact ? "" : `\n[字元截取，非精确 token：${excerpt.warning || "缺少 tokenizer"}]`;
  return excerpt.head + gap + excerpt.tail + warning;
}

function updateRequest(live, event) {
  if (!event.request_id) return false;
  if (!live.requests) live.requests = new Map();
  let request = live.requests.get(event.request_id);
  if (!request) {
    request = { reasoning: "", content: "", finished: false };
    live.requests.set(event.request_id, request);
  }
  if (event.type === "token") {
    if (request.finished || !["reasoning", "content"].includes(event.channel)) return false;
    request[event.channel] = displayText(event.replace ? event.text : request[event.channel] + event.text);
  } else if (["finished", "streaming"].includes(event.status)) {
    // Replace streamed/provisional text with the authoritative complete response.
    request.reasoning = displayText(excerptText(event.output_excerpt) || event.reasoning || (!event.text ? event.output || "" : ""));
    request.content = displayText(event.text || event.failure_reason || "");
    request.finished = event.status === "finished";
  }
  const requests = [...live.requests.values()];
  live.reasoning = requests.map(item => item.reasoning).filter(Boolean).join("\n\n");
  live.content = requests[requests.length - 1].content;
  return true;
}

function requestView(choices) {
  const live = { reasoning: "", content: "" };
  for (const event of choices || []) {
    if (event.type === "model_request" || event.type === "token") updateRequest(live, event);
  }
  if (!live.requests || !live.requests.size) {
    const tokens = (choices || []).filter(event => event.type === "token");
    for (const event of tokens) {
      if (["reasoning", "content"].includes(event.channel)) live[event.channel] = event.replace ? event.text : live[event.channel] + event.text;
    }
    if (!tokens.length) live.reasoning = (choices || []).map(event => excerptText(event.output_excerpt) || event.output || "").filter(Boolean).join("\n\n");
  }
  return live;
}

function modelOutput(choices) {
  const view = requestView(choices);
  return view.reasoning || (view.requests?.size ? "" : view.content);
}

function toolTrace(choices) {
  return choiceTrace((choices || []).filter((event) => event.stage === "tool" || event.type === "tool_call" || event.type === "route_recovery"));
}

function appendTrace(seat, event) {
  const root = document.getElementById(`${column(seat)}-trace`);
  if (!root) return;
  const request = event.type === "model_request";
  let box = request ? [...root.querySelectorAll("details[data-request-id]")].find((node) => node.dataset.requestId === event.request_id) : null;
  if (!box) {
    box = document.createElement("details");
    if (request) box.dataset.requestId = event.request_id;
    box.appendChild(document.createElement("summary"));
    box.appendChild(document.createElement("pre"));
    root.appendChild(box);
    // Formatting and raw HTTP loading happen only after the user opens this row.
    box.addEventListener("toggle", () => {
      const pre = box.querySelector("pre");
      if (!box.open) { pre.textContent = ""; return; }
      pre.textContent = traceText(box.traceEvent);
    });
  }
  box.traceEvent = event;
  const tokens = event.usage && event.usage.completion_tokens !== undefined ? event.usage.completion_tokens : event.completion_tokens;
  const time = event.latency_ms !== undefined ? ` ${(event.latency_ms / 1000).toFixed(2)}s` : "";
  const name = request ? `模型请求 ${event.request_id} · ${event.stage}` : event.type === "tool_call" ? `工具 ${event.tool}` : event.type === "choice" ? `选项 ${event.stage}` : event.type;
  box.querySelector("summary").textContent = `${name} · ${event.status || event.reason || ""}${time}${tokens !== undefined ? ` · ${tokens} tokens${event.tokens_exact === false ? "（估算）" : ""}` : ""}${event.finish_reason ? ` · ${event.finish_reason}` : ""}${event.failure_reason ? ` · ${event.failure_reason}` : ""}`;
  if (event.raw_ref && !box.rawButton) {
    const button = document.createElement("button");
    button.textContent = "按需载入本机完整事件";
    const gameId = state.id || replay.id;
    button.addEventListener("click", async () => {
      button.disabled = true;
      const pre = box.querySelector("pre");
      try {
        const response = await fetch(`/api/games/${gameId}/event?offset=${box.traceEvent.raw_ref.offset}`);
        if (!response.ok) throw new Error("此机器没有完整原文；GitHub 仅保存精简记录");
        pre.textContent = traceText(await response.json());
      } catch (error) { pre.textContent = String(error); }
      finally { button.disabled = false; }
    });
    box.appendChild(button);
    box.rawButton = button;
  }
  if (box.open) box.querySelector("pre").textContent = traceText(event);
}

function traceText(event) {
  const meta = { ...event };
  for (const key of ["prompt", "text", "reasoning", "output", "raw_response", "request_payload", "output_excerpt"]) delete meta[key];
  const parts = [JSON.stringify(meta, null, 2)];
  if (event.prompt) parts.push(`实际提示\n${event.prompt}`);
  if (event.output_excerpt) parts.push(`输出头尾（${event.output_excerpt.unit}）\n${excerptText(event.output_excerpt)}`);
  if (event.reasoning) parts.push(`思考全文\n${event.reasoning}`);
  else if (event.output) parts.push(`模型原始输出\n${event.output}`);
  if (event.text) parts.push(`正式回答\n${event.text}`);
  if (event.request_payload) parts.push(`请求参数\n${JSON.stringify(event.request_payload, null, 2)}`);
  if (event.raw_response) parts.push(`原始 HTTP 响应\n${event.raw_response}`);
  return parts.join("\n\n");
}

async function loadHarnessLibrary() {
  try {
    const response = await fetch("/api/harness");
    if (!response.ok) throw new Error("工具库读取失败");
    const data = await response.json();
    const tools = document.getElementById("tool-library");
    const skills = document.getElementById("skill-library");
    tools.replaceChildren(); skills.replaceChildren();
    for (const tool of data.tools || []) {
      const item = document.createElement("article");
      item.textContent = `${tool.mode === "automatic" ? "自动" : tool.letter || ""} ${tool.name} — ${tool.description}`;
      tools.appendChild(item);
    }
    for (const card of data.skills || []) {
      const item = document.createElement("details");
      const title = document.createElement("summary"); title.textContent = `${card.id} · ${card.title}`;
      const pre = document.createElement("pre"); pre.textContent = `触发条件：${card.when}\n动作：${card.rule}\n反例：${card.counterexample}`;
      item.append(title, pre);
      for (const url of card.sources || []) {
        const link = document.createElement("a"); link.href = url; link.textContent = "棋谱／来源 "; link.target = "_blank"; link.rel = "noopener"; item.appendChild(link);
      }
      skills.appendChild(item);
    }
    document.getElementById("harness-budget").textContent = JSON.stringify({ enable_thinking: data.enable_thinking, max_plies: data.max_plies, ...data.config }, null, 2);
  } catch (error) {
    document.getElementById("tool-library").textContent = String(error);
  }
}

function choiceMarks(event) {
  const marks = [];
  for (const sq of event.origins || []) {
    const point = squareMark(sq, "pick");
    if (point) marks.push(point);
  }
  for (const sq of event.targets || []) {
    const point = squareMark(sq, "dot");
    if (point) marks.push(point);
  }
  return marks;
}

function movePoints(uci) {
  if (!uci) return [];
  if (state.game === "gomoku") {
    const found = /^([a-o])(\d{1,2})$/i.exec(uci);
    if (!found) return [];
    return [{ cls: "to", file: found[1].toLowerCase().charCodeAt(0) - 97, rank: Number(found[2]) - 1 }];
  }
  if (state.game === "chess") {
    const found = /^([a-h])([1-8])([a-h])([1-8])/i.exec(uci);
    if (!found) return [];
    const file = (ch) => ch.toLowerCase().charCodeAt(0) - 97;
    return [
      { cls: "from", file: file(found[1]), rank: Number(found[2]) - 1 },
      { cls: "to", file: file(found[3]), rank: Number(found[4]) - 1 },
    ];
  }
  const found = /^([a-i])(\d)([a-i])(\d)/i.exec(uci);
  if (!found) return [];
  return [
    { cls: "from", file: FILES.indexOf(found[1]), rank: Number(found[2]) },
    { cls: "to", file: FILES.indexOf(found[3]), rank: Number(found[4]) },
  ];
}

function inputText(event) {
  if (!event) return "";
  return event.llm_input || event.prompt || "";
}

function archive(seat) {
  const live = ensureLive(seat);
  const trace = document.getElementById(`${column(seat)}-trace`);
  if (!live.prompt && !live.reasoning && !live.content && !live.tools && (!trace || !trace.firstChild)) return;
  const box = document.createElement("details");
  const title = document.createElement("summary");
  title.textContent = `${seatName(seat)}先前一手的提示和输出`;
  const pre = document.createElement("pre");
  const saved = {prompt: live.prompt, reasoning: live.reasoning, tools: live.tools, content: live.content};
  box.addEventListener("toggle", () => {
    pre.textContent = box.open ? `送给模型的提示\n${saved.prompt}\n\n思考\n${saved.reasoning}\n\n工具\n${saved.tools || ""}\n\n正式输出\n${saved.content}` : "";
  });
  box.append(title, pre);
  if (trace) while (trace.firstChild) box.appendChild(trace.firstChild);
  const archiveRoot = document.getElementById(`${column(seat)}-archive`);
  archiveRoot.prepend(box);
  while (archiveRoot.children.length > 6) archiveRoot.children[archiveRoot.children.length - 1].remove();
  live.prompt = "";
  live.reasoning = "";
  live.tools = "";
  live.content = "";
  live.blackboardLines = [];
  live.requests = new Map();
  setLive(seat);
}

function setLive(seat) {
  const live = state.live[seat];
  const node = column(seat);
  document.getElementById(`${node}-prompt`).textContent = live ? live.prompt : "";
  document.getElementById(`${node}-reasoning`).textContent = live ? live.reasoning : "";
  const toolsNode = document.getElementById(`${node}-tools`);
  if (toolsNode) toolsNode.textContent = live ? live.tools || "" : "";
  document.getElementById(`${node}-content`).textContent = live ? live.content : "";
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
      const node = document.getElementById(`${column(paintedSeat)}-${paintedChannel}`);
      const follow = node.scrollHeight - node.scrollTop - node.clientHeight < 80;
      node.textContent = displayText(state.live[paintedSeat]?.[paintedChannel] || "");
      if (follow) node.scrollTop = node.scrollHeight;
    }
    dirtyChannels.clear();
  });
}

async function watchLive(gameId) {
  if (state.socket && state.id === gameId && state.socket.readyState <= 1) return;
  state.id = gameId;
  document.getElementById("start").disabled = true;
  document.getElementById("stop").disabled = false;
  document.getElementById("status").textContent = "正在接上进行中的对局…";
  if (state.socket) state.socket.close();
  if (state.externalTimer) clearInterval(state.externalTimer);
  state.externalTimer = null;
  const info = await (await fetch(`/api/games/${gameId}`)).json();
  state.modelSeat = info.model_seat || null;
  state.modelName = info.model || "Qwen3.5-4B";
  markSeats();
  if (!info.live) {
    await openReplay(gameId);
    return;
  }
  if (info.external) {
    // CLI games tail events.jsonl through the same WebSocket as web-started games.
    document.getElementById("stop").disabled = true;
  }
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/ws/games/${gameId}`);
  state.socket = socket;
  socket.onmessage = (message) => onEvent(JSON.parse(message.data));
  for (const button of document.querySelectorAll(".game-list button")) {
    button.classList.toggle("active", button.dataset.id === gameId);
  }
}

function addMove(attempt) {
  const row = document.createElement("div");
  row.className = attempt.ok ? "move" : "move bad";
  const reason = attempt.error_detail || ERRORS[attempt.error_type] || attempt.error_type || "";
  const random = attempt.source === "random" ? " 随机" : "";
  const label = attempt.ok
    ? `${seatName(attempt.seat)} ${attempt.uci} ${attempt.zh || ""}${random}`
    : `${seatName(attempt.seat)} 第 ${attempt.attempt_index + 1} 次 ${attempt.uci || "没有走出"}：${reason}`;
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

function renderSummary(node, summary, title, rulesName) {
  node.innerHTML = "";
  if (!summary) {
    node.textContent = "还没有数据。";
    return;
  }
  const head = document.createElement("div");
  head.innerHTML = `<div>${title}</div><div class="rate">${percent(summary.first_try_legal_rate)}</div><div>首步合法 ${summary.first_try_legal || 0} / ${summary.first_try_total || 0}</div>`;
  if (summary.result) head.insertAdjacentHTML("beforeend", `<div>结果：${summary.result}（${summary.result_reason || ""}）</div>`);
  node.appendChild(head);
  for (const seat of Object.keys(summary.by_seat || {})) {
    const item = summary.by_seat[seat];
    if (!item) continue;
    const line = document.createElement("p");
    const rate = item.first_try_total ? item.first_try_legal / item.first_try_total : null;
    line.textContent = `${seatName(seat, rulesName)}${summary.harness ? "棋盘合法率" : "首步合法率"} ${percent(rate)}（${item.first_try_legal}/${item.first_try_total}）`;
    node.appendChild(line);
  }
  bars(node, summary.error_types);
  if (summary.harness) {
    const diagnostic = document.createElement("p");
    const choices = summary.model_choices ?? Math.max(0, (summary.model_turns || 0) - (summary.forced_moves || 0));
    diagnostic.textContent = `模型选择完成 ${summary.model_decisions || 0}/${choices}（${percent(choices ? summary.model_decisions / choices : null)}）；强制步 ${summary.forced_moves || 0}；随机后备 ${summary.random_moves || 0}（${percent(summary.random_rate)}）；请求 ${summary.request_count || 0}；生成 ${summary.completion_tokens || 0} tokens${summary.tokens_exact === false ? "（含估算）" : ""}；后备原因 ${JSON.stringify(summary.fallback_reasons || {})}。棋盘合法率包含程式后备。`;
    node.appendChild(diagnostic);
  }
  if (summary.finish_length_count) {
    const note = document.createElement("p");
    note.textContent = `有 ${summary.finish_length_count} 次碰到模型长度上限。网页使用精简输出；完整原文保存在本机 events.jsonl。`;
    node.appendChild(note);
  }
}

function clearColumns() {
  for (const node of ["red", "black"]) {
    document.getElementById(`${node}-prompt`).textContent = "";
    document.getElementById(`${node}-reasoning`).textContent = "";
    const toolsNode = document.getElementById(`${node}-tools`);
    if (toolsNode) toolsNode.textContent = "";
    document.getElementById(`${node}-content`).textContent = "";
    document.getElementById(`${node}-archive`).innerHTML = "";
    document.getElementById(`${node}-trace`).replaceChildren();
  }
}

function resetView() {
  state.attempts = [];
  state.last = null;
  state.live = {};
  document.getElementById("moves").innerHTML = "";
  state.choiceMarks = [];
  const boardNote = document.getElementById("blackboard");
  if (boardNote) boardNote.textContent = "";
  clearColumns();
  setGame(state.game || "xiangqi");
  state.pieces = state.game === "xiangqi" ? initialPieces() : state.game === "chess" ? initialChessPieces() : [];
  state.side = gameSpec(state.game).first;
  renderBoard();
}

function onEvent(event) {
  if (event.type === "replay_begin") {
    resetView();
    return;
  }
  if (event.type === "turn_started") {
    archive(event.seat);
    ensureLive(event.seat).prompt = inputText(event);
    setLive(event.seat);
    if (event.tool_library) appendTrace(event.seat, event);
    document.getElementById("status").textContent = `${seatName(event.seat)}正在思考，第 ${event.attempt_index + 1} 次`;
    return;
  }
  if (["model_request", "tool_call", "fallback", "route_recovery"].includes(event.type)) {
    appendTrace(event.seat, event);
    const live = ensureLive(event.seat);
    if (event.type === "model_request") {
      live.prompt = event.prompt || live.prompt;
      updateRequest(live, event);
    }
    if (event.type === "tool_call" && event.status === "finished") {
      live.tools = [live.tools, `${event.tool}:\n${(event.result || []).join("\n")}`].filter(Boolean).join("\n\n");
      document.getElementById("blackboard").textContent = (event.blackboard || []).join("\n");
    }
    if (event.type === "fallback") live.content = `random ${event.uci} · ${event.reason}`;
    setLive(event.seat);
    return;
  }
  if (event.type === "token") {
    if (!["reasoning", "content"].includes(event.channel)) return;
    const live = ensureLive(event.seat);
    if (event.request_id) updateRequest(live, event);
    else live[event.channel] = displayText(event.replace ? event.text : live[event.channel] + event.text);
    schedulePaint(event.seat, "reasoning");
    schedulePaint(event.seat, "content");
    return;
  }
  if (event.type === "attempt") {
    state.attempts.push(event);
    const live = ensureLive(event.seat);
    live.prompt = inputText(event) || live.prompt;
    if (!live.reasoning) {
      live.reasoning = event.source === "engine" ? "这一步是 Fairy-Stockfish，不是模型。" : (event.reasoning || "");
    }
    live.content = event.content || live.content;
    setLive(event.seat);
    addMove(event);
    return;
  }
  if (event.type === "choice") {
    const seat = event.seat;
    const live = ensureLive(seat);
    const freshPly = event.stage === "tool" && event.status === "asking" && !(event.blackboard || []).length;
    if (freshPly && !event.turn_id) archive(seat);
    if (event.skills && event.skills.length && event.status === "asking") appendTrace(seat, event);
    live.prompt = event.prompt || live.prompt;
    if (event.output || event.output_excerpt) {
      const output = excerptText(event.output_excerpt) || event.output;
      live.reasoning = displayText(live.reasoning ? `${live.reasoning}\n\n${output}` : output);
    }
    if (event.stage === "tool" && (event.status === "answered" || event.source === "random")) {
      const piece = choiceTrace([event], live.blackboardLines || []);
      live.blackboardLines = event.blackboard || live.blackboardLines || [];
      if (piece) live.tools = live.tools ? `${live.tools}\n\n${piece}` : piece;
    } else if (event.blackboard) {
      live.blackboardLines = event.blackboard;
    }
    if (event.status === "answered" || event.source === "random") {
      live.content = event.source === "random" ? `random ${event.uci || ""}` : (event.uci || event.letter || live.content);
    }
    setLive(seat);
    const note = document.getElementById("blackboard");
    note.textContent = (event.blackboard || []).join("\n");
    state.choiceMarks = choiceMarks(event);
    renderBoard();
    const stage = { tool: "选工具", piece: "选子", move: "选走法" }[event.stage] || "选择";
    const picked = event.uci ? `，选择 ${event.uci}${event.source === "forced" ? "（唯一合法步）" : ""}` : event.letter ? `，回了 ${event.letter}` : "";
    const random = event.source === "random" ? `，随机 ${event.uci}` : "";
    document.getElementById("status").textContent = `${seatName(seat)}正在${stage}${picked}${random}`;
    return;
  }
  if (event.type === "position") {
    setGame(event.game || "xiangqi", event.files, event.ranks);
    state.pieces = event.pieces;
    state.side = event.side;
    state.last = event.last_move;
    state.choiceMarks = [];
    renderBoard();
    document.getElementById("status").textContent = event.in_check ? `轮到${seatName(event.side)}，正在被将军` : `轮到${seatName(event.side)}`;
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
let replayLoad = 0;

function showAttempt(attempt) {
  if (!attempt) return;
  const seat = attempt.seat;
  ensureLive(seat).prompt = inputText(attempt);
  ensureLive(seat).reasoning = attempt.reasoning || "";
  ensureLive(seat).content = attempt.content || "";
  setLive(seat);
}

function showFrame(index) {
  if (!replay.frames.length) return;
  replay.index = Math.max(0, Math.min(index, replay.frames.length - 1));
  const frame = replay.frames[replay.index];
  setGame(frame.position.game || "xiangqi", frame.position.files, frame.position.ranks);
  state.pieces = frame.position.pieces;
  state.side = frame.position.side;
  state.last = frame.position.last_move;
  state.choiceMarks = [];
  const notebook = [...(frame.choices || [])].reverse().find((event) => event.blackboard);
  document.getElementById("blackboard").textContent = notebook ? notebook.blackboard.join("\n") : "";
  renderBoard();
  state.live = {};
  clearColumns();
  const lastBySeat = new Map();
  for (let i = 0; i <= replay.index; i += 1) {
    const past = replay.frames[i];
    if (past.attempt || past.pending) lastBySeat.set(past.attempt ? past.attempt.seat : past.seat, past);
  }
  for (const past of lastBySeat.values()) {
    const pendingView = requestView(past.choices);
    const pendingPrompt = [...(past.choices || [])].reverse().find(event => event.prompt);
    const attempt = past.attempt || {seat: past.seat, prompt: pendingPrompt?.prompt || "", content: pendingView.content};
    if (!attempt) continue;
    const engine = attempt.source === "engine" ? "这一步是 Fairy-Stockfish，不是模型。" : "";
    state.live[attempt.seat] = {
      prompt: inputText(attempt) || pendingPrompt?.prompt || "",
      reasoning: modelOutput(past.choices) || attempt.reasoning || engine,
      tools: toolTrace(past.choices),
      content: attempt.content || "",
    };
    const root = document.getElementById(`${column(attempt.seat)}-trace`);
    root.replaceChildren();
    for (const event of past.choices || []) {
      if (["model_request", "tool_call", "fallback", "route_recovery"].includes(event.type) || event.skills) appendTrace(attempt.seat, event);
    }
  }
  setLive(gameSpec(state.game).first);
  setLive(gameSpec(state.game).second);
  const label = document.getElementById("replay-label");
  const who = state.modelSeat ? `模型执${seatName(state.modelSeat)}。` : "";
  label.textContent = `${replay.id}  ${replay.index + 1}/${replay.frames.length}  ${frame.label}`;
  document.getElementById("status").textContent = frame.position.in_check
    ? `${who}回放：轮到${seatName(frame.position.side)}，正在被将军`
    : `${who}回放：轮到${seatName(frame.position.side)}`;
}

function buildFrames(events) {
  const frames = [];
  let position = null;
  let accepted = null;
  let choices = [];
  for (const event of events) {
    if (["choice", "model_request", "tool_call", "fallback", "route_recovery", "token"].includes(event.type)) {
      choices.push(event);
    } else if (event.type === "position" && !event.last_move) {
      position = event;
      frames.push({ position, attempt: null, choices: [], label: "开局" });
    } else if (event.type === "attempt" && !event.ok && position) {
      frames.push({
        position,
        attempt: event,
        choices,
        label: `${seatName(event.seat)} ${event.uci || "没有走出"} 未走成：${event.error_detail || ERRORS[event.error_type] || ""}`,
      });
      choices = [];
    } else if (event.type === "attempt" && event.ok) {
      accepted = event;
    } else if (event.type === "position" && event.last_move) {
      position = event;
      const move = event.last_move;
      frames.push({
        position,
        attempt: accepted,
        choices,
        label: `${seatName(move.seat)} ${move.uci} ${move.zh || ""}`,
      });
      accepted = null;
      choices = [];
    }
  }
  if (choices.length && position) {
    const latest = [...choices].reverse().find(event => event.seat);
    frames.push({position, attempt: null, choices, pending: true, seat: latest?.seat,
      label: "尚未落子的模型回应（含未完成的生成）"});
  }
  return frames;
}

async function openReplay(gameId, options) {
  const loadId = ++replayLoad;
  if (state.socket) { state.socket.close(); state.socket = null; }
  if (state.externalTimer) clearInterval(state.externalTimer);
  state.externalTimer = null;
  state.id = null;
  if (replay.timer) {
    clearInterval(replay.timer);
    replay.timer = null;
    document.getElementById("replay-play").textContent = "播放";
  }
  // Byte cursors avoid rescanning raw token history. Yield between bounded pages.
  const data = {events: []};
  let cursor = 0;
  do {
    const response = await fetch(`/api/games/${gameId}/events?view=compact&limit=512&after=${cursor}`);
    if (!response.ok) throw new Error(`无法载入对局：${response.status}`);
    const page = await response.json();
    if (loadId !== replayLoad) return;
    data.events.push(...page.events);
    Object.assign(data, {model_seat: page.model_seat, model: page.model});
    if (!page.more || page.cursor === cursor) break;
    cursor = page.cursor;
  } while (true);
  const opening = (data.events || []).find((event) => event.type === "position");
  if (opening) setGame(opening.game || "xiangqi", opening.files, opening.ranks);
  replay.id = gameId;
  state.modelSeat = data.model_seat || null;
  state.modelName = data.model || "Qwen3.5-4B";
  markSeats();
  replay.frames = buildFrames(data.events || []);
  document.getElementById("moves").innerHTML = "";
  for (const event of data.events || []) {
    if (event.type === "attempt") addMove(event);
  }
  const summary = [...(data.events || [])].reverse().find((event) => event.type === "summary" || event.type === "game_finished");
  const report = summary && (summary.game || summary.summary);
  if (report) renderSummary(document.getElementById("game-report"), report, "本局首步合法率");
  showFrame(options && options.fromStart ? 0 : Math.max(0, replay.frames.length - 1));
  for (const button of document.querySelectorAll(".game-list button")) {
    button.classList.toggle("active", button.dataset.id === gameId);
  }
}

const RULE_TITLE = { xiangqi: "中国象棋", chess: "国际象棋", gomoku: "五子棋" };

function emptyAggregate() {
  return { games: 0, first_try_total: 0, first_try_legal: 0, first_try_legal_rate: null, error_types: {}, by_seat: {} };
}

function pageName() {
  return gameSpec(document.getElementById("game").value).name;
}

function showPage(name) {
  if (state.externalTimer) clearInterval(state.externalTimer);
  state.externalTimer = null;
  stopPlayback();
  if (state.socket) {
    const socket = state.socket;
    state.socket = null;
    state.id = null;
    socket.close();
  }
  document.getElementById("start").disabled = false;
  document.getElementById("stop").disabled = true;
  document.getElementById("status").textContent = "还没有开局";
  document.getElementById("game-report").textContent = "开始对局后，这里显示首步合法率和错误类型。";
  const spec = gameSpec(name);
  document.getElementById("game").value = spec.name;
  document.querySelector("h1").textContent = RULE_TITLE[spec.name];
  document.getElementById("list-title").textContent = `${RULE_TITLE[spec.name]}对局`;
  state.game = spec.name;
  resetView();
  loadReports();
}

async function loadReports() {
  const response = await fetch("/api/reports");
  const data = await response.json();
  const grouped = { ...(data.by_game || {}) };
  if (!data.by_game) {
    for (const game of data.games || []) {
      const name = game.game || "xiangqi";
      grouped[name] = grouped[name] || { games: [], aggregate: null };
      grouped[name].games.push(game);
    }
  }
  const name = pageName();
  const bucket = grouped[name] || { games: [], aggregate: emptyAggregate() };
  const root = document.getElementById("game-lists");
  root.innerHTML = "";
  const report = document.createElement("div");
  report.className = "report";
  renderSummary(report, bucket.aggregate || emptyAggregate(), `共 ${(bucket.aggregate || {}).games || 0} 局`, name);
  const list = document.createElement("div");
  list.className = "game-list";
  const rows = await Promise.all((bucket.games || []).map(async (game) => {
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
    return { game, stateLabel, live };
  }));
  rows.sort((a, b) => Number(b.live) - Number(a.live) || (a.game.game_id < b.game.game_id ? 1 : -1));
  if (!rows.length) {
    const empty = document.createElement("p");
    empty.textContent = "还没有对局。";
    list.appendChild(empty);
  }
  let liveId = null;
  for (const row of rows) {
    if (row.live && !liveId) liveId = row.game.game_id;
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.id = row.game.game_id;
    const choices = Math.max(0, (row.game.model_turns || 0) - (row.game.forced_moves || 0));
    const rate = row.game.harness && typeof row.game.model_decisions === "number"
      ? `模型作答 ${percent(choices ? row.game.model_decisions / choices : null)}`
      : `棋盘合法率 ${percent(row.game.first_try_legal_rate)}`;
    const side = row.game.model_seat === "white" ? "模型执白" : row.game.model_seat === "black" ? "模型执黑" : "";
    button.textContent = `${row.game.game_id}  ${side}  ${row.stateLabel}  ${rate}`.replace(/ +/g, " ");
    button.addEventListener("click", () => {
      if (row.live) watchLive(row.game.game_id);
      else openReplay(row.game.game_id);
    });
    list.appendChild(button);
  }
  root.append(report, list);
  if (liveId && !state.socket && !state.externalTimer && state.id !== liveId) watchLive(liveId);
  for (const button of document.querySelectorAll(".game-list button")) {
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
    max_tokens: form.get("backend") === "harness" || form.get("game") === "chess" ? 32768 : 16384,
    record_video: form.get("record_video") === "on",
    temperature: Number(form.get("red_temperature")),
    game: form.get("backend") === "harness" ? "chess" : (form.get("game") || "xiangqi"),
    opponent: form.get("opponent"),
    model_seat: form.get("model_seat"),
  };
  setGame(body.game);
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

function stopPlayback() {
  if (!replay.timer) return;
  clearInterval(replay.timer);
  replay.timer = null;
  document.getElementById("replay-play").textContent = "播放";
}

function startPlayback() {
  if (!replay.frames.length || replay.timer) return;
  document.getElementById("replay-play").textContent = "暂停";
  replay.timer = setInterval(() => {
    if (replay.index >= replay.frames.length - 1) {
      stopPlayback();
      return;
    }
    showFrame(replay.index + 1);
  }, 700);
}

document.getElementById("replay-prev").addEventListener("click", () => showFrame(replay.index - 1));
document.getElementById("replay-next").addEventListener("click", () => showFrame(replay.index + 1));
document.getElementById("replay-play").addEventListener("click", () => {
  if (replay.timer) {
    stopPlayback();
    return;
  }
  startPlayback();
});
document.getElementById("replay-restart").addEventListener("click", async () => {
  stopPlayback();
  const gameId = replay.id || state.id;
  if (!gameId) return;
  await openReplay(gameId, { fromStart: true });
  startPlayback();
});

document.getElementById("game").addEventListener("change", (event) => {
  showPage(event.target.value);
});
showPage("chess");
checkHealth();
loadHarnessLibrary();
