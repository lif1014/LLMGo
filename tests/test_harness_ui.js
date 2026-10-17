/* In-memory DOM smoke tests: no browser server or network required. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
class Element {
  constructor(tag = 'div') { this.tagName = tag; this.children = []; this.style = {}; this.dataset = {}; this.value = 'chess'; this.classList = { toggle() {} }; this._text = ''; }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  set innerHTML(value) { this._text = String(value); this.children = []; }
  get innerHTML() { return this._text; }
  get firstChild() { return this.children[0] || null; }
  appendChild(node) {
    if (node.parentNode) node.parentNode.children.splice(node.parentNode.children.indexOf(node), 1);
    this.children.push(node); node.parentNode = this; return node;
  }
  append(...nodes) { nodes.forEach(n => this.appendChild(n)); }
  prepend(node) { this.appendChild(node); this.children.unshift(this.children.pop()); }
  replaceChildren(...nodes) { this.children.forEach(n => n.parentNode = null); this.children = []; this._text = ''; this.append(...nodes); }
  querySelector(selector) { return this.children.find(n => n.tagName === selector) || null; }
  querySelectorAll(selector) {
    const all = this.children.flatMap(n => [n, ...n.querySelectorAll('*')]);
    return selector === 'details[data-request-id]' ? all.filter(n => n.tagName === 'details' && n.dataset.requestId) : all;
  }
  setAttribute() {}
  addEventListener() {}
  insertAdjacentHTML(_, html) { this._text += html; }
}
const nodes = new Map();
const node = id => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); };
const document = { getElementById: node, createElement: tag => new Element(tag), querySelector: node, querySelectorAll: () => [] };
const context = vm.createContext({ document, console, Map, Set, JSON, Number, String, Boolean, Math,
  fetch: async () => ({ ok: true, json: async () => ({ by_game: {}, tools: [], skills: [], config: {} }) }),
  requestAnimationFrame: fn => fn(), setInterval: () => 1, clearInterval() {}, location: { protocol: 'http:', host: 'test' } });
vm.runInContext(fs.readFileSync(path.join(__dirname, '../src/llmgo/server/static/app.js'), 'utf8'), context);
vm.runInContext(`
  if (state.pieces.length !== 32 || state.game !== 'chess') throw new Error('initial chess board');
  onEvent({type:'turn_started',seat:'white',prompt:'',tool_library:[]});
  onEvent({type:'choice',seat:'white',stage:'tool',status:'asking',turn_id:'0:white',prompt:'actual prompt',blackboard:[]});
  onEvent({type:'model_request',status:'started',seat:'white',stage:'tool',request_id:'0:white:1',prompt:'actual prompt',max_tokens:4096});
  onEvent({type:'model_request',status:'finished',seat:'white',stage:'tool',request_id:'0:white:1',prompt:'actual prompt',text:'A',reasoning:'<script>literal text</script>',output:'reasoning and A',raw_response:'raw response body',usage:{completion_tokens:23},tokens_exact:true,finish_reason:'stop',latency_ms:125});
`, context);
const trace = node('red-trace');
assert.equal(trace.querySelectorAll('details[data-request-id]').length, 1);
assert.match(trace.textContent, /23 tokens/);
assert.match(trace.textContent, /raw response body/);
assert.match(trace.textContent, /<script>literal text<\/script>/);
vm.runInContext(`onEvent({type:'turn_started',seat:'white',prompt:'next',tool_library:[]});`, context);
assert.match(node('red-archive').textContent, /raw response body/);
const events = [
  {type:'position',game:'chess',pieces:[],side:'white',last_move:null},
  {type:'model_request',status:'started',seat:'white',stage:'tool',request_id:'1'},
  {type:'model_request',status:'finished',seat:'white',stage:'tool',request_id:'1',output:'full thinking',raw_response:'raw replay'},
  {type:'tool_call',status:'started',seat:'white',tool:'preview',parameters:{uci:'e2e4'}},
  {type:'tool_call',status:'finished',seat:'white',tool:'preview',result:['preview exact result'],blackboard:['preview exact result']},
  {type:'fallback',seat:'white',stage:'move',reason:'length_without_choice',uci:'e2e4'},
  {type:'attempt',seat:'white',ok:true,uci:'e2e4',content:'<move>e2e4</move>',source:'random'},
  {type:'position',game:'chess',pieces:[],side:'black',last_move:{seat:'white',uci:'e2e4'}}
];
context.sample = events;
vm.runInContext(`
  replay.frames = buildFrames(sample); replay.id = 'smoke';
  if (replay.frames.length !== 2 || replay.frames[1].choices.length !== 5) throw new Error('missing trace events');
  if (modelOutput(replay.frames[1].choices) !== 'full thinking') throw new Error('duplicate output');
  showFrame(1);
`, context);
assert.match(node('red-trace').textContent, /raw replay/);
assert.match(node('red-trace').textContent, /length_without_choice/);
assert.equal(node('blackboard').textContent, 'preview exact result');
// The new protocol uses UCI choices and automatic tools; old letter replay above remains supported.
vm.runInContext(`
  onEvent({type:'turn_started',seat:'white',framework:'single_choice',tool_library:[]});
  onEvent({type:'tool_call',status:'finished',seat:'white',tool:'candidates',result:['e2e4: example observation'],blackboard:['e2e4: example observation'],data:{legal_count:20}});
  onEvent({type:'choice',stage:'move',status:'asking',seat:'white',turn_id:'0:white',prompt:'all legal UCI options',options:[{uci:'e2e3',label:'e2e3 (e3)'},{uci:'e2e4',label:'e2e4 (e4)'}],blackboard:[]});
  onEvent({type:'choice',stage:'move',status:'answered',seat:'white',uci:'e2e4',source:'model',options:[{uci:'e2e3',label:'e2e3 (e3)'},{uci:'e2e4',label:'e2e4 (e4)'}],blackboard:[]});
  if (choiceTrace([{type:'choice',stage:'move',status:'answered',uci:'e2e4',options:[{uci:'e2e3',label:'e2e3 (e3)'},{uci:'e2e4',label:'e2e4 (e4)'}]}]).includes('e2e3')) throw new Error('selected the wrong UCI label');
`, context);
assert.equal(node('red-content').textContent, 'e2e4');
assert.match(node('red-tools').textContent, /candidates/);
vm.runInContext(`onEvent({type:'choice',stage:'move',status:'answered',seat:'white',uci:'h8h7',source:'forced',blackboard:[]});`, context);
assert.match(node('status').textContent, /唯一合法步/);
// Live chunks are visible before finish, reconciled once, and separated across retries.
vm.runInContext(`
  onEvent({type:'turn_started',seat:'white',tool_library:[]});
  onEvent({type:'model_request',status:'started',seat:'white',request_id:'stream:1',stage:'move',prompt:'stream prompt'});
  onEvent({type:'token',seat:'white',request_id:'stream:1',channel:'reasoning',text:'Live ',replace:false});
  onEvent({type:'token',seat:'white',request_id:'stream:1',channel:'reasoning',text:'thought',replace:false});
`, context);
assert.equal(node('red-reasoning').textContent, 'Live thought');
assert.equal(node('red-content').textContent, '');
vm.runInContext(`
  onEvent({type:'token',seat:'white',request_id:'stream:1',channel:'content',text:'{"move":"e2e4"}',replace:false});
  onEvent({type:'model_request',status:'finished',seat:'white',request_id:'stream:1',stage:'move',reasoning:'Live thought',text:'{"move":"e2e4"}',output:'Live thought</think>{"move":"e2e4"}',streamed:true});
  onEvent({type:'token',seat:'white',request_id:'stream:1',channel:'reasoning',text:'late duplicate',replace:false});
`, context);
assert.equal(node('red-reasoning').textContent, 'Live thought');
assert.equal(node('red-content').textContent, '{"move":"e2e4"}');
vm.runInContext(`
  onEvent({type:'model_request',status:'started',seat:'white',request_id:'stream:2',stage:'move',prompt:'retry prompt'});
  onEvent({type:'token',seat:'white',request_id:'stream:2',channel:'reasoning',text:'provisional',replace:false});
  onEvent({type:'token',seat:'white',request_id:'stream:2',channel:'reasoning',text:'',replace:true});
  onEvent({type:'token',seat:'white',request_id:'stream:2',channel:'content',text:'{"move":"d2d4"}',replace:false});
  onEvent({type:'model_request',status:'finished',seat:'white',request_id:'stream:2',stage:'move',reasoning:'Second thought',text:'{"move":"d2d4"}'});
`, context);
assert.equal(node('red-reasoning').textContent, 'Live thought\n\nSecond thought');
assert.equal(node('red-content').textContent, '{"move":"d2d4"}');
context.partialSample = [
  {type:'position',game:'chess',pieces:[],side:'white',last_move:null},
  {type:'model_request',status:'started',seat:'white',stage:'move',request_id:'partial:1',prompt:'unfinished prompt'},
  {type:'token',seat:'white',request_id:'partial:1',channel:'reasoning',text:'Preserved partial thinking',replace:false}
];
vm.runInContext(`
  replay.frames = buildFrames(partialSample); replay.id = 'partial';
  if (replay.frames.length !== 2 || !replay.frames[1].pending) throw new Error('lost unfinished response');
  showFrame(1);
`, context);
assert.equal(node('red-reasoning').textContent, 'Preserved partial thinking');
assert.equal(node('red-prompt').textContent, 'unfinished prompt');
console.log('Harness UI trace, archive, replay and initial board: passed');
