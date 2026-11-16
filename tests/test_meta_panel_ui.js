const assert=require('node:assert/strict');
const {view,stage}=require('../src/llmgo/server/static/rsi-meta.js');
const gpu={current:{model:'Qwen3.5-4B',execution:'gpu_job',thinking:true,attempts:[{status:'running',worker_stage:'loading_model'}]},meta_gpu_hours:1.25};
assert.equal(view(gpu).status,'加载模型');assert.match(view(gpu).title,/Thinking 开启/);
assert.equal(view(gpu).metaGpuHours,'1.250 GPU-hours');
assert.equal(stage({}, {status:'queued',activity:'thinking'}),'等待GPU排程');
assert.equal(stage({}, {status:'invalid',worker_stage:'completed'}),'无效尝试');
assert.equal(view({current:{execution:'api',thinking:null,status:'paused'}}).metaGpuHours,'不适用（API推理）');
assert.equal(view({current:{execution:'api'}}).stages.length,4);
assert.equal(view(gpu).stages.length,6);
console.log('Shared Meta panel API/GPU/invalid/loading projection passed');

assert.equal(stage({}, {status:'queued',worker_stage:'loading_model'}),'加载模型');
assert.equal(stage({}, {status:'running',worker_stage:'completed'}),'交接已落盘，等待任务结束');
assert.equal(stage({}, {status:'invalid',worker_stage:'completed'}),'无效尝试');

const large={current:{model:'Qwen3.8-27B',execution:'gpu_job',thinking:true,inference:{meta_precision:'bfloat16',meta_context_length:131072,meta_max_output_tokens:65536,meta_reasoning_effort:'xhigh'}}};
assert.match(view(large).settings,/131072/);assert.match(view(large).settings,/65536/);assert.match(view(large).settings,/bfloat16/);
