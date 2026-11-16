const assert=require('node:assert/strict');
const {rows}=require('../src/llmgo/server/static/rsi-meta-acceptance.js');
assert.deepEqual(rows({}),[]);
const r=rows({models:{qwen38:{task_id:7,pool:'model-new',priority:'PREEMPTABLE',worker_stage:'long_context_check',
 benchmark:{completed:3,total:60,summary:{cases:3,passed:2,schema_measured_cases:3,first_check_schema_passes:1}},long_context:{status:'passed'}}}})[0];
assert.equal(r.model,'Qwen3.8-27B');assert.equal(r.progress,'3 / 60');assert.equal(r.passed,'2 / 3');assert.equal(r.schema,'1 / 3');
assert.equal(r.context,'passed');assert.match(r.task,/PREEMPTABLE/);
console.log('Independent acceptance panel separates progress, factual and schema results');
