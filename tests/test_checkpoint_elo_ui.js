'use strict';
const assert=require('node:assert/strict');
const {display}=require('../src/llmgo/server/static/rsi-elo.js');
assert.deepEqual(display({stage:'playing',completed_games:12,total_games:40,wins:8,draws:2,losses:2,score_rate:.75,formal:false,elo:1993}),
  {state:'对局中',progress:'12/40',wdl:'8 / 2 / 2',score:'75.0%',rating:'—'});
assert.equal(display({stage:'completed',formal:true,elo:2600,nominal_endpoint:true}).rating,'2600 *');
assert.equal(display({stage:'completed',formal:true,elo:1910}).rating,'1910');
assert.equal(display({stage:'Queued'}).state,'排队');
console.log('Checkpoint arena: partial ratings withheld, completed table scores and nominal endpoints passed');
