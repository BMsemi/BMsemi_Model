import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import data from './activation-preview.js';
import {activationNames,savedValue,prefixMAC,vectorMAC,crossbarSlice} from './mac-math.mjs';

const source=await readFile(new URL('../x1_prompt_run/software_crossbar/x1_prompt_run_bf16_values.txt',import.meta.url));
assert.equal(createHash('sha256').update(source).digest('hex'),data.sha256);
// Check every exported BF16 word against its source, including token row boundaries.
const exported=new Map(Object.entries(data.tensors).map(([name,item])=>[name,Buffer.from(item.words,'base64')]));
let count=0;
for(const line of source.toString('utf8').split('\n')){
 if(!line.startsWith('prefill\t')&&!line.startsWith('decode\t'))continue;
 const [phase,name,flat,,word]=line.split('\t');
 assert.equal(exported.get(`${phase}/${name}`).readUInt16LE(Number(flat)*2),parseInt(word,16));count++;
}
const cache=new Map(),input=activationNames(0).input;
assert.equal(savedValue(data,'prefill',input,0,0,cache),-0.07763671875);
assert.equal(savedValue(data,'prefill',input,0,1,cache),-0.02099609375);
assert.equal(savedValue(data,'prefill',input,10,0,cache),-0.030029296875);
assert.equal(savedValue(data,'prefill',input,10,2047,cache),0.11328125);
assert.equal(savedValue(data,'decode',input,0,0,cache),-0.047119140625);
assert.equal(savedValue(data,'decode',input,1,0,cache),null);
assert.equal(activationNames(3).input,null);
assert.equal(activationNames(6,'model.layers.0.mlp.gate_proj.weight').output,'layer_00_mlp_gate_proj_output');
assert.equal(activationNames(7).input,'layer_00_mlp_gated_activation_before_down_proj');
// Synthetic arithmetic fixture only: a missing operand must never become a zero product.
assert.equal(prefixMAC([2,-3,4],[5,2,-1],0),10);
assert.equal(prefixMAC([2,-3,4],[5,2,-1],2),0);
assert.equal(prefixMAC([2,null],[5,2],1),null);
assert.equal(prefixMAC([2,-3],[null,2],1),null);
// The crossbar result uses every row, including rows outside the four-row detail crop.
const vector=Array.from({length:32},(_,i)=>i+1);
assert.equal(vectorMAC(vector,Array(32).fill(1)),528);
assert.equal(vectorMAC(vector,Array.from({length:32},(_,i)=>i===31?2:0)),64);
assert.equal(vectorMAC(vector,Array.from({length:32},(_,i)=>i===31?null:1)),null);
assert.deepEqual(crossbarSlice({c0:0,c1:64},1),{index:1,count:2,c0:32,c1:64});
assert.deepEqual(crossbarSlice({c0:64,c1:192},3),{index:3,count:4,c0:160,c1:192});
assert.deepEqual(crossbarSlice({c0:512,c1:513}),{index:0,count:1,c0:512,c1:513});
assert.deepEqual(crossbarSlice({c0:16,c1:48},0),{index:0,count:2,c0:16,c1:32});
assert.deepEqual(crossbarSlice({c0:16,c1:48},1),{index:1,count:2,c0:32,c1:48});
console.log('PASS: full parallel vector sums, hidden-row contributions, 32-row slice boundaries and partial tails.');
console.log(`PASS: ${count} original BF16 words, all 11 token rows, decode indexing, projection inputs, prefix sums and missing operands.`);
