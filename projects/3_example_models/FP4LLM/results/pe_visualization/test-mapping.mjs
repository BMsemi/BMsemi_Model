import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {assignment,tensorFor} from './mapping-math.mjs';
import {decodeValue,openWeightFiles} from './weight-reader.mjs';
const data=JSON.parse(await readFile(new URL('./mapping-data.json',import.meta.url),'utf8'));
const make=(phase,op)=>{const d=data.phases[phase][op].layer;return {d,c:d.c,k:d.k,y:d.y};};
let m=make('prefill',0),a=assignment(m,'logical',0,0);
assert.deepEqual([a.c0,a.c1,a.steps,a.kCount,a.yCount],[0,32,64,2048,2]);
assert.deepEqual(a.pes.map(p=>[p.k0,p.k1,p.y0,p.y1]),[[0,1,0,256],[1,2,0,256],[2,3,0,256],[3,4,0,256],[0,1,256,512],[1,2,256,512],[2,3,256,512],[3,4,256,512]]);
const next=assignment(m,'logical',1,0);assert.deepEqual([next.c0,next.c1],[32,64]);assert.equal(next.pes[0].k0,a.pes[0].k0);
a=assignment(make('decode',0),'logical',1,0);assert.equal(a.overlap,true);assert.deepEqual([a.c0,a.c1],[64,192]);
a=assignment(make('decode',0),'compiled',1,0);assert.equal(a.overlap,false);assert.deepEqual([a.c0,a.c1,a.pes[0].k0,a.pes[0].k1],[64,128,0,32]);
a=assignment(make('decode',4),'compiled',8,0);assert.deepEqual([a.c0,a.c1],[512,513]);
for(const phase of Object.values(data.phases))for(const src of phase){const d=src.layer,m={d,c:d.c,k:d.k,y:d.y};const a=assignment(m,'compiled',0,0);assert.equal(a.steps,src.plan.row_tiles);assert.equal(a.kCount,src.plan.col_tiles);assert.equal(a.yCount,src.plan.token_blocks);for(const mode of ['logical','compiled']){const last=assignment(m,mode,1e9,1e9);assert.ok(last.c0<last.c1&&last.c1<=m.c);assert.ok(last.pes.every(p=>p.k0<p.k1&&p.k1<=m.k&&p.y0<p.y1&&p.y1<=m.y));}}
assert.equal(tensorFor(27,6,'gate'),'model.layers.27.mlp.gate_proj.weight');
const word=new DataView(new ArrayBuffer(2));word.setUint16(0,0xbf80,true);assert.equal(decodeValue(word,0,'BF16'),-1);word.setUint16(0,0x3c00,true);assert.equal(decodeValue(word,0,'F16'),1);
// Tiny synthetic fixture tests [K,C] storage -> [C,K] display, not model values.
const header=new TextEncoder().encode(JSON.stringify({tensor:{dtype:'F32',shape:[2,3],data_offsets:[0,24]}}));
const prefix=new DataView(new ArrayBuffer(8));prefix.setBigUint64(0,BigInt(header.length),true);
const file=new Blob([prefix.buffer,header,new Float32Array([1,2,3,4,5,6])]);
const reader=await openWeightFiles([file]);const tile=await reader.read('tensor',[0,1],[1,2],[2,3]);assert.deepEqual(tile.values,[[2,5],[3,6]]);
await assert.rejects(()=>reader.read('tensor',[0],[0],[3,2]),/dimensions/);
console.log('PASS: 448 layer plans, spatial K/Y assignments, temporal windows, decode overlap, tail tiles, BF16/F16 values and weight transpose.');
