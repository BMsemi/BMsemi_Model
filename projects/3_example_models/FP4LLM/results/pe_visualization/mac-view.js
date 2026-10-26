import activations from './activation-preview.js';
import preview from './run-preview.js';
import {activationNames,savedValue,vectorMAC,crossbarSlice} from './mac-math.mjs';

const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number=v=>v===null||v===undefined?'missing':Number(v).toPrecision(6);

export function createMACView(host){
 let state=null,reader=null,values=null,token=0,c=0,ki=0,stage=0,shard=0,timer=null,serial=0,key='',load=()=>{},select=()=>{},windowChange=()=>{};
 const cache=new Map();
 host.className='mac-view';
 host.innerHTML=`<h3>Input vector → parallel crossbar multiply</h3>
 <p class="mac-intro">All active rows receive their input signals together. Each input drives its row’s weights, and column currents sum the row contributions concurrently.</p>
 <div class="mac-tokens" aria-label="Select input token"></div><p class="mac-tensor"></p>
 <div class="mac-toolbar"><button data-action="play">▶ Play parallel tile</button><button data-action="window-prev">← Temporal window</button><button data-action="window-next">Temporal window →</button><label>32-row crossbar slice <select class="mac-slice" aria-label="32-row crossbar slice"></select></label></div>
 <div class="mac-activity"></div>
 <div class="mac-toolbar"><button data-action="prev" aria-label="Inspect previous input channel">← Inspect</button><button data-action="next" aria-label="Inspect next input channel">Inspect →</button><label>Inspect a cell · all rows stay active <output></output><input type="range" min="0" value="0" step="1" aria-label="Inspect input channel without changing execution"></label></div>
 <div class="mac-stage" role="status"></div><div class="mac-diagram"></div>
 <div class="mac-calculation" aria-live="polite"></div>
 <div class="mac-weight-message"><span></span><button data-action="load">Load Qwen weight file</button></div>
 <details class="mac-source"><summary>Value sources and PE mapping</summary><p></p></details>`;
 const q=s=>host.querySelector(s),button=name=>q(`[data-action="${name}"]`);
 function stop(){if(timer)clearInterval(timer);timer=null;button('play').textContent='▶ Play parallel tile';}
 function inputAt(channel){if(!state||state.context.scope!=='run'||state.context.block!==0)return null;return savedValue(activations,state.context.phase,activationNames(state.context.op,state.tensor).input,token,channel,cache);}
 function weightAt(channel,index){return values?.[channel-state.a.c0]?.[index]??null;}
 function tileSum(index){const slice=crossbarSlice(state.a,shard),cs=Array.from({length:slice.c1-slice.c0},(_,i)=>slice.c0+i);return vectorMAC(cs.map(inputAt),cs.map(channel=>weightAt(channel,index)));}
 function draw(){
  if(!state)return;
  const {m,context,a,tensor,ks}=state,attention=[3,4].includes(context.op),names=activationNames(context.op,tensor),slice=crossbarSlice(a,shard);
  q('.mac-tensor').textContent=`${tensor} · ${attention?'runtime operand':'W[c,k] = checkpoint weight[k,c]'} · representative 32 × 32 macro, C ${slice.c0}–${slice.c1-1}. Four rows and four output columns enlarged below; this is a detail crop.`;
  q('.mac-slice').innerHTML=Array.from({length:slice.count},(_,i)=>{const s=crossbarSlice(a,i);return `<option value="${i}" ${i===slice.index?'selected':''}>C ${s.c0}–${s.c1-1}</option>`;}).join('');
  button('window-prev').disabled=a.step===0;button('window-next').disabled=a.step===a.steps-1;
  q('.mac-activity').innerHTML=`<b>${slice.c1-slice.c0} active input rows together · ${['vector applied','multiply and column sum','readout'][stage]}</b><div>${Array.from({length:slice.c1-slice.c0},(_,i)=>`<span title="X[${token},${slice.c0+i}] drives row ${slice.c0+i}" aria-label="Channel ${slice.c0+i} active"></span>`).join('')}</div><small>The row selector below changes the detail crop only. Time is spent on other tiles, input encodings and readout—not a scalar walk through these cells.</small>`;
  const tokens=context.phase==='prefill'?preview.tokens:[{position:0,text:'decode',id:1790}];
  if(token>=Math.min(tokens.length,m.y))token=0;
  q('.mac-tokens').innerHTML=tokens.slice(0,m.y).map(t=>`<button data-token="${t.position}" aria-pressed="${token===t.position}"><small>${t.position} · ID ${t.id}</small>${esc(t.text.replaceAll(' ','␣'))}</button>`).join('');
  host.querySelectorAll('[data-token]').forEach(b=>b.onclick=()=>{stop();token=+b.dataset.token;stage=0;draw();select(token);});
  q('output').textContent=`c = ${c}`;
  q('input').min=slice.c0;q('input').max=slice.c1-1;q('input').value=c;
  button('prev').disabled=c===slice.c0;button('next').disabled=c===slice.c1-1;
  q('.mac-stage').textContent=`${['1 · Apply the whole input vector to all active rows','2 · All cell products and column sums form together','3 · Read out column partials for digital accumulation'][stage]} · token ${token}, C ${slice.c0}–${slice.c1-1}`;
  const begin=slice.c0+Math.floor((c-slice.c0)/4)*4,cs=Array.from({length:Math.min(4,slice.c1-begin)},(_,i)=>begin+i);
  const colx=i=>310+i*158,width=260+ks.length*158,height=410;
  let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="All ${slice.c1-slice.c0} input rows for token ${token} are active together. Four rows shown in detail, with concurrent column sums."><defs><marker id="mac-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#56e8c2"/></marker></defs><text x="12" y="22" fill="#8ef3d5">Input vector X[${token},C]</text>`;
  ks.forEach((k,i)=>{svg+=`<text x="${colx(i)}" y="22" text-anchor="middle" fill="#adcdff">Crossbar column ${k%32}</text><text x="${colx(i)}" y="43" text-anchor="middle" fill="#9fb4cc">output k = ${k}</text><path d="M${colx(i)} 62V337" stroke="#ddad64" stroke-width="2"/>`;});
  cs.forEach((channel,r)=>{
   const y=88+r*65,inspected=channel===c,x=inputAt(channel);
   svg+=`<rect x="0" y="${y-25}" width="${width}" height="56" rx="6" fill="#1a443b" opacity=".6"/>`;
   svg+=`<path d="M155 ${y}H${colx(ks.length-1)+62}" stroke="#56e8c2" stroke-width="3" marker-end="url(#mac-arrow)"/><rect x="7" y="${y-23}" width="150" height="48" rx="5" fill="#143a32" stroke="#83f3d3"/><text x="17" y="${y-4}" fill="#a4d2c8">X[${token},${channel}]</text><text x="17" y="${y+16}" fill="#a8ffe4" font-weight="bold">${number(x)}</text>`;
   ks.forEach((k,i)=>{
    const wx=colx(i),w=weightAt(channel,i);
    svg+=`<g data-cell="${channel}:${i}" tabindex="0" role="button" aria-label="Inspect input channel ${channel} times weight output ${k}; all rows remain active"><rect x="${wx-65}" y="${y-23}" width="130" height="48" rx="5" fill="#254c78" stroke="${inspected&&ki===i?'#fff0b1':'#8fbfff'}" stroke-width="2"/><text x="${wx}" y="${y-4}" text-anchor="middle" fill="#b6d4ff">${attention?'R':'W'}[${channel},${k}]</text><text x="${wx}" y="${y+16}" text-anchor="middle" fill="#ecf5ff" font-weight="bold">${w===null?(attention?'not exported':'load weights'):number(w)}</text></g>`;
    if(stage===1)svg+=`<circle r="5" fill="#ffd289"><animateMotion dur="1.1s" repeatCount="1" path="M${wx} ${y+25}L${wx} 338"/></circle>`;
    if(stage===0)svg+=`<circle r="5" fill="#adffe9"><animateMotion dur="1.1s" repeatCount="1" path="M${wx-88} ${y}L${wx-66} ${y}"/></circle>`;
   });
  });
  ks.forEach((k,i)=>{const wx=colx(i),sum=tileSum(i);svg+=`<rect x="${wx-72}" y="338" width="144" height="62" rx="6" fill="#3b2c16" stroke="#ddad64"/><text x="${wx}" y="359" text-anchor="middle" fill="#ffdda6">Σ ${slice.c1-slice.c0} rows → k=${k}</text><text x="${wx}" y="383" text-anchor="middle" fill="#ffe8c5">${stage===0?'await vector':sum===null?'needs weights':number(sum)}</text>`;});
  q('.mac-diagram').innerHTML=svg+'</svg>';
  host.querySelectorAll('[data-cell]').forEach(cell=>{const choose=()=>{[c,ki]=cell.dataset.cell.split(':').map(Number);draw();};cell.onclick=choose;cell.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}};});
  const k=ks[ki],x=inputAt(c),w=weightAt(c,ki),product=x===null||w===null?null:x*w,sum=tileSum(ki);
  const recorded=context.scope==='run'&&context.block===0?savedValue(activations,context.phase,names.output,token,k,cache):null;
  q('.mac-calculation').innerHTML=`<b>Inspect one cell · other cells compute concurrently</b><div><span class="mac-x">X[${token},${c}] = ${number(x)}</span> × <span class="mac-w">${attention?'R':'W'}[${c},${k}] = ${number(w)}</span> = <span class="mac-sum">${product===null?'product unavailable':number(product)}</span></div><div>Whole-slice partial: ΔO[${token},${k}] = Σ(c=${slice.c0}…${slice.c1-1}) X[${token},c] × W[c,${k}] = ${number(sum)}</div><small>The sum includes all ${slice.c1-slice.c0} active rows, including those outside the four-row crop. Larger C tiles combine multiple 32-row slice results digitally. ${recorded===null?'':`Saved full output O[${token},${k}] = ${number(recorded)}; it includes all input channels and the run’s quantization.`}</small>`;
  q('.mac-source p').textContent=`${names.input?`Input source: ${names.input}, original BF16 words from ${activations.source}. `:'The saved export does not contain the per-head attention operands. '}Weight source: ${tensor}. Display W[c,k] equals checkpoint weight[k,c]. One input vector drives the active rows together, and every connected column forms a dot product. The drawing groups four adjacent K columns into a representative 32 × 32 macro; logical SpatialMap assignments below do not establish physical placement. C tile ${Math.floor(c/64)}, 32-row shard ${Math.floor(c%64/32)}, cell row ${c%32}; K tile ${Math.floor(k/32)}, cell column ${k%32}. Animation stages are explanatory, not clock cycles. PWM/bit-plane encoding, row-group limits, shared converters and tile scheduling can require multiple phases; this view illustrates parallel crossbar arithmetic and does not assert one-cycle FPGA execution. The ZCU106 needs a digital implementation or an external crossbar. Numerical sums use unquantized JavaScript arithmetic, not measured currents or a replay of the integer run. ${a.overlap?'Overlapping directive windows are inspected separately and never added together.':''}`;
 }
 async function fetchWeights(){
  const ticket=++serial;values=null;
  if(!state)return;
  const {m,a,tensor,ks,context}=state;
  if([3,4].includes(context.op)){q('.mac-weight-message span').textContent='Attention needs per-head runtime operands, which this saved export does not contain.';button('load').hidden=true;draw();return;}
  button('load').hidden=false;
  if(!reader){q('.mac-weight-message span').textContent=`${context.scope==='run'&&context.block===0?'Saved input values are connected.':'Saved input values are available in Saved run, block 1 only.'} Numerical weights are missing: select the Qwen .safetensors checkpoint to see products and sums.`;draw();return;}
  q('.mac-weight-message span').textContent='Reading this C window and the four displayed output columns…';draw();
  try{
   const cs=Array.from({length:a.c1-a.c0},(_,i)=>a.c0+i),result=await reader.read(tensor,ks,cs,[m.k,m.c]);
   if(ticket!==serial)return;values=result.values;
   q('.mac-weight-message span').textContent=`${result.dtype} checkpoint values loaded. Checkpoint identity is not verified against this saved run; sums show the mathematical product, before hardware quantization.`;draw();
  }catch(error){if(ticket===serial){q('.mac-weight-message span').textContent=error.message;draw();}}
 }
 button('load').onclick=()=>load();
 button('prev').onclick=()=>{c=Math.max(crossbarSlice(state.a,shard).c0,c-1);draw();};
 button('next').onclick=()=>{c=Math.min(crossbarSlice(state.a,shard).c1-1,c+1);draw();};
 q('input').oninput=e=>{c=+e.target.value;draw();};
 q('.mac-slice').onchange=e=>{stop();shard=+e.target.value;c=crossbarSlice(state.a,shard).c0;stage=0;draw();};
 button('window-prev').onclick=()=>{stop();windowChange(-1);};button('window-next').onclick=()=>{stop();windowChange(1);};
 button('play').onclick=()=>{if(timer){stop();draw();return;}stage=0;timer=setInterval(()=>{stage++;if(stage===2)stop();draw();},1400);button('play').textContent='Ⅱ Pause parallel tile';draw();};
 return {
  render(m,context,a,tensor){const next=[context.phase,context.scope,context.block,context.op,tensor,a.compiled,a.c0,a.c1,a.base].join(':');const k0=a.pes[0].k0;state={m,context:{...context},a,tensor,ks:Array.from({length:Math.min(4,m.k-k0,32-k0%32)},(_,i)=>k0+i)};if(key!==next){stop();key=next;c=a.c0;ki=0;shard=0;stage=0;fetchWeights();}else draw();},
  setReader(value){stop();reader=value;fetchWeights();},pause:stop,onLoad(fn){load=fn;},onWindow(fn){windowChange=fn;},onSelect(fn){select=fn;},selectToken(value){stop();token=value;stage=0;draw();}
 };
}
