import {assignment,tensorFor} from './mapping-math.mjs';
import {openWeightFiles} from './weight-reader.mjs';
import {createTokenWeightView} from './token-weight-view.js';
import {createMACView} from './mac-view.js';
const range=(a,b)=>`${a}:${b}`;
const inclusive=(a,b)=>a===b-1?`${a}`:`${a}–${b-1}`;
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function createMappingView(host){
 let model,context,mode='logical',step=0,base=0,ffn='up',timer=null,identity='',weights=null,revision=0;
 host.innerHTML=`<div class="mapping-equation" aria-label="Matrix multiplication"><div><b id="map-left-symbol">X</b><span id="map-left-shape"></span></div><strong>×</strong><div><b id="map-right-symbol">W</b><span id="map-right-shape"></span></div><strong>=</strong><div><b>O</b><span id="map-out-shape"></span></div></div>
 <div class="map-formula" id="map-formula"></div>
 <div class="map-mode"><button data-map-mode="logical" aria-pressed="true">.m spatial / temporal directives</button><button data-map-mode="compiled" aria-pressed="false">Compiled 64 × 32 tiles</button></div>
 <div class="map-directives" id="map-directives"></div>
 <div class="map-controls"><label>Temporal C window <output id="map-step-text"></output><input type="range" id="map-time" aria-label="Temporal C window" min="0" step="1" value="0"></label><div><button id="map-prev">← C</button><button id="map-play">Play C sweep</button><button id="map-next">C →</button></div></div>
 <div id="map-overlap" class="map-overlap" hidden></div>
 <div class="map-matrices" id="map-matrices"></div>
 <div class="map-flow-note" id="map-flow-note"></div>
 <div class="map-spatial-head"><h3>Spatial assignments · simultaneous logical work</h3><div><button id="map-k-prev" aria-label="Previous output assignments">← K</button><span id="map-k-range"></span><button id="map-k-next" aria-label="Next output assignments">K →</button></div></div>
 <div class="map-pe-grid" id="map-pe-grid"></div>
 <p class="map-caption" id="map-assignment-note"></p>
 <div class="map-weight-head"><h3>Corresponding weights</h3><label id="map-ffn-label" hidden>FFN branch <select id="map-ffn"><option value="up">Up projection</option><option value="gate">Gate projection</option></select></label></div>
 <code id="map-tensor-path"></code><p class="map-caption" id="map-weight-orientation"></p>
 <label class="map-file">Show checkpoint values <input id="map-files" type="file" accept=".safetensors" multiple></label><p id="map-file-status" class="map-caption" role="status">Choose Qwen .safetensors file(s). Files are read locally in your browser; they are not uploaded.</p>
 <div class="map-weight-scroll"><table class="map-weight-table"><thead id="map-weight-cols"></thead><tbody id="map-weight-cells"></tbody></table></div>
 <p class="map-caption">Before a checkpoint is loaded, entries identify exact weight coordinates. Loaded values are checkpoint weights, not the quantized programmed-cell representation used by the hardware run.</p>
 <div class="map-reduction" id="map-reduction"></div>`;
 const get=id=>host.querySelector('#map-'+id);
 const tokenHost=document.createElement('section');tokenHost.className='token-run';host.prepend(tokenHost);const tokenView=createTokenWeightView(tokenHost);tokenView.onLoad(()=>get('files').click());
 const macHost=document.createElement('section');host.prepend(macHost);const macView=createMACView(macHost);macView.onLoad(()=>get('files').click());macView.onSelect(token=>tokenView.selectToken(token));tokenView.onSelect(token=>macView.selectToken(token));
 macView.onWindow(delta=>{stop();step=Math.max(0,Math.min(assignment(model,mode,step,base).steps-1,step+delta));paint();});
 function stop(){if(timer)clearInterval(timer);timer=null;get('play').textContent='Play C sweep';}
 function drawMatrix(label,rows,cols,row0,row1,col0,col1,color){
  const H=94,W=164;let svg=`<svg viewBox="0 0 184 120" role="img" aria-label="${escape(label)}; rows ${rows}, columns ${cols}; selected rows ${row0} through ${row1-1}, columns ${col0} through ${col1-1}"><rect x="8" y="8" width="164" height="94" fill="${color}" fill-opacity=".12" stroke="${color}" stroke-opacity=".5"/>`;
  for(let i=1;i<12;i++)svg+=`<path d="M ${8+i*W/12} 8 v 94 M 8 ${8+i*H/12} h 164" stroke="${color}" stroke-opacity=".18"/>`;
  const w=Math.max(2,W*(col1-col0)/cols),h=Math.max(2,H*(row1-row0)/rows),x=Math.min(8+W-w,8+col0/cols*W),y=Math.min(8+H-h,8+row0/rows*H);
  svg+=`<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${color}" fill-opacity=".85"/></svg>`;
  return `<div class="map-matrix"><b>${escape(label)}</b>${svg}<span>${rows} × ${cols}</span></div>`;
 }
 function paint(){
  if(!model)return;const m=model,a=assignment(m,mode,step,base);step=a.step;base=a.base;
  const attention=context.op===3||context.op===4,left=context.op===3?'Q':context.op===4?'A':'X',right=context.op===3?'Kᵀ':context.op===4?'V':'W';
  const tensor=tensorFor(context.block,context.op,ffn);
  tokenView.render(m,context,a,tensor);
  macView.render(m,context,a,tensor);
  get('left-symbol').textContent=left;get('right-symbol').textContent=right;
  get('left-shape').textContent=`Y × C = ${m.y} × ${m.c}`;get('right-shape').textContent=`C × K = ${m.c} × ${m.k}`;get('out-shape').textContent=`Y × K = ${m.y} × ${m.k}`;
  get('formula').textContent=`O[y,k] = Σ(c=0…${m.c-1}) ${left}[y,c] × ${right}[c,k]`;
  get('directives').innerHTML=['k','c','y'].map(d=>`<div><b>${d.toUpperCase()} · ${d==='c'?'temporal reduction':d==='k'?'spatial outputs':'spatial token groups'}</b><code>${m.d[d+'_map']}(${m.d[d+'_tile']},${m.d[d+'_stride']}) ${d.toUpperCase()};</code></div>`).join('');
  host.querySelectorAll('[data-map-mode]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.mapMode===mode));
  get('time').max=a.steps-1;get('time').value=step;get('step-text').textContent=`${step+1} / ${a.steps} · C ${inclusive(a.c0,a.c1)}`;
  get('prev').disabled=step===0;get('next').disabled=step===a.steps-1;get('k-prev').disabled=base===0;get('k-next').disabled=base+4>=a.kCount;
  get('overlap').hidden=!a.overlap;get('overlap').textContent=`Window size ${a.cSize}, stride ${a.cStride}: adjacent C windows overlap by ${a.cSize-a.cStride} channels. This is a directive-window preview, not a sum of overlapping window results. The compiler uses disjoint 64-row tiles; switch views to see the actual tile boundaries.`;
  const p=a.pes[0],last=a.pes[Math.min(3,a.pes.length-1)];
  get('matrices').innerHTML=drawMatrix(`${left}[Y,C]`,m.y,m.c,p.y0,p.y1,a.c0,a.c1,'#48e5c2')+'<strong>×</strong>'+drawMatrix(`${right}[C,K]`,m.c,m.k,a.c0,a.c1,p.k0,last.k1,'#79a9ff')+'<strong>→</strong>'+drawMatrix('O[Y,K]',m.y,m.k,p.y0,p.y1,p.k0,last.k1,'#f6bb68');
  get('flow-note').textContent=`Broadcast ${left}[${range(p.y0,p.y1)}, ${range(a.c0,a.c1)}] across output assignments → multiply by different ${right} columns → retain separate output partial sums.`;
  get('k-range').textContent=`K ${p.k0}–${last.k1-1}`;
  get('pe-grid').innerHTML=a.pes.map((pe,i)=>`<article class="map-pe" style="--slot-color:${['#79a9ff','#c29bff','#f6bb68','#48e5c2'][i%4]}"><h4>${a.compiled?'Tile slot':'Logical PE slot'} (${pe.yi},${pe.ki})</h4><div class="map-pe-weight">${right}[${range(pe.c0,pe.c1)}, ${range(pe.k0,pe.k1)}]</div><div>${left}[${range(pe.y0,pe.y1)}, ${range(pe.c0,pe.c1)}]</div><b class="map-multiply">× ${right}[${range(pe.c0,pe.c1)}, ${range(pe.k0,pe.k1)}]</b><div class="map-pe-sum">↓ Σ C → ΔO[${range(pe.y0,pe.y1)}, ${range(pe.k0,pe.k1)}]</div></article>`).join('');
  get('assignment-note').textContent=`${a.compiled?'Representative compiler tile assignments; physical concurrency and endpoint IDs are not given by the manifest.':'Representative logical spatial assignments, not measured physical PE addresses.'} Showing up to 4 of ${a.kCount} K assignments and 2 of ${a.yCount} Y groups. Index ranges are half-open [start:end). ${context.phase==='decode'&&attention&&context.scope!=='run'?'Decode Y=513 is the literal mapped extent; select Saved run for one query.':''}`;
  get('ffn-label').hidden=context.op!==6;get('tensor-path').textContent=tensor;
  get('weight-orientation').textContent=attention?'The right operand comes from the runtime KV cache; it is not a fixed model weight tensor.':`Qwen stores this tensor as [K,C] = [${m.k},${m.c}]. The multiplication above displays its transpose: W[c,k] = weight[k,c].`;
  const ks=Array.from({length:Math.min(4,m.k-p.k0)},(_,i)=>p.k0+i),cs=Array.from({length:Math.min(4,a.c1-a.c0)},(_,i)=>a.c0+i);
  get('weight-cols').innerHTML='<tr><th>C ↓ / K →</th>'+ks.map(k=>`<th>k = ${k}</th>`).join('')+'</tr>';
  get('weight-cells').innerHTML=cs.map(c=>`<tr><th>c = ${c}</th>${ks.map(k=>`<td>${attention?`${right}[${c},${k}]`:`weight[${k},${c}]`}</td>`).join('')}</tr>`).join('');
  const request=++revision;
  if(weights&&!attention){get('file-status').textContent='Reading the selected weight slice…';weights.read(tensor,ks,cs,[m.k,m.c]).then(result=>{if(request!==revision)return;get('weight-cells').innerHTML=cs.map((c,ci)=>`<tr><th>c = ${c}</th>${ks.map((k,ki)=>`<td><b>${Number(result.values[ci][ki]).toPrecision(5)}</b><small>weight[${k},${c}]</small></td>`).join('')}</tr>`).join('');get('file-status').textContent=`${result.dtype} checkpoint values · exact displayed indices · checkpoint identity has not been matched to the saved run.`;tokenView.setValues(result,cs,ks,`${result.dtype} checkpoint preview.`);}).catch(error=>{if(request===revision){get('file-status').textContent=error.message;tokenView.setStatus(error.message);}});}
  else if(weights&&attention)get('file-status').textContent='Attention uses runtime KV-cache values, not pretrained checkpoint weights.';
  get('reduction').innerHTML=`<b>Temporal progression: ${step===a.steps-1?'last C window':'next C window keeps the same output assignment'}</b><div>${a.overlap?'Preview only: overlapping directive windows must not be added twice.':`O[Y,K] = ${Array.from({length:Math.min(a.steps,3)},(_,i)=>`ΔO<sub>${i}</sub>`).join(' + ')}${a.steps>3?' + … + ΔO<sub>'+(a.steps-1)+'</sub>':''}. Each term uses a different C range; the output indices stay fixed.`}</div><span>${a.compiled?'Compiled tasks iterate token block → input row tile → output column tile. SpatialMap metadata does not prove simultaneous hardware scheduling.':`TemporalMap changes C from ${a.c0} to ${Math.min(m.c,(step+1)*a.cStride)} while SpatialMap assigns different K outputs and Y groups.`}</span>`;
 }
 get('time').oninput=e=>{stop();step=+e.target.value;paint();};
 get('prev').onclick=()=>{stop();step=Math.max(0,step-1);paint();};get('next').onclick=()=>{stop();step++;paint();};
 get('k-prev').onclick=()=>{stop();base=Math.max(0,base-4);paint();};get('k-next').onclick=()=>{stop();base+=4;paint();};
 get('play').onclick=()=>{if(timer){stop();return;}if(step===assignment(model,mode,step,base).steps-1)step=0;paint();get('play').textContent='Pause C sweep';timer=setInterval(()=>{const a=assignment(model,mode,step,base);if(step>=a.steps-1){stop();return;}step++;paint();},900);};
 host.querySelectorAll('[data-map-mode]').forEach(b=>b.onclick=()=>{stop();mode=b.dataset.mapMode;step=0;base=0;paint();});
 get('ffn').onchange=e=>{ffn=e.target.value;paint();};
 get('files').onchange=async e=>{stop();macView.pause();const files=Array.from(e.target.files);if(!files.length)return;get('file-status').textContent='Reading checkpoint headers…';try{weights=await openWeightFiles(files);paint();macView.setReader(weights);}catch(error){weights=null;macView.setReader(null);get('file-status').textContent=error.message;tokenView.setStatus(error.message);}};
 return {render(m,s){const key=[s.block,s.op,s.phase,s.scope].join(':');if(key!==identity){stop();step=0;base=0;identity=key;}model=m;context=s;paint();},pause(){stop();macView.pause();}};
}
