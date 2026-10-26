import preview from './run-preview.js';
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function createTokenWeightView(host){
 let token=0,last=null,cached=null,select=()=>{};
 host.innerHTML=`<div class="token-run-heading"><h3>11 prompt tokens → shared weights → 11 output rows</h3><p id="token-prompt"></p></div><div class="token-product"><div><h4>Input tokens</h4><div id="token-inputs"></div></div><div class="token-shared"><h4>× Shared weight matrix</h4><code id="token-weight-name"></code><div id="token-weight-shape"></div><canvas id="token-weight-grid" width="256" height="256" role="img" aria-label="Weight tile: input channels along rows and output channels along columns"></canvas><div class="token-weight-numbers" id="token-weight-numbers"></div><p id="token-weight-status" role="status"></p><button type="button" id="token-load-weights">Load numerical weights</button></div><div><h4>→ Output rows</h4><div id="token-outputs"></div></div></div><p id="token-row-equation" class="token-row-equation" aria-live="polite"></p>`;
 const q=id=>host.querySelector('#token-'+id);
 q('prompt').textContent=preview.prompt;
 function render(m,context,a,tensor){
  last={m,context,a,tensor};host.hidden=context.scope!=='run'||context.phase!=='prefill';if(host.hidden)return;
  const attention=[3,4].includes(context.op),left=context.op===3?'Q':context.op===4?'A':'X',right=context.op===3?'Kᵀ':context.op===4?'V':'W';
  q('inputs').innerHTML=preview.tokens.map(t=>`<button type="button" class="token-row" data-token="${t.position}" aria-pressed="${t.position===token}" aria-label="Token ${t.position}: ${esc(t.text)}; ID ${t.id}"><span>${t.position}</span><b>${esc(t.text.replaceAll(' ','␣'))}</b><small>${t.id}</small></button>`).join('');
  q('outputs').innerHTML=preview.tokens.map(t=>`<button type="button" class="token-row token-output" data-token="${t.position}" aria-pressed="${t.position===token}"><span>${t.position}</span><b>${context.op===0?'Q':'O'}[${t.position}, :]</b></button>`).join('');
  host.querySelectorAll('[data-token]').forEach(b=>b.onclick=()=>{token=+b.dataset.token;render(m,context,a,tensor);select(token);});
  q('weight-name').textContent=tensor;q('weight-shape').textContent=`${m.c} input channels × ${m.k} outputs`;
  q('row-equation').textContent=`Token ${token} “${preview.tokens[token].text}”: ΔO[${token}, k] = Σ(c=${a.c0}…${a.c1-1}) ${left}[${token},c] × ${right}[c,k]. All 11 rows reuse the same ${right} slice; tokens are not separate weight copies.`;
  q('weight-status').textContent=attention?'Runtime KV-cache values are required for this operand.':'Outlined cells are weight locations. Load a checkpoint to reveal the numerical values.';
  q('weight-numbers').innerHTML='<table><tbody>'+Array.from({length:Math.min(4,a.c1-a.c0)},(_,r)=>'<tr>'+Array.from({length:Math.min(4,m.k-a.pes[0].k0)},(_,c)=>`<td>W[${a.c0+r},${a.pes[0].k0+c}]</td>`).join('')+'</tr>').join('')+'</tbody></table>';
  draw(null);
  if(cached&&cached.key===[tensor,a.c0,a.pes[0].k0].join(':'))setValues(...cached.args);
 }
 function draw(values){
  const canvas=q('weight-grid'),ctx=canvas.getContext('2d'),n=values?.length||8,k=values?.[0]?.length||8;
  ctx.clearRect(0,0,256,256);ctx.fillStyle='#10233c';ctx.fillRect(0,0,256,256);
  const max=values?Math.max(...values.flat().map(Math.abs),1e-12):1;
  for(let r=0;r<n;r++)for(let c=0;c<k;c++){const v=values?.[r]?.[c];if(v!==undefined){const alpha=.15+.85*Math.abs(v)/max;ctx.fillStyle=v<0?`rgba(249,159,93,${alpha})`:`rgba(107,173,255,${alpha})`;ctx.fillRect(c*256/k+1,r*256/n+1,256/k-2,256/n-2);}else{ctx.strokeStyle='#466f9c';ctx.strokeRect(c*32+1,r*32+1,30,30);}}
 }
 function setValues(result,cs,ks,status){if(!last||host.hidden)return;cached={key:[last.tensor,last.a.c0,last.a.pes[0].k0].join(':'),args:[result,cs,ks,status]};draw(result.values);q('weight-numbers').innerHTML='<table><tbody>'+result.values.map(row=>'<tr>'+row.map(v=>`<td>${Number(v).toPrecision(3)}</td>`).join('')+'</tr>').join('')+'</tbody></table>';q('weight-status').textContent=`${status} C ${cs[0]}–${cs.at(-1)}, K ${ks[0]}–${ks.at(-1)}. Color: blue ≥ 0, orange < 0.`;}
 return {render,setValues,onSelect(fn){select=fn;},selectToken(value){token=value;if(last)render(last.m,last.context,last.a,last.tensor);},onLoad(fn){q('load-weights').onclick=fn;},setStatus(text){q('weight-status').textContent=text;}};
}
