import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import data from './mapping-data.js';
import { createMappingView } from './mapping-view.js';
import preview from './run-preview.js';

const $ = id => document.getElementById(id);
const colors = {weights:0x79a9ff,activations:0x48e5c2,cache:0xb69aff,partials:0xf6bb68,wire:0x46647e,board:0x101c2b};
const roles = ['Query projection','Key projection','Value projection','Attention scores','Attention × V','Output projection','FFN gate / up','FFN down'];
const shortRoles = ['Query','Key','Value','Q × Kᵀ','A × V','Output','Gate / up','Down'];
const stages = ['Load','Program','Compute','SAR/TDC','Accumulate','NoC'];
const state = {phase:'prefill',block:0,op:0,view:'mapping',scope:'run',row:0,col:0,token:0,stage:-1,playing:false,time:0};
const mappingPanel=document.createElement('div');mappingPanel.id='mapping-panel';$('viewport').append(mappingPanel);
const mappingView=createMappingView(mappingPanel);
const mappingButton=document.createElement('button');mappingButton.dataset.view='mapping';mappingButton.innerHTML='<span>00</span> Spatial / temporal <b>↗</b>';document.querySelector('.views').prepend(mappingButton);
const fmt = n => n.toLocaleString('en-US');
const objects = [], labels = [], dynamic = [], hitTargets = [];
let scene, camera, renderer, controls, world, frameTime=0, disposed=false, transition=null;
const matrix = new THREE.Matrix4(), dummy = new THREE.Object3D();
const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2();
let rowSweep=null, pulsePath=[], particleSystem=null, highlighted=null;
const stageColors=[colors.activations,colors.weights,colors.activations,colors.partials,colors.partials,colors.cache];

function selected(){
 const source=data.phases[state.phase][state.block*8+state.op],d=source.layer;
 const ctx=state.scope==='run'?(state.phase==='prefill'?11:12):(state.phase==='prefill'?512:513);
 const y=state.scope==='run'?(state.phase==='prefill'?11:1):d.y;
 const c=state.op===4?ctx:d.c,k=state.op===3?ctx:d.k;
 const rows=Math.ceil(c/64),cols=Math.ceil(k/32),tokens=Math.ceil(y/d.y_tile);
 return {source,d,c,k,y,ctx,rows,cols,tokens,c0:state.row*64,c1:Math.min(c,(state.row+1)*64),k0:state.col*32,k1:Math.min(k,(state.col+1)*32),t0:state.token*d.y_tile,t1:Math.min(y,(state.token+1)*d.y_tile)};
}
function material(color,opacity=1){return new THREE.MeshStandardMaterial({color,metalness:.35,roughness:.52,transparent:opacity<1,opacity});}
const gridTextures=new Map();
function gridTexture(color){
 if(gridTextures.has(color))return gridTextures.get(color);
 const canvas=document.createElement('canvas');canvas.width=canvas.height=512;const ctx=canvas.getContext('2d');
 ctx.fillStyle='#'+new THREE.Color(color).multiplyScalar(.45).getHexString();ctx.fillRect(0,0,512,512);
 ctx.strokeStyle='#'+new THREE.Color(color).multiplyScalar(.85).getHexString();ctx.lineWidth=1;
 for(let i=0;i<=32;i++){ctx.beginPath();ctx.moveTo(i*16,0);ctx.lineTo(i*16,512);ctx.moveTo(0,i*16);ctx.lineTo(512,i*16);ctx.stroke();}
 const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;texture.anisotropy=4;gridTextures.set(color,texture);return texture;
}
function box(parent,x,y,z,w,h,d,color,{grid=false,opacity=1,edges=true,click=null}={}){
 const geo=new THREE.BoxGeometry(w,h,d),mat=material(color,opacity);if(grid)mat.map=gridTexture(color);
 const mesh=new THREE.Mesh(geo,mat);mesh.position.set(x,y,z);parent.add(mesh);objects.push(mesh);
 if(edges){const edge=new THREE.LineSegments(new THREE.EdgesGeometry(geo),new THREE.LineBasicMaterial({color,transparent:true,opacity:.42}));mesh.add(edge);}
 if(click){mesh.userData.action=click;hitTargets.push(mesh);}return mesh;
}
function wire(parent,points,color=colors.wire,opacity=.65){const g=new THREE.BufferGeometry().setFromPoints(points.map(p=>new THREE.Vector3(...p)));const l=new THREE.Line(g,new THREE.LineBasicMaterial({color,transparent:true,opacity}));parent.add(l);return l;}
function label(text,position,{major=false}={}){const el=document.createElement('div');el.className='scene-label'+(major?' major':'');el.textContent=text;$('labels').append(el);labels.push({el,position:new THREE.Vector3(...position)});return el;}
function clearScene(){
 if(world){world.traverse(o=>{if(o.geometry)o.geometry.dispose();if(o.material){for(const m of Array.isArray(o.material)?o.material:[o.material])m.dispose();}});scene.remove(world);}
 for(const l of labels)l.el.remove();labels.length=0;objects.length=0;hitTargets.length=0;dynamic.length=0;
 world=new THREE.Group();scene.add(world);rowSweep=null;highlighted=null;pulsePath=[];particleSystem=null;
}
function init(){
 scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(42,1,.05,600);
 renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,powerPreference:'high-performance'});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor(0x080e18,0);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.35;$('canvas-host').append(renderer.domElement);
 scene.add(new THREE.AmbientLight(0xb0cbef,1.6));const key=new THREE.DirectionalLight(0xd5e8ff,2.5);key.position.set(10,22,18);scene.add(key);const rim=new THREE.DirectionalLight(0x66bfff,2);rim.position.set(-15,6,-10);scene.add(rim);
 controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.07;controls.minDistance=4;controls.maxDistance=140;controls.maxPolarAngle=Math.PI*.93;
 controls.addEventListener('start',()=>{transition=null;});
 new ResizeObserver(resize).observe($('canvas-host'));resize();
 let down=null;renderer.domElement.addEventListener('pointerdown',e=>down=[e.clientX,e.clientY]);
 renderer.domElement.addEventListener('pointerup',e=>{if(!down||Math.hypot(e.clientX-down[0],e.clientY-down[1])>5)return;const rect=renderer.domElement.getBoundingClientRect();pointer.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);raycaster.setFromCamera(pointer,camera);const hits=raycaster.intersectObjects(hitTargets,false);if(hits.length)hits[0].object.userData.action?.(hits[0]);});
 buildScene();requestAnimationFrame(animate);
}
function resize(){if(!renderer)return;const r=$('canvas-host').getBoundingClientRect();renderer.setSize(r.width,r.height);camera.aspect=r.width/Math.max(1,r.height);camera.updateProjectionMatrix();}
function cameraPreset(animate=true){
 const narrow=camera.aspect<.8;
 const presets={model:{p:[27,18,narrow?57:45],t:[-1.7,0,0]},crossbar:{p:[21,24,narrow?40:30],t:[0,0,0]}};
 const p=presets[state.view];
 if(animate&&!matchMedia('(prefers-reduced-motion:reduce)').matches)transition={from:camera.position.clone(),target:controls.target.clone(),to:new THREE.Vector3(...p.p),look:new THREE.Vector3(...p.t),start:performance.now()};
 else{camera.position.set(...p.p);controls.target.set(...p.t);controls.update();}
}
function modelScene(){
 const m=selected(),group=world;
 // A compact, selectable stack gives every transformer block a visible place.
 for(let b=0;b<28;b++){
  const y=12.3-b*.88,active=b===state.block;
  box(group,-11,y,0,3.1,.28,3.4,active?colors.activations:0x314d72,{grid:true,opacity:active?1:.65,click:()=>chooseBlock(b)});
  if(b===0||b===27||active)label(`Block ${b+1}`,[-13.1,y,.4],{major:active});
 }
 wire(group,[[-11,13.5,0],[-11,-12.5,0]],colors.activations,.4);
 label('QWEN3 · 28 BLOCKS',[-11,14.5,0],{major:true});
 wire(group,[[-9.4,12.3-state.block*.88,1],[ -7.7,12.3-state.block*.88,1],[-7.7,0,1],[-5.8,0,1]],colors.activations,.55);
 const nodes=[[-4,8,0,3,3],[0,8,0,2.3,3],[4,8,0,2.3,3],[-2,4,0,4.2,3],[2,1,0,4.2,3],[0,-2,0,5,3],[-1.5,-5.4,0,6.2,3.5],[0,-9,0,6.2,3.5]];
 const tensorDepth=state.phase==='decode'?0.25:2;
 box(group,0,11.2,0,6,.16,tensorDepth,colors.activations,{grid:true});label(`${m.y} token${m.y===1?'':'s'} · 2,048 channels`,[0,12.2,0],{major:true});label('RMSNorm · Q/K norm + RoPE after projections',[.2,10.1,2]);
 if(state.scope==='run'&&state.phase==='prefill')for(let i=0;i<11;i++){
  const y=13+(10-i)*.56;box(group,8.5,y,0,4.5,.16,.7,colors.activations,{grid:true});label(`${i} · ${preview.tokens[i].text.replaceAll(' ','␣')}`,[8.5,y,.65]);
 }
 nodes.forEach(([x,y,z,w,d],i)=>{
  const active=i===state.op;const mesh=box(group,x,y,z,w,.28,d,i===3||i===4?colors.cache:colors.weights,{grid:true,opacity:active?1:.66,click:()=>chooseOp(i)});
  if(active){
   highlighted=mesh;
   const glow=box(group,x,y-.2,z,w+.22,.05,d+.22,colors.activations,{opacity:.3,edges:false});dynamic.push(glow);
   // Selected C/K tile sits at its actual normalized location on the weight plane.
   box(group,x-w/2+w*(m.k0+(m.k1-m.k0)/2)/m.k,y+.18,z-d/2+d*(m.c0+(m.c1-m.c0)/2)/m.c,Math.max(.045,w*(m.k1-m.k0)/m.k),.035,Math.max(.045,d*(m.c1-m.c0)/m.c),colors.partials,{edges:false});
  }
  label(['X × Wq','X × Wk','X × Wv','Q × Kᵀ','A × V','H × Wo','X × Wgate/up','G × Wdown'][i],[x,y+.9,z+.6],{major:active});
  box(group,x,y-.7,z,w,.11,tensorDepth*.62,colors.activations,{grid:true,opacity:.65});
 });
 label('SwiGLU · gate and up share the FFN_1 mapping',[.2,-7.4,1]);
 label('Causal mask → softmax',[-2,2.65,1.6]);
 wire(group,[[0,11.2,0],[0,9.6,0],[-4,9.6,0],[-4,8.2,0]],colors.activations);
 wire(group,[[0,9.6,0],[4,9.6,0],[4,8.2,0]],colors.activations);wire(group,[[0,9.6,0],[0,8.2,0]],colors.activations);
 wire(group,[[-4,7.3,0],[-4,5.5,0],[-2,5.5,0],[-2,4.2,0]],colors.activations);
 wire(group,[[0,7.3,0],[0,5.5,0],[-2,5.5,0]],colors.cache);
 wire(group,[[-2,3.3,0],[-2,2,0],[2,2,0],[2,1.2,0]],colors.activations);
 wire(group,[[4,7.3,0],[6,7.3,0],[6,1.2,0],[2,1.2,0]],colors.cache);
 wire(group,[[2,.3,0],[2,-.8,0],[0,-.8,0],[0,-2,0],[0,-4.5,0],[-1.5,-4.5,0],[-1.5,-5.4,0],[-1.5,-8,0],[0,-8,0],[0,-9,0],[0,-11,0]],colors.activations);
 // Residual path stays outside the GEMM stack.
 wire(group,[[-5.6,11.2,-2],[-5.6,-3.5,-2],[0,-3.5,-2]],0x667c98,.65);wire(group,[[-5.6,-3.5,-2],[-5.6,-10.5,-2],[0,-10.5,-2]],0x667c98,.65);label('Residual',[-6.5,-6.5,-2]);
 for(let h=0;h<3;h++)box(group,8,5.5-h*.48,0,2.3,.18,3.5,colors.cache,{grid:true,opacity:.8});label(`KV cache · ${m.ctx} positions`,[8,7,0]);label('16 Q heads / 8 KV heads',[7,3.2,0]);
 wire(group,[[6,7.3,0],[8,7.3,0],[8,5.5,0]],colors.cache,.6);
 box(group,0,-11,0,6,.16,tensorDepth,colors.activations,{grid:true});label('Block output',[0,-12,0],{major:true});
 const [x,y,z]=nodes[state.op];pulsePath=[[x-5,y+1,2],[x,y+1,2],[x,y,0],[x,y-.7,0],[x+5,y-.7,0]];
 addParticles();
}
function crossbarScene(){
 const m=selected();
 box(world,0,-.48,0,17,.55,14,colors.board);box(world,0,-.82,0,17.6,.12,14.6,0x415969);
 const positions=[[-4,-3.2],[3.5,-3.2],[-4,3.2],[3.5,3.2]],size=4.5,pitch=size/32;
 positions.forEach(([x,z],lane)=>{
  box(world,x,.02,z,size+.5,.3,size+.5,0x273b53,{edges:true});
  const geo=new THREE.BoxGeometry(pitch*.58,.065,pitch*.58),mat=material(colors.weights),cells=new THREE.InstancedMesh(geo,mat,1024);
  for(let r=0;r<32;r++)for(let c=0;c<32;c++){dummy.position.set(x-size/2+(c+.5)*pitch,.245,z-size/2+(r+.5)*pitch);dummy.updateMatrix();cells.setMatrixAt(r*32+c,dummy.matrix);}
  cells.userData.action=()=>{};world.add(cells);cells.instanceMatrix.needsUpdate=true;
  // Orthogonal electrodes intersect at the 1,024 displayed cells.
  const lines=[];for(let i=0;i<32;i++){let p=(i+.5)*pitch-size/2;lines.push(x-size/2,.29,z+p,x+size/2,.29,z+p,x+p,.23,z-size/2,x+p,.23,z+size/2);}
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(lines,3));world.add(new THREE.LineSegments(g,new THREE.LineBasicMaterial({color:0x6c8798,transparent:true,opacity:.5})));
  box(world,x-size/2-.35,.15,z,.34,.2,size,0x3d867b);box(world,x,.15,z+size/2+.35,size,.2,.32,0x957143);
  label(`Lane ${lane} · 32 × 32`,[x,.4,z-size/2-.6],{major:true});
 });
 label('FOUR MACRO LANES · ONE ENDPOINT',[0,1.5,-8],{major:true});
 box(world,-11,0,0,2,.6,10,0x377f71,{grid:true});label('Input driver',[-11,1.1,-6],{major:true});
 box(world,11,0,-3,3,.6,4,0x8c6b3d);label('SAR/TDC',[11,1.1,-5.7],{major:true});
 box(world,11,0,3,3,.6,4,0x956d39);label('Local accumulator',[11,1.1,6],{major:true});
 wire(world,[[-10,.5,0],[-8.5,.5,0],[-8.5,.5,-3.2],[-6.7,.5,-3.2]],colors.activations,.9);wire(world,[[-8.5,.5,0],[-8.5,.5,3.2],[-6.7,.5,3.2]],colors.activations,.9);
 wire(world,[[6,.4,-3.2],[9.5,.4,-3.2]],colors.partials,.9);wire(world,[[6,.4,3.2],[8,.4,3.2],[8,.4,-3.2]],colors.partials,.9);wire(world,[[11,.4,-1],[11,.4,1]],colors.partials,.9);
 rowSweep=new THREE.Group();world.add(rowSweep);
 for(const [x,z] of positions)for(let row=0;row<32;row++)box(rowSweep,x,.38,z-size/2+(row+.5)*pitch,size,.04,.035,colors.activations,{opacity:.8,edges:false});
 label(`Logical tile: C ${m.c0}–${m.c1-1} · K ${m.k0}–${m.k1-1}`,[0,-1,8.4],{major:true});
 label('64 × 32 logical tile → 2 row shards; bank-to-lane placement illustrative',[0,-2.2,9.5]);
 pulsePath=[[-11,.8,0],[-8,.8,0],[-8,.8,-3.2],[-4,.8,-3.2],[3.5,.8,-3.2],[11,.8,-3],[11,.8,3],[14,.8,3]];addParticles();
}
function addParticles(){const n=24,g=new THREE.SphereGeometry(.07,6,6),mat=new THREE.MeshBasicMaterial({color:colors.activations});particleSystem=new THREE.InstancedMesh(g,mat,n);particleSystem.visible=false;world.add(particleSystem);}
function buildScene(reset=true){
 document.body.classList.toggle('mapping-view',state.view==='mapping');mappingPanel.hidden=state.view!=='mapping';
 if(state.view!=='mapping')mappingView.pause();
 if(!scene)return;clearScene();
 if(state.view==='mapping'){mappingView.render(selected(),state);updateSceneHead();return;}
 ({model:modelScene,crossbar:crossbarScene})[state.view]();if(reset)cameraPreset(frameTime>0);updateSceneHead();
}
function updateSceneHead(){
 const titles={model:['01 / TRANSFORMER',`Block ${state.block+1}, expanded.`,'Select a tensor. Follow its mapping into a PE.'],crossbar:['02 / INSIDE A PE','From input voltage to partial sum.','Four 32 × 32 macro lanes, input drivers, conversion and accumulation.']};
 titles.mapping=['00 / MATRIX MULTIPLICATION → PE ASSIGNMENTS','Spatial work. Temporal reduction.','Which weight slice is used, where the output goes, and what changes at the next C step.'];
 const t=titles[state.view];$('scene-kicker').textContent=t[0];$('scene-title').textContent=t[1];$('scene-subtitle').textContent=t[2];
}
function chooseOp(op){state.op=op;resetSelection();update();}
function chooseBlock(block){state.block=block;if(state.scope==='run')state.scope='capacity';resetSelection();update();}
function setView(view){state.view=view;update(false);buildScene();}
function resetSelection(){state.row=state.col=state.token=0;state.stage=-1;state.playing=false;state.time=0;}
function update(rebuild=true){
 const m=selected();state.row=Math.min(state.row,m.rows-1);state.col=Math.min(state.col,m.cols-1);state.token=Math.min(state.token,m.tokens-1);
 document.querySelectorAll('[data-phase]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.phase===state.phase));document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.view===state.view));document.querySelectorAll('[data-op]').forEach(b=>b.setAttribute('aria-pressed',+b.dataset.op===state.op));
 $('block').value=state.block;$('block').disabled=state.scope==='run';$('scope').value=state.scope;
 $('op-title').textContent=roles[state.op];$('layer-name').textContent=m.d.name;
 $('input-shape').textContent=`${fmt(m.y)}×${fmt(m.c)}`;$('weight-shape').textContent=`${fmt(m.c)}×${fmt(m.k)}`;$('output-shape').textContent=`${fmt(m.y)}×${fmt(m.k)}`;
 $('weight-label').textContent=state.op===3?'Kᵀ / cache':state.op===4?'V / cache':'Weights';$('input-label').textContent=state.op===4?'Attention':'Input';
 const descriptions=[
  'Project the hidden state into 16 query heads. The same weights are reused across every token in a prefill block.',
  'Produce eight key heads and update the KV cache. Two query heads share each KV head.',
  'Produce eight value heads. Values stay in the KV cache and are reused for subsequent decode queries.',
  'Multiply each query head by cached keys. Causal masking and softmax follow this GEMM; they are separate from the mapped array operation.',
  'Reduce over cached positions to form a 128-channel output for each attention head.',
  'Mix the concatenated attention-head outputs back into the 2,048-channel hidden state.',
  'Expand from 2,048 to 6,144 channels. The runtime reuses this mapping for both gate and up projections; SiLU gating follows.',
  'Reduce 6,144 input channels into the 2,048-channel residual stream. Input-tile contributions accumulate before the result leaves the endpoint.'
 ];$('op-description').textContent=descriptions[state.op];
 $('tile-grid').textContent=`${m.rows} × ${m.cols}`;$('token-blocks').textContent=m.tokens;$('task-count').textContent=fmt(m.rows*m.cols*m.tokens);$('tasks-label').textContent=state.scope==='run'?'Shape-derived tasks':'Compiled tile tasks';
 for(const [id,value,max] of [['row',state.row,m.rows],['col',state.col,m.cols],['token',state.token,m.tokens]]){$(id).max=max-1;$(id).value=value;$(id).disabled=max===1;$(id+'-value').textContent=`${value+1} / ${max}`;}
 $('selected-range').textContent=`C ${m.c0}–${m.c1-1} · K ${m.k0}–${m.k1-1}\nTokens ${m.t0}–${m.t1-1}`;
 $('directives').textContent=['K','C','Y'].map(axis=>{const a=axis.toLowerCase();return `${m.d[a+'_map']}(${m.d[a+'_tile']},${m.d[a+'_stride']}) ${axis};`;}).join('\n');
 $('evidence').textContent=`${data.sources[state.phase].file}. ${m.source.windows.length?m.source.windows.length+' streamed weight window(s).':'Dynamic attention operands; no static weight window.'} Compiler boundaries use 64 × 32 tiles, independently of the C-map size/stride metadata.`;
 $('decode-note').textContent=state.phase==='decode'&&[3,4].includes(state.op)?'Decode attention mapping has Y=513. The actual decode runtime uses one query. Choose Saved run to see those runtime shapes.':'All 448 mapping layer plans match the saved compiler manifest. The 28-block stack represents mapping coverage, not a full-model hardware run.';
 updateStage();mappingView.render(selected(),state);if(rebuild)buildScene(false);
}
function updateStage(){
 document.querySelectorAll('[data-stage]').forEach(b=>{b.classList.toggle('current',+b.dataset.stage===state.stage);b.classList.toggle('done',+b.dataset.stage<state.stage);});
 $('play').textContent=state.playing?'Ⅱ':'▶';$('play').setAttribute('aria-label',state.playing?'Pause execution':'Play execution');
 const m=selected(),dynamicOp=[3,4].includes(state.op);
 const explanations=[dynamicOp?'Read keys / values from the KV cache.':'DMA brings a streamed weight window into die SRAM.',dynamicOp?'Present dynamic attention operands to the array.':'Program the weight window; reuse it across tile tasks.','Drive active rows; column currents form analog partials.','Convert row-group partials to digital values.',`Accumulate C tile ${state.row+1} of ${m.rows} for K ${m.k0}–${m.k1-1}.`,state.row===m.rows-1?'This output tile is reduced and ready for the NoC.':'Retain the local partial sum; more C tiles are required.'];
 $('stage-name').textContent=state.stage<0?'Ready to execute':stages[state.stage];$('stage-detail').textContent=state.stage<0?'Select Play to follow one logical tile.':explanations[state.stage];
}
function pathPoint(f){const q=Math.min(.99999,Math.max(0,f))*(pulsePath.length-1),i=Math.floor(q);return new THREE.Vector3(...pulsePath[i]).lerp(new THREE.Vector3(...pulsePath[i+1]),q-i);}
function animate(now){
 if(disposed)return;requestAnimationFrame(animate);const dt=Math.min(.1,(now-frameTime)/1000||0);frameTime=now;
 if(state.view==='mapping')return;
 if(transition){const t=Math.min(1,(now-transition.start)/900),e=t*t*(3-2*t);camera.position.lerpVectors(transition.from,transition.to,e);controls.target.lerpVectors(transition.target,transition.look,e);if(t===1)transition=null;}
 controls.update();
 if(state.playing){state.time+=dt;if(state.time>=1.6){state.time=0;if(state.stage<5)state.stage++;else{state.playing=false;}updateStage();}}
 const running=state.stage>=0,progress=state.time/1.6;
 if(particleSystem){particleSystem.visible=running;if(running){particleSystem.material.color.setHex(stageColors[state.stage]);for(let i=0;i<24;i++){const p=pathPoint(((state.stage+progress)/6+i/24*.18)%1);dummy.position.copy(p);dummy.scale.setScalar(state.playing?1:.75);dummy.updateMatrix();particleSystem.setMatrixAt(i,dummy.matrix);}particleSystem.instanceMatrix.needsUpdate=true;}}
 if(rowSweep){rowSweep.visible=running&&state.stage===2;for(const row of rowSweep.children)row.material.opacity=.6+.25*Math.sin(progress*Math.PI);}
 if(highlighted)highlighted.material.emissive.setHex(running?stageColors[state.stage]:0x000000),highlighted.material.emissiveIntensity=running?.24:0;
 const bounds=$('canvas-host').getBoundingClientRect();
 for(const l of labels){const v=l.position.clone().project(camera);const x=(v.x*.5+.5)*bounds.width,y=(-v.y*.5+.5)*bounds.height;l.el.style.left=x+'px';l.el.style.top=y+'px';l.el.hidden=v.z>1||v.z< -1||x<10||x>bounds.width-10||y<115||y>bounds.height-100;}
 renderer.render(scene,camera);
}

for(let b=0;b<28;b++){const option=document.createElement('option');option.value=b;option.textContent=`Block ${String(b+1).padStart(2,'0')} of 28`;$('block').append(option);}
roles.forEach((role,i)=>{const b=document.createElement('button');b.dataset.op=i;b.innerHTML=`<span>${String(i+1).padStart(2,'0')}</span>${role}`;b.onclick=()=>chooseOp(i);$('operations').append(b);});
stages.forEach((name,i)=>{const b=document.createElement('button');b.textContent=`${i+1} ${name}`;b.dataset.stage=i;b.onclick=()=>{state.stage=i;state.playing=false;state.time=.6;updateStage();};$('stages').append(b);});
document.querySelectorAll('[data-phase]').forEach(b=>b.onclick=()=>{state.phase=b.dataset.phase;resetSelection();update();});
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>setView(b.dataset.view));
$('block').onchange=e=>chooseBlock(+e.target.value);
$('scope').onchange=e=>{state.scope=e.target.value;if(state.scope==='run')state.block=0;resetSelection();update();};
for(const id of ['row','col','token'])$(id).oninput=e=>{state[id]=+e.target.value;state.stage=-1;state.playing=false;update();};
$('inspect-pe').onclick=()=>setView('crossbar');$('reset-camera').onclick=()=>cameraPreset();
$('play').onclick=()=>{if(state.playing)state.playing=false;else{if(state.stage<0||state.stage===5){state.stage=0;state.time=0;}state.playing=true;}updateStage();};
$('step').onclick=()=>{state.playing=false;state.time=.6;state.stage=(state.stage+1)%6;updateStage();};
try{init();update(false);document.documentElement.dataset.ready='true';}catch(error){$('scene-error').hidden=false;console.error(error);}
window.addEventListener('pagehide',()=>{disposed=true;renderer?.dispose();controls?.dispose();});
