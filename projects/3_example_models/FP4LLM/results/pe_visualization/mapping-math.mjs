export function tensorFor(block,op,ffn='up'){
 const prefix=`model.layers.${block}`;
 const paths=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj',null,null,'self_attn.o_proj',`mlp.${ffn}_proj`,'mlp.down_proj'];
 return paths[op]?`${prefix}.${paths[op]}.weight`:op===3?'KV_cache.keysᵀ':'KV_cache.values';
}
export function assignment(m,mode,step,kBase){
 const compiled=mode==='compiled',cSize=compiled?64:m.d.c_tile,cStride=compiled?64:m.d.c_stride,kSize=compiled?32:m.d.k_tile,kStride=compiled?32:m.d.k_stride;
 const steps=Math.ceil(m.c/cStride),t=Math.max(0,Math.min(step,steps-1));
 const c0=t*cStride,c1=Math.min(m.c,c0+cSize),ys=m.d.y_tile, yStride=m.d.y_stride;
 const yCount=Math.ceil(m.y/yStride),kCount=Math.ceil(m.k/kStride);
 const base=Math.max(0,Math.min(kBase,Math.max(0,kCount-4)));
 const pes=[];
 for(let yi=0;yi<Math.min(yCount,2);yi++)for(let ki=base;ki<Math.min(base+4,kCount);ki++){
  const k0=ki*kStride,k1=Math.min(m.k,k0+kSize),y0=yi*yStride,y1=Math.min(m.y,y0+ys);
  pes.push({ki,yi,k0,k1,y0,y1,c0,c1});
 }
 return {compiled,cSize,cStride,kSize,kStride,steps,step:t,c0,c1,yCount,kCount,base,pes,overlap:!compiled&&cSize>cStride};
}
