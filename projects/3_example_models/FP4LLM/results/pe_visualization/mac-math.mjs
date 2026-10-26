import {decodeValue} from './weight-reader.mjs';
export function activationNames(op, tensor='') {
 const input=['input_layernorm_output','input_layernorm_output','input_layernorm_output',null,null,'attn_context_before_o_proj','post_attention_layernorm_output','mlp_gated_activation_before_down_proj'][op];
 const output=['q_proj_output','k_proj_output','v_proj_output',null,null,'o_proj_output',tensor.includes('gate_proj')?'mlp_gate_proj_output':'mlp_up_proj_output','mlp_down_proj_output'][op];
 return {input:input&&`layer_00_${input}`,output:output&&`layer_00_${output}`};
}
export function savedValue(data,phase,name,token,channel,cache=new Map()) {
 const key=`${phase}/${name}`,item=data.tensors[key];
 if(!item||token<0||channel<0||token>=item.shape[1]||channel>=item.shape[2])return null;
 if(!cache.has(key)){const raw=Uint8Array.from(atob(item.words),c=>c.charCodeAt(0));cache.set(key,new DataView(raw.buffer));}
 return decodeValue(cache.get(key),(token*item.shape[2]+channel)*2,'BF16');
}
export function prefixMAC(input,weights,end) {
 let sum=0;
 for(let i=0;i<=end;i++){if(!Number.isFinite(input[i])||!Number.isFinite(weights[i]))return null;sum+=input[i]*weights[i];}
 return sum;
}
export function crossbarSlice(a,index=0){
 const first=Math.floor(a.c0/32),count=Math.ceil(a.c1/32)-first;
 const selected=Math.max(0,Math.min(index,count-1));
 return {index:selected,count,c0:Math.max(a.c0,(first+selected)*32),c1:Math.min(a.c1,(first+selected+1)*32)};
}
export function vectorMAC(input,weights){
 return prefixMAC(input,weights,input.length-1);
}
