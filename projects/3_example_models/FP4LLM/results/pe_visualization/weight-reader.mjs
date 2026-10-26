export function decodeValue(view,offset,dtype){
 if(dtype==='F32')return view.getFloat32(offset,true);
 if(dtype==='BF16'){const b=new ArrayBuffer(4),v=new DataView(b);v.setUint32(0,view.getUint16(offset,true)*65536,true);return v.getFloat32(0,true);}
 if(dtype==='F16'){const x=view.getUint16(offset,true),sign=x&32768?-1:1,e=(x>>10)&31,f=x&1023;return e===0?sign*2**-14*f/1024:e===31?(f?NaN:sign*Infinity):sign*2**(e-15)*(1+f/1024);}
 throw Error(`Unsupported dtype: ${dtype}`);
}
export async function openWeightFiles(files){
 const tensors=new Map();
 for(const file of files){
  if(file.size<8)throw Error('Invalid safetensors file.');
  const size=Number(new DataView(await file.slice(0,8).arrayBuffer()).getBigUint64(0,true));
  if(!Number.isSafeInteger(size)||size<2||size>100000000||size+8>file.size)throw Error('Invalid safetensors header size.');
  const header=JSON.parse(new TextDecoder().decode(await file.slice(8,8+size).arrayBuffer()));
  for(const [name,info] of Object.entries(header))if(name!=='__metadata__')tensors.set(name,{file,info,start:8+size});
 }
 return {count:tensors.size,async read(name,ks,cs,expected){
  const tensor=tensors.get(name);if(!tensor)throw Error('This tensor is not in the selected file(s). Select its checkpoint shard.');
  const {file,info,start}=tensor,bytes={BF16:2,F16:2,F32:4}[info.dtype];
  if(!bytes)throw Error(`Numerical preview supports BF16, F16 and F32; this tensor is ${info.dtype}.`);
  if(info.shape.length!==2||info.shape[0]!==expected[0]||info.shape[1]!==expected[1])throw Error('Checkpoint tensor dimensions do not match the mapping.');
  const [begin,end]=info.data_offsets;if(!Number.isSafeInteger(begin)||!Number.isSafeInteger(end)||begin<0||end-begin!==expected[0]*expected[1]*bytes||start+end>file.size)throw Error('Invalid tensor offsets.');
  const values=await Promise.all(ks.map(async k=>{const off=start+begin+(k*info.shape[1]+cs[0])*bytes;const view=new DataView(await file.slice(off,off+cs.length*bytes).arrayBuffer());return cs.map((c,i)=>decodeValue(view,i*bytes,info.dtype));}));
  return {dtype:info.dtype,values:cs.map((c,ci)=>ks.map((k,ki)=>values[ki][ci]))};
 }};
}
