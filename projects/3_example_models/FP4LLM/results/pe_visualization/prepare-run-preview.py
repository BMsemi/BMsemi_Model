"""Retrieve bounded BF16 weight previews and the exact saved input tokens."""
from pathlib import Path
import json, struct, urllib.request, concurrent.futures, sys

out=Path(__file__).resolve().parent
repo=out.parents[1]
revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
base=f'https://huggingface.co/Qwen/Qwen3-1.7B/resolve/{revision}/'
def request_json(url):
    with urllib.request.urlopen(url,timeout=40) as r:return json.load(r)
def byte_range(filename,start,end):
    req=urllib.request.Request(base+filename+f'?preview_range={start}-{end}',headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(req,timeout=50) as r:
        if r.status!=206:raise ValueError('Server did not honor the byte range; full download cancelled.')
        content_range=r.headers.get('Content-Range','')
        if not content_range.startswith(f'bytes {start}-{end}/'):raise ValueError('Unexpected response range: '+content_range)
        result=r.read(end-start+2)
        if len(result)!=end-start+1:raise ValueError('Incorrect range length')
        return result

metadata=json.loads((repo/'results/x1_prompt_run/software_crossbar/metadata.json').read_text())
tokenizer=json.loads((repo/'tokenizer.json').read_text(encoding='utf8'))
vocab={v:k for k,v in tokenizer['model']['vocab'].items()}
tokens=[{'position':i,'id':n,'text':vocab[n].replace('\u0120',' ').replace('\u010a','\n')} for i,n in enumerate(metadata['prefill_input_ids'][0])]
assert len(tokens)==11
token_data={'tokens':tokens,'prompt':''.join(t['text'] for t in tokens),'source':'results/x1_prompt_run/software_crossbar/metadata.json','tensors':{}}
(out/'run-preview.js').write_text('export default '+json.dumps(token_data,separators=(',',':'))+';\n',encoding='utf8')
if '--tokens-only' in sys.argv:
    print('Saved all 11 exact input tokens.');sys.exit(0)
index=request_json(base+'model.safetensors.index.json')
names=[f'model.layers.0.{s}.weight' for s in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.up_proj','mlp.gate_proj','mlp.down_proj']]
headers={}
for filename in {index['weight_map'][n] for n in names}:
    first=byte_range(filename,0,7);length=struct.unpack('<Q',first)[0]
    if length>10000000:raise ValueError('Header too large')
    headers[filename]=(json.loads(byte_range(filename,8,7+length)),8+length)
    print('Read checkpoint header:',filename,flush=True)
def load(name):
    filename=index['weight_map'][name];header,data_start=headers[filename];info=header[name]
    assert info['dtype']=='BF16' and len(info['shape'])==2
    rows=64;cols=info['shape'][1];start=data_start+info['data_offsets'][0]
    raw=byte_range(filename,start,start+rows*cols*2-1)
    # Store the original BF16 words. JS decodes only the on-screen values.
    import base64
    print('Retrieved',name,rows,'x',cols,flush=True)
    return name,{'shape':info['shape'],'rows':rows,'dtype':'BF16','words':base64.b64encode(raw).decode()}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:tensors=dict(pool.map(load,names))
data={'tokens':tokens,'prompt':''.join(t['text'] for t in tokens),'source':base.replace('/resolve/','/tree/'),'revision':revision,'note':'Published Qwen checkpoint preview, not verified as the checkpoint or quantized programmed weights of the saved software run. First 64 output rows of block 1 projections only.','tensors':tensors}
(out/'run-preview.js').write_text('export default '+json.dumps(data,separators=(',',':'))+';\n',encoding='utf8')
print('Saved real token and BF16 weight preview.',flush=True)
