"""Fetch just block 0 and selected embeddings from a pinned public Qwen revision.

The resulting bundle explicitly identifies a reference checkpoint; it is not
claimed to be the unavailable checkpoint used by the historical saved run.
No prompt text is uploaded. Only public model data is downloaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import urllib.request

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from export_fpga_bundle import export_from_reader
from src.ps_int8_quantization import round_bf16, unpack_bf16


class PublicQwenReader:
    def __init__(self, revision, proxy=None):
        if not re.fullmatch('[0-9a-f]{40}',revision):
            raise ValueError('Use an immutable 40-character checkpoint commit')
        self.source = f'https://huggingface.co/Qwen/Qwen3-1.7B/tree/{revision}'
        self.base = self.source.replace('/tree/','/resolve/')+'/'
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({'https':proxy,'http':proxy}) if proxy else urllib.request.ProxyHandler())
        self.index = self.json('model.safetensors.index.json')['weight_map']
        self.headers = {}
        self.hashes = {}

    def json(self, filename):
        with self.opener.open(self.base+filename,timeout=60) as response:
            return json.load(response)

    def byte_range(self, filename, start, end):
        # Query key avoids a proxy reusing a cached response for another range.
        request = urllib.request.Request(self.base+filename+f'?qwen6_range={start}-{end}',
                                         headers={'Range':f'bytes={start}-{end}'})
        with self.opener.open(request,timeout=90) as response:
            if response.status != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                raise ValueError('Server ignored the requested byte range; full-model download refused')
            data = response.read(end-start+2)
            if len(data) != end-start+1:
                raise ValueError('Truncated or oversized model byte range')
            return data

    def read(self, key, rows=None):
        filename = self.index[key]
        if filename not in self.headers:
            length = struct.unpack('<Q',self.byte_range(filename,0,7))[0]
            if not 2 <= length <= 100_000_000:
                raise ValueError('Invalid safetensors header length')
            self.headers[filename] = (json.loads(self.byte_range(filename,8,7+length)),8+length)
        header, base = self.headers[filename]
        spec = header[key]
        if spec['dtype'] != 'BF16':
            raise ValueError(f'{key} must be BF16 for this reference export')
        lo, hi = spec['data_offsets']
        shape = tuple(spec['shape'])
        if hi-lo != int(np.prod(shape))*2:
            raise ValueError(f'Invalid tensor extent: {key}')
        if rows is None:
            data = self.byte_range(filename,base+lo,base+hi-1)
        else:
            if len(shape)!=2 or min(rows)<0 or max(rows)>=shape[0]:
                raise ValueError('Embedding row outside checkpoint vocabulary')
            width = shape[1]*2
            data = b''.join(self.byte_range(filename,base+lo+row*width,base+lo+(row+1)*width-1) for row in rows)
            shape = (len(rows),shape[1])
        self.hashes[key] = hashlib.sha256(data).hexdigest()
        print(f'Read real BF16 {key} {shape}',flush=True)
        return unpack_bf16(np.frombuffer(data,dtype='<u2').reshape(shape))


def main():
    from tokenizers import Tokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision',required=True)
    parser.add_argument('--prompt',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--proxy',help='Optional per-command download proxy')
    parser.add_argument('--max-tokens',type=int,default=128)
    args = parser.parse_args()
    reader = PublicQwenReader(args.revision,args.proxy)
    config = reader.json('config.json')
    tokenizer = Tokenizer.from_str(json.dumps(reader.json('tokenizer.json')))
    export_from_reader(reader,config,tokenizer,args.prompt,args.out,reader.source,args.max_tokens,
                       reference_checkpoint=True)
    # Provenance includes a digest of every selected BF16 tensor byte range.
    args.out.with_suffix('.source.json').write_text(json.dumps({
        'checkpoint':reader.source,'reference_checkpoint':True,
        'sha256_bf16_ranges':reader.hashes,
        'note':'Embedding hash covers only selected token rows, in token order.'},indent=2)+'\n')


if __name__ == '__main__':
    main()
