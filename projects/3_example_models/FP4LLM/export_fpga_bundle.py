"""Export only real checkpoint tensors and token embeddings for FPGA prefill.

This offline preparation runs on the workstation. No transformer block is
evaluated here. Safetensors data is memory-mapped, so the full model need not
fit RAM. Requires numpy and Hugging Face tokenizers (for tokenizer.json).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import struct

import numpy as np
from src.ps_int8_quantization import round_bf16, unpack_bf16
from src.qwen_fpga6 import WEIGHT_KEYS, validate_bundle


class Checkpoint:
    def __init__(self, root):
        self.root = Path(root)
        self.tensors = {}
        for path in sorted(self.root.glob('*.safetensors')):
            size = path.stat().st_size
            with path.open('rb') as stream:
                raw = stream.read(8)
                if len(raw) != 8:
                    raise ValueError(f"Truncated safetensors file: {path}")
                length = struct.unpack('<Q', raw)[0]
                if not 2 <= length <= min(size-8, 100_000_000):
                    raise ValueError(f"Invalid safetensors header: {path}")
                header = json.loads(stream.read(length))
            for key, spec in header.items():
                if key == '__metadata__':
                    continue
                if key in self.tensors:
                    raise ValueError(f"Duplicate tensor {key}; use one checkpoint snapshot directory")
                self.tensors[key] = (path, 8+length, spec)
        if not self.tensors:
            raise FileNotFoundError(f"No safetensors checkpoint found in {self.root}")

    def read(self, key, rows=None):
        if key not in self.tensors:
            raise KeyError(f"Checkpoint is missing {key}")
        path, base, spec = self.tensors[key]
        dtype = {'BF16': '<u2', 'F16': '<f2', 'F32': '<f4'}.get(spec['dtype'])
        if dtype is None:
            raise ValueError(f"{key}: expected BF16/F16/F32, got {spec['dtype']}; packed FP4 is unsupported")
        lo, hi = spec['data_offsets']
        shape = tuple(spec['shape'])
        if (not shape or min(shape) <= 0 or lo < 0 or hi < lo or
            hi-lo != int(np.prod(shape))*np.dtype(dtype).itemsize or base+hi > path.stat().st_size):
            raise ValueError(f"Invalid tensor extent for {key}")
        mapped = np.memmap(path, mode='r', dtype=dtype, offset=base+lo, shape=shape)
        values = mapped if rows is None else mapped[rows]
        result = unpack_bf16(values) if spec['dtype'] == 'BF16' else np.asarray(values, dtype=np.float32)
        return round_bf16(result)


def export(checkpoint_dir, prompt, destination, max_tokens=128):
    from tokenizers import Tokenizer
    root = Path(checkpoint_dir)
    cfg = json.loads((root/'config.json').read_text())
    tokenizer = Tokenizer.from_file(str(root/'tokenizer.json'))
    return export_from_reader(Checkpoint(root), cfg, tokenizer, prompt, destination,
                              str(root.resolve()), max_tokens)


def export_from_reader(checkpoint, cfg, tokenizer, prompt, destination, source, max_tokens=128,
                       *, reference_checkpoint=False):
    encoding = tokenizer.encode(prompt, add_special_tokens=False)
    ids = encoding.ids
    if not 1 <= len(ids) <= min(128, max_tokens):
        raise ValueError(f"Prompt has {len(ids)} tokens; expected 1..{min(128, max_tokens)} (no silent truncation)")
    data = {key: checkpoint.read('model.layers.0.'+key) for key in WEIGHT_KEYS}
    data['input'] = checkpoint.read('model.embed_tokens.weight', ids)
    d = int(cfg['head_dim'])
    # Positional constants are prepared once, outside the transformer block.
    inv_frequency = 1.0 / (np.float32(cfg['rope_theta']) ** (np.arange(0, d, 2, dtype=np.float32)/d))
    phase = np.arange(len(ids), dtype=np.float32)[:, None] * inv_frequency[None, :]
    phase = np.concatenate((phase, phase), axis=1)
    data['rotary'] = round_bf16(np.concatenate((np.cos(phase), np.sin(phase)), axis=1))
    metadata = {'format': 'qwen6-bundle-v1', 'block_index': 0, 'prompt': prompt,
                'token_ids': ids, 'tokens': encoding.tokens, 'config': cfg,
                'checkpoint': source, 'reference_checkpoint': reference_checkpoint,
                'source_tensors': 'model.layers.0.*',
                'position_ids': list(range(len(ids)))}
    validate_bundle(data, metadata)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, metadata=np.array(json.dumps(metadata)), **data)
    print(f"Exported {len(ids)} real tokens and block-1 weights to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=128)
    args = parser.parse_args()
    export(args.checkpoint, args.prompt, args.out, args.max_tokens)


if __name__ == '__main__':
    main()
