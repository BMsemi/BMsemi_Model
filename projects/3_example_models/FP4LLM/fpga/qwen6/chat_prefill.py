"""Executed on the board via board_dma_heap.py for a fresh chat prompt."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

root = Path('/home/xilinx/qwen6_20261005')
sys.path.insert(0, str(root))
from src.qwen_fpga6 import Fpga6Backend, WEIGHT_KEYS, run_block, validate_bundle

directory = Path(sys.argv[1])
with np.load(root/'block1_bundle.npz', allow_pickle=False) as weights, np.load(directory/'input.npz', allow_pickle=False) as patch:
    request = json.loads(str(patch['metadata']))
    metadata = json.loads(str(weights['metadata']))
    if metadata['config'] != request['config']:
        raise ValueError('Board checkpoint config differs from chat checkpoint')
    data = {key: weights[key] for key in WEIGHT_KEYS}
    hashes = {key: hashlib.sha256(data[key].tobytes()).hexdigest() for key in WEIGHT_KEYS}
    if hashes != request['hashes']:
        raise ValueError('Board block-0 weights differ from chat checkpoint; refusing execution')
    data.update(input=patch['input'], rotary=patch['rotary'])
    metadata.update(token_ids=request['token_ids'], position_ids=list(range(len(request['token_ids']))), prompt='chat token sequence')
validate_bundle(data, metadata)
backend = Fpga6Backend.open(root/'fpga/qwen6/overlay/qwen6.bit', root/'fpga/qwen6/axi_x2_mvm_driver.py')
started = time.monotonic()
result = run_block(backend, data, metadata)
report = {'physical_hardware': True, 'phase': 'prefill', 'block_index': 0,
          'token_ids': request['token_ids'], 'weight_hashes': hashes,
          'engine_operations': dict(backend.counts), 'wall_seconds': time.monotonic()-started,
          'lpc_dtype': 'INT8', 'accumulator_dtype': 'INT32', 'transport': 'COM17 serial'}
np.savez_compressed(directory/'output.npz', report=np.array(json.dumps(report)), **result)
print(json.dumps(report), flush=True)
