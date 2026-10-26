"""Run on PYNQ: numerical bring-up of all six *physical* engines.

NumPy below is an independent test oracle, never a model-runtime fallback.
The signed test operands cross 32-channel/tile boundaries and use all four LPCs.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ps_int8_quantization import prepare_mvm, round_bf16
from src.qwen_fpga6 import (Fpga6Backend, RMS_NORM, ROPE, CAUSAL_SOFTMAX,
                           SILU_MUL, RESIDUAL_ADD)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bitstream', type=Path, required=True)
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--report', type=Path, default=Path('board_smoke.json'))
    args = parser.parse_args()
    board = Fpga6Backend.open(args.bitstream, args.driver)
    rng = np.random.default_rng(607)
    x = round_bf16(rng.normal(size=(11,65)).astype(np.float32))
    w = round_bf16(rng.normal(size=(145,65)).astype(np.float32))
    prepared = prepare_mvm(x,w)
    expected = prepared.reconstruct(prepared.activations.astype(np.int32) @ prepared.weights.astype(np.int32))
    actual = board.matmul(x,w,name='signed_tail_tiles')
    np.testing.assert_array_equal(actual,expected)
    a = round_bf16(rng.normal(size=(11,8)).astype(np.float32))
    norm = round_bf16(a/np.sqrt(np.mean(a*a,axis=-1,keepdims=True)+1e-6))
    np.testing.assert_allclose(board.vector(RMS_NORM,a,np.ones(8,np.float32)),norm,rtol=.008,atol=1e-5)
    table = np.tile(np.array([0]*8+[1]*8,np.float32),(11,1))
    rotated = np.concatenate((-a[:,4:],a[:,:4]),axis=1)
    np.testing.assert_array_equal(board.vector(ROPE,a,table),rotated)
    scores = np.zeros((11,11),np.float32)
    softmax = round_bf16(np.tril(np.ones_like(scores))/np.arange(1,12,dtype=np.float32)[:,None])
    np.testing.assert_array_equal(board.vector(CAUSAL_SOFTMAX,scores,scale=.125),softmax)
    silu = round_bf16(round_bf16(a/(1+np.exp(-a)))*a)
    np.testing.assert_allclose(board.vector(SILU_MUL,a,a),silu,rtol=.008,atol=1e-5)
    np.testing.assert_array_equal(board.vector(RESIDUAL_ADD,a,-a),np.zeros_like(a))
    for engine in range(4):
        assert board.counts[f'engine_{engine}_mvm']>0
    for engine in (4,5):
        for op in range(1,7):
            assert board.counts[f'engine_{engine}_op_{op}']>0
    report = {'passed': True, 'physical_hardware': True, 'synthetic_operands': True,
              'real_qwen_checkpoint_test': False, 'engine_operations': dict(board.counts),
              'bitstream': str(args.bitstream.resolve())}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(f'All six physical engines passed bring-up. Report: {args.report}')


if __name__ == '__main__':
    main()
