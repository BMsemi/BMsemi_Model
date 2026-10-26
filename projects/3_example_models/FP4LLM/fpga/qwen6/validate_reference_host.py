"""Host-only validation of a real checkpoint bundle. This is NOT a board run.

LPC test doubles use NumPy; NLPC invokes the actual C++ kernel. Compare with
un-tiled INT8 and BF16-rounded NumPy block references. Requires a C++ compiler.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from test_qwen_fpga6 import (NativeNlpc, FakeLpc, KernelAndScheduleTests,
                            reference_gemm, reference_vector)
from src.qwen_fpga6 import Fpga6Backend, run_block, WEIGHT_KEYS
from src.ps_int8_quantization import round_bf16


def error(actual, expected):
    delta = actual.astype(np.float64)-expected.astype(np.float64)
    norm = float(np.linalg.norm(expected.astype(np.float64)))
    return {'relative_l2': float(np.linalg.norm(delta)/norm) if norm else None,
            'max_abs': float(np.max(np.abs(delta)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--outputs',type=Path)
    args = parser.parse_args()
    with np.load(args.bundle,allow_pickle=False) as z:
        metadata = json.loads(str(z['metadata']))
        data = {key:z[key] for key in ('input','rotary')+WEIGHT_KEYS}
    NativeNlpc.build()
    events = []
    backend = Fpga6Backend([FakeLpc(i,events) for i in range(4)],[NativeNlpc(),NativeNlpc()])
    try:
        start = time.monotonic()
        actual = run_block(backend,data,metadata)
        elapsed = time.monotonic()-start
        class Int8Reference:
            matmul = staticmethod(lambda x,w,**kw: reference_gemm(x,w))
            vector = staticmethod(reference_vector)
        class Bf16Reference:
            matmul = staticmethod(lambda x,w,**kw: round_bf16(x@w.T))
            vector = staticmethod(reference_vector)
        quantized = run_block(Int8Reference(),data,metadata)
        bf16 = run_block(Bf16Reference(),data,metadata)
        report = {'physical_hardware':False,
                  'validation_kind':'LPC NumPy test doubles + actual NLPC C++ kernel',
                  'checkpoint':metadata['checkpoint'],
                  'reference_checkpoint':metadata.get('reference_checkpoint',False),
                  'tokens':len(data['input']), 'token_ids':metadata['token_ids'],
                  'shape':list(actual['output'].shape),
                  'all_finite':bool(np.isfinite(actual['output']).all()),
                  'engine_operations_emulated':dict(backend.counts),
                  'host_wall_seconds':elapsed,
                  'comparison_note':'NumPy references; not bitwise PyTorch or physical FPGA validation',
                  'error_vs_int8_reference':error(actual['output'],quantized['output']),
                  'error_vs_bf16_reference':error(actual['output'],bf16['output'])}
        if not report['all_finite']:
            raise RuntimeError('Host validation produced nonfinite output')
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
        if args.outputs:
            args.outputs.parent.mkdir(parents=True,exist_ok=True)
            np.savez(args.outputs,**actual,int8_numpy_reference=quantized['output'],
                     bf16_numpy_reference=bf16['output'])
        print(json.dumps(report,indent=2))
    finally:
        KernelAndScheduleTests.tearDownClass()


if __name__ == '__main__':
    main()
