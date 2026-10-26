"""Local validation only. Fake LPCs model INT8 hardware; NLPC uses actual C++.

These tests never identify emulation as a physical FPGA run.
"""
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.ps_int8_quantization import prepare_mvm, round_bf16
from src.qwen_fpga6 import (Fpga6Backend, NlpcDriver, run_block, validate_bundle, WEIGHT_KEYS,
                           PROBE, ADD_I32, RMS_NORM, ROPE, CAUSAL_SOFTMAX, SILU_MUL, RESIDUAL_ADD)

ROOT = Path(__file__).resolve().parent


class EntrypointTests(unittest.TestCase):
    def test_precision_defaults_and_software_forwarding(self):
        import argparse
        import types
        import main
        import src
        from unittest.mock import Mock
        parser = main.build_parser()
        for flags, expected in (([], (7, 7)),
                                (['--execution-target', 'fpga6'], (8, 8)),
                                (['--weight-bits', '6', '--execution-target', 'fpga6', '--act-bits', '5'], (6, 5))):
            args = parser.parse_args(['--prompt', 'test', *flags])
            self.assertEqual((args.weight_bits, args.act_bits), expected)
        # The existing software tests call the runner directly after parsing.
        # Check its numeric forwarding without requiring a model or PyTorch.
        runtime_parser = argparse.ArgumentParser()
        runtime_parser.add_argument('--weight-bits', type=int)
        runtime_parser.add_argument('--act-bits', type=int)
        runtime = types.SimpleNamespace(
            parse_args=lambda argv: runtime_parser.parse_known_args(argv)[0], run=Mock())
        args = parser.parse_args(['--prompt', 'test'])
        with patch.object(src, 'qwen_crossbar_runtime', runtime, create=True), patch.object(Path, 'exists', return_value=True):
            main.run_prompt_crossbar(args, {'prefill': Path('prefill.m'), 'decode': Path('decode.m')}, ROOT)
        forwarded = runtime.run.call_args.args[0]
        self.assertEqual((forwarded.weight_bits, forwarded.act_bits), (7, 7))


class NativeNlpc:
    library = None
    directory = None

    @classmethod
    def build(cls):
        if cls.library is not None:
            return
        hls_compiler = Path('C:/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin/g++.exe')
        compiler = str(hls_compiler) if hls_compiler.is_file() else shutil.which('g++')
        if compiler is None:
            raise unittest.SkipTest('C++ compiler required for actual NLPC kernel tests')
        cls.directory = tempfile.TemporaryDirectory(prefix='qwen_nlpc_test_')
        folder = Path(cls.directory.name)
        wrapper = folder/'wrapper.cpp'
        wrapper.write_text('#include "qwen_nlpc.h"\nextern "C" void call_nlpc(const uint32_t*a,const uint32_t*b,uint32_t*out,unsigned r,unsigned c,unsigned op,unsigned off,float eps,float scale,unsigned*s){qwen_nlpc(a,b,out,r,c,op,off,eps,scale,*s);}\n')
        lib = folder/('kernel.dll' if os.name == 'nt' else 'kernel.so')
        source = ROOT/'fpga/qwen6/hls'
        command = [compiler, '-std=c++11', '-shared', '-O2', '-fno-fast-math', '-ffp-contract=off',
                   '-static-libgcc', '-static-libstdc++', '-I'+str(source), str(wrapper),
                   str(source/'qwen_nlpc.cpp'), '-o', str(lib)]
        if os.name != 'nt':
            command.insert(1, '-fPIC')
        environment = dict(os.environ)
        environment['PATH'] = str(Path(compiler).parent) + os.pathsep + environment.get('PATH', '')
        compiled = subprocess.run(command, capture_output=True, text=True, env=environment)
        if compiled.returncode:
            raise RuntimeError(f"C++ kernel compilation failed: {compiled.stdout}\n{compiled.stderr}")
        cls.library = ctypes.CDLL(str(lib))
        cls.library.call_nlpc.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_uint]*4 + [ctypes.c_float]*2 + [ctypes.POINTER(ctypes.c_uint)]

    def prepare(self, op, a, b=None, **kw):
        return dict(op=op, a=np.ascontiguousarray(a), b=b, kw=kw, started=False)

    def start(self, job):
        job['started'] = True

    def wait(self, job):
        assert job['started']
        a = job['a']
        dtype = np.int32 if job['op'] == ADD_I32 else np.float32
        a = np.ascontiguousarray(a, dtype=dtype)
        b = np.ascontiguousarray(job['b'] if job['b'] is not None else [0], dtype=dtype)
        out = np.zeros_like(a)
        kw = job['kw']
        status = ctypes.c_uint(999)
        self.library.call_nlpc(a.ctypes.data, b.ctypes.data, out.ctypes.data, *a.shape,
                              job['op'], kw.get('row_offset', 0), kw.get('epsilon', 1e-6),
                              kw.get('scale', 1), ctypes.byref(status))
        if status.value:
            raise RuntimeError(f'Native NLPC error {status.value}')
        return out

    def release(self, job):
        pass


class FakeLpc:
    def __init__(self, engine, events):
        self.engine, self.events = engine, events

    def prepare_mvm_batch(self, x, w):
        assert x.dtype == np.int8 and w.dtype == np.int8
        assert max(*x.shape, *w.shape) <= 32
        return [x.copy(), w.copy(), False]

    def start_job(self, job):
        job[2] = True
        self.events.append(('start', self.engine))

    def wait_job(self, job):
        assert job[2]
        self.events.append(('wait', self.engine))
        return job[0].astype(np.int32) @ job[1].astype(np.int32)

    def release_job(self, job, abort=False):
        pass


def reference_gemm(x, w):
    prepared = prepare_mvm(x, w)
    return prepared.reconstruct(prepared.activations.astype(np.int32) @ prepared.weights.astype(np.int32))


def reference_vector(op, a, b=None, **kw):
    if op == RMS_NORM:
        return round_bf16(round_bf16(a/np.sqrt(np.mean(a*a, axis=-1, keepdims=True)+kw.get('epsilon', 1e-6)))*b)
    if op == ROPE:
        d = a.shape[1]
        rotated = np.concatenate((-a[:, d//2:], a[:, :d//2]), axis=1)
        return round_bf16(round_bf16(a*b[:, :d])+round_bf16(rotated*b[:, d:]))
    if op == CAUSAL_SOFTMAX:
        scores = a * np.float32(kw.get('scale', 1))
        mask = np.arange(a.shape[1])[None, :] > np.arange(len(a))[:, None]+kw.get('row_offset', 0)
        scores = np.where(mask, -np.inf, scores)
        exp = np.exp(scores-np.max(scores, axis=-1, keepdims=True))
        return round_bf16(exp/np.sum(exp, axis=-1, keepdims=True))
    if op == SILU_MUL:
        return round_bf16(round_bf16(a/(1+np.exp(-a)))*b)
    if op == RESIDUAL_ADD:
        return round_bf16(a+b)
    raise ValueError(op)


def fixture(tokens=11):
    rng = np.random.default_rng(71)
    h, d, heads, kv, ff = 40, 8, 6, 2, 65
    shapes = [(h,), (h,), (heads*d,h), (kv*d,h), (kv*d,h), (h,heads*d),
              (d,), (d,), (ff,h), (ff,h), (h,ff)]
    data = {key: round_bf16(rng.normal(0, .12, shape).astype(np.float32)) for key, shape in zip(WEIGHT_KEYS, shapes)}
    for key in (WEIGHT_KEYS[0], WEIGHT_KEYS[1], WEIGHT_KEYS[6], WEIGHT_KEYS[7]):
        data[key] = np.ones_like(data[key])
    data['input'] = round_bf16(rng.normal(size=(tokens,h)).astype(np.float32))
    phase = np.arange(tokens, dtype=np.float32)[:,None]*np.arange(1,d//2+1,dtype=np.float32)[None,:]*.03
    phase = np.concatenate((phase,phase), axis=1)
    data['rotary'] = round_bf16(np.concatenate((np.cos(phase), np.sin(phase)), axis=1))
    metadata = {'format': 'qwen6-bundle-v1', 'block_index': 0,
                'token_ids': list(range(tokens)), 'position_ids': list(range(tokens)),
                'config': {'hidden_size': h, 'head_dim': d,
                'num_attention_heads': heads, 'num_key_value_heads': kv, 'intermediate_size': ff,
                'rms_norm_eps': 1e-6, 'hidden_act': 'silu', 'model_type': 'qwen3'}}
    return data, metadata


class KernelAndScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        NativeNlpc.build()

    @classmethod
    def tearDownClass(cls):
        if os.name == 'nt' and NativeNlpc.library is not None:
            import _ctypes
            _ctypes.FreeLibrary(NativeNlpc.library._handle)
        NativeNlpc.library = None
        if NativeNlpc.directory is not None:
            NativeNlpc.directory.cleanup()

    def backend(self):
        self.events = []
        return Fpga6Backend([FakeLpc(i, self.events) for i in range(4)], [NativeNlpc(), NativeNlpc()])

    def test_tiled_gemm_tail_tiles_four_engines_and_integer_partials(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(35,65)).astype(np.float32)
        w = rng.normal(size=(145,65)).astype(np.float32)
        backend = self.backend()
        np.testing.assert_array_equal(backend.matmul(x,w,name='tail_test'), reference_gemm(x,w))
        self.assertEqual(self.events[:4], [('start',i) for i in range(4)])
        for engine in range(4):
            self.assertGreater(backend.counts[f'engine_{engine}_mvm'],0)
        for engine in (4,5):
            self.assertGreater(backend.counts[f'engine_{engine}_op_1'],0)

    def test_nonlinear_kernels_nontrivial_values_and_row_splits(self):
        rng = np.random.default_rng(92)
        a = round_bf16(rng.normal(size=(11,8)).astype(np.float32))
        for op, b in ((RMS_NORM, np.ones(8,np.float32)),
                      (ROPE, round_bf16(rng.normal(size=(11,16)).astype(np.float32))),
                      (SILU_MUL,a[::-1].copy()), (RESIDUAL_ADD,a[::-1].copy())):
            with self.subTest(op=op):
                np.testing.assert_allclose(self.backend().vector(op,a,b), reference_vector(op,a,b), rtol=.008, atol=1e-5)
        scores = round_bf16(rng.normal(size=(11,11)).astype(np.float32))
        actual = self.backend().vector(CAUSAL_SOFTMAX,scores,scale=.125)
        np.testing.assert_allclose(actual, reference_vector(CAUSAL_SOFTMAX,scores,scale=.125), rtol=.008, atol=1e-5)
        np.testing.assert_array_equal(np.triu(actual,1),0)

    def test_full_block_11_tokens_gqa_and_causality(self):
        data, metadata = fixture()
        actual = run_block(self.backend(), data, metadata)
        # Independent un-tiled numerical backend checks the composed graph.
        class Reference:
            matmul = staticmethod(lambda x,w,**kw: reference_gemm(x,w))
            vector = staticmethod(reference_vector)
        expected = run_block(Reference(), data, metadata)
        np.testing.assert_allclose(actual['output'],expected['output'],rtol=.015,atol=.016)
        np.testing.assert_allclose(actual['key_cache'],expected['key_cache'],rtol=.015,atol=.008)
        # A future token must not influence any earlier output.
        changed = dict(data)
        changed['input'] = data['input'].copy()
        changed['input'][-1] = changed['input'][-1, ::-1]
        future = run_block(self.backend(), changed, metadata)
        np.testing.assert_array_equal(actual['output'][:-1],future['output'][:-1])

    def test_integer_overflow_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'3'):
            self.backend().vector(ADD_I32,np.array([[2147483647]],np.int32),np.array([[1]],np.int32))


class BoundaryTests(unittest.TestCase):
    def test_fault_recovery_resets_overlay_before_releasing_retained_dma(self):
        from src.qwen_fpga6 import _QUARANTINED_BUFFERS
        events = []
        class Overlay:
            def download(self): events.append('reset')
        class Buffer:
            def freebuffer(self): events.append('free')
        saved = list(_QUARANTINED_BUFFERS)
        _QUARANTINED_BUFFERS[:] = [Buffer()]
        try:
            backend = Fpga6Backend([None]*4,[None]*2)
            backend.overlay = Overlay()
            backend._reset_after_fault()
            self.assertEqual(events,['reset','free'])
            with self.assertRaises(RuntimeError): backend.matmul(None,None,name='must_not_run')
        finally:
            _QUARANTINED_BUFFERS[:] = saved

    def test_nlpc_dma_words_pointer_halves_and_cache_sync(self):
        import types
        buffers = []
        class Buffer(np.ndarray):
            def __new__(cls, shape, dtype):
                obj = np.zeros(shape,dtype).view(cls)
                obj.physical_address = 0x100000000 + (len(buffers)+1)*0x1000
                obj.flushed = obj.invalidated = obj.freed = False
                buffers.append(obj)
                return obj
            def flush(self): self.flushed = True
            def invalidate(self): self.invalidated = True
            def freebuffer(self): self.freed = True
        class Mmio:
            def __init__(self,*args): self.regs = {}; self.done = False
            def write(self,offset,value):
                self.regs[offset] = value
                if offset == 0 and value == 1: self.done = True
            def read(self,offset):
                if offset == 0:
                    value = 2 if self.done else 0
                    self.done = False
                    return value
                return 0
        overlay = types.SimpleNamespace(ip_dict={'test':{'type':'qwen.local:hls:qwen_nlpc:1.0',
                                                         'phys_addr':0xA0040000,'addr_range':65536}})
        with patch.dict('sys.modules',pynq=types.SimpleNamespace(MMIO=Mmio,allocate=Buffer)):
            driver = NlpcDriver(overlay,'test')
            a = np.array([[-1.5,2.75]],np.float32)
            b = np.array([[.5,.25]],np.float32)
            job = driver.prepare(RESIDUAL_ADD,a,b)
            np.testing.assert_array_equal(buffers[0].view(np.float32),a.reshape(-1))
            self.assertTrue(all(buf.flushed for buf in buffers))
            self.assertEqual(driver.mmio.regs[0x14],1)
            self.assertEqual(driver.mmio.regs[0x10],0x1000)
            self.assertEqual(driver.mmio.regs[0x2c],1)
            buffers[2][:] = (a+b).reshape(-1).view(np.uint32)
            driver.start(job)
            np.testing.assert_array_equal(driver.wait(job),a+b)
            self.assertTrue(buffers[2].invalidated)
            self.assertTrue(all(buf.freed for buf in buffers))

    def test_real_bf16_checkpoint_reader_and_embedding_row_selection(self):
        import struct
        from export_fpga_bundle import Checkpoint
        values = np.array([[1.5,-2],[3,.25],[0,-.5]],np.float32)
        words = (values.view(np.uint32)>>16).astype('<u2').tobytes()
        header = json.dumps({'model.embed_tokens.weight': {'dtype':'BF16', 'shape':[3,2],
                                                           'data_offsets':[0,len(words)]}}).encode()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.safetensors'
            path.write_bytes(struct.pack('<Q',len(header))+header+words)
            checkpoint = Checkpoint(folder)
            np.testing.assert_array_equal(checkpoint.read('model.embed_tokens.weight',[2,0,2]),values[[2,0,2]])
            # Reject truncation instead of reading unrelated bytes as weights.
            path.write_bytes(struct.pack('<Q',len(header))+header+words[:-2])
            with self.assertRaises(ValueError): checkpoint.read('model.embed_tokens.weight')

    def test_bad_bundle_rejected(self):
        data, metadata = fixture()
        validate_bundle(data,metadata)
        metadata['block_index'] = 1
        with self.assertRaises(ValueError): validate_bundle(data,metadata)
        metadata['block_index'] = 0
        data['self_attn.q_proj.weight'] = data['self_attn.q_proj.weight'].T
        with self.assertRaises(ValueError): validate_bundle(data,metadata)

    def test_old_overlay_rejected_before_programming(self):
        class OldOverlay:
            ip_dict = {name: {} for name in ('axi_x2_mvm','axi_x2_mvm_1','axi_x2_mvm_2','axi_x2_mvm_3','axi_x2_mvm_4','axi_x2_mvm_5')}
            def __init__(self,*args,**kwargs): self.downloaded = False
            def download(self): raise AssertionError('Must not program old overlay')
        import types
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'old.bit'
            path.touch(); path.with_suffix('.hwh').touch()
            with patch.dict('sys.modules',pynq=types.SimpleNamespace(Overlay=OldOverlay)):
                with self.assertRaisesRegex(RuntimeError,'missing qwen_nlpc_4'):
                    Fpga6Backend.open(path,'unused.py')

    def test_timeout_retains_buffers_and_disables_driver(self):
        class Buffer:
            def freebuffer(self): raise AssertionError('Cannot free live DMA allocation')
        class Mmio:
            def read(self, offset): return 0
        driver = NlpcDriver.__new__(NlpcDriver)
        driver.mmio, driver.failed = Mmio(), False
        job = {'started': True, 'deadline': 0, 'buffers': [Buffer(),Buffer(),Buffer()]}
        driver.active = job
        with self.assertRaises(TimeoutError): driver.wait(job)
        self.assertTrue(driver.failed)
        with self.assertRaises(RuntimeError): driver.prepare(ADD_I32,np.zeros((1,1),np.int32))

    def test_hardware_cli_rejects_other_blocks_before_model_loading(self):
        result = subprocess.run([os.sys.executable,'-B',str(ROOT/'main.py'),'--prompt','test',
                                '--execution-target','fpga6','--block-index','1'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('only block 1',result.stderr)


if __name__ == '__main__':
    unittest.main()
