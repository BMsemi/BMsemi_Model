"""Hardware-only Qwen3 block-0 prefill orchestration.

PS arithmetic is limited to quantization/dequantization. Tensor layout changes
and copies run here; matrix products, reduction of C tiles, and nonlinear block
operations are dispatched to PL. There is deliberately no CPU fallback.
"""
from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

from .ps_int8_quantization import prepare_mvm, round_bf16

PROBE, ADD_I32, RMS_NORM, ROPE, CAUSAL_SOFTMAX, SILU_MUL, RESIDUAL_ADD = range(7)
ABI = 0x514E0001
MAX_ROWS, MAX_COLS = 4096, 6144
LPC_NAMES = ("axi_x2_mvm", "axi_x2_mvm_1", "axi_x2_mvm_2", "axi_x2_mvm_3")
NLPC_NAMES = ("qwen_nlpc_4", "qwen_nlpc_5")
# Keep DMA allocations alive after timeout: HLS has no command-level abort.
# The overlay must be reset before attempting another command.
_QUARANTINED_BUFFERS = []


def _word(value):
    return int(np.asarray(value, dtype=np.float32).view(np.uint32))


class NlpcDriver:
    def __init__(self, overlay, name, timeout_s=30):
        from pynq import MMIO, allocate
        info = overlay.ip_dict[name]
        if info.get("type") != "qwen.local:hls:qwen_nlpc:1.0":
            raise RuntimeError(f"{name}: incorrect NLPC IP type: {info.get('type')}")
        self.mmio = MMIO(info["phys_addr"], info["addr_range"])
        self.allocate = allocate
        self.timeout_s = timeout_s
        self.failed = False
        self.active = None

    def prepare(self, op, a, b=None, *, row_offset=0, epsilon=1e-6, scale=1.0):
        if self.failed or self.active is not None:
            raise RuntimeError("NLPC busy or faulted; reload the overlay after a timeout")
        dtype = np.int32 if op == ADD_I32 else np.float32
        a = np.asarray(a, dtype=dtype, order="C")
        if a.ndim != 2 or not 1 <= a.shape[0] <= MAX_ROWS or not 1 <= a.shape[1] <= MAX_COLS:
            raise ValueError("NLPC expects 1..4096 rows and 1..6144 columns")
        rows, cols = a.shape
        if op not in range(7):
            raise ValueError("Unknown NLPC opcode")
        if op == ROPE and cols % 2:
            raise ValueError("RoPE requires an even head dimension")
        if op == CAUSAL_SOFTMAX and not (0 <= row_offset < cols and rows + row_offset <= cols):
            raise ValueError("Invalid causal query row offset")
        required_b = cols if op == RMS_NORM else a.size * (2 if op == ROPE else 1)
        if op in (PROBE, CAUSAL_SOFTMAX):
            b = np.zeros(1, dtype=dtype)
        else:
            b = np.asarray(b, dtype=dtype, order="C")
            if b.size != required_b:
                raise ValueError(f"Opcode {op} requires {required_b} B words")
        buffers = []
        try:
            for data in (a, b, np.zeros_like(a)):
                buf = self.allocate(shape=(data.size,), dtype=np.uint32)
                buffers.append(buf)
                buf[:] = np.ascontiguousarray(data).reshape(-1).view(np.uint32)
                buf.flush()
            for reg, buf in zip((0x10, 0x1c, 0x28), buffers):
                address = int(buf.physical_address)
                self.mmio.write(reg, address & 0xffffffff)
                self.mmio.write(reg + 4, address >> 32)
            for reg, value in ((0x34, rows), (0x3c, cols), (0x44, op),
                               (0x4c, row_offset), (0x54, _word(epsilon)), (0x5c, _word(scale))):
                self.mmio.write(reg, int(value))
            job = {"buffers": buffers, "shape": a.shape, "dtype": dtype, "started": False}
            self.active = job
            return job
        except Exception:
            for buf in buffers:
                buf.freebuffer()
            raise

    def start(self, job):
        if job is not self.active or job["started"]:
            raise RuntimeError("Invalid NLPC job")
        self.mmio.read(0)  # clear any previous ap_done (clear-on-read)
        job["started"] = True  # retain allocations even if the MMIO write fails
        job["deadline"] = time.monotonic() + self.timeout_s
        self.mmio.write(0, 1)

    def wait(self, job):
        if job is not self.active or not job["started"]:
            raise RuntimeError("NLPC job has not started")
        try:
            while not self.mmio.read(0) & 2:
                if time.monotonic() > job["deadline"]:
                    raise TimeoutError("NLPC timeout; reload overlay before reusing DMA buffers")
                time.sleep(.0001)
            job["started"] = False  # ap_done proves the memory transactions completed
            status = self.mmio.read(0x64)
            if status:
                raise RuntimeError(f"NLPC status {status}: 1=shape, 2=opcode, 3=INT32 overflow, 4=nonfinite")
            job["buffers"][2].invalidate()
            return np.asarray(job["buffers"][2]).view(job["dtype"]).reshape(job["shape"]).copy()
        finally:
            self.release(job)

    def release(self, job):
        if job is not self.active:
            return
        if job["started"]:
            self.failed = True
            _QUARANTINED_BUFFERS.extend(job["buffers"])
        else:
            for buf in job["buffers"]:
                buf.freebuffer()
        self.active = None


class Fpga6Backend:
    def __init__(self, lpc, nlpc):
        if len(lpc) != 4 or len(nlpc) != 2:
            raise ValueError("Four LPC drivers and two NLPC drivers are required")
        self.lpc, self.nlpc = lpc, nlpc
        self.counts = Counter()
        self.stages = []
        self.faulted = False

    def _reset_after_fault(self):
        """Stop PL masters before releasing allocations retained by timed-out IP.

        This backend cannot be resumed after a fault: construct a new backend.
        The real-board reset/drain behavior still requires board validation.
        """
        self.faulted = True
        overlay = getattr(self, 'overlay', None)
        if overlay is not None:
            overlay.download()
            for buf in _QUARANTINED_BUFFERS:
                buf.freebuffer()
            _QUARANTINED_BUFFERS.clear()

    @classmethod
    def open(cls, bitstream, driver_path, timeout_s=30):
        from pynq import Overlay
        bitstream = Path(bitstream)
        if not bitstream.is_file() or not bitstream.with_suffix('.hwh').is_file():
            raise FileNotFoundError("Supply the Qwen6 .bit file with its matching .hwh")
        # Parse HWH before programming. An old six-MVM overlay is incompatible.
        overlay = Overlay(str(bitstream), download=False)
        for name in LPC_NAMES + NLPC_NAMES:
            if name not in overlay.ip_dict:
                raise RuntimeError(f"Not a Qwen6 overlay: missing {name}")
        for name in NLPC_NAMES:
            if overlay.ip_dict[name].get('type') != 'qwen.local:hls:qwen_nlpc:1.0':
                raise RuntimeError(f"Not a Qwen6 NLPC core: {name}")
        spec = importlib.util.spec_from_file_location("qwen_x2_board_driver", str(driver_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load existing X2 driver: {driver_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        overlay.download()
        backend = cls([module.AxiX2MvmPynq(overlay=overlay, accel_name=name, timeout_s=timeout_s)
                       for name in LPC_NAMES],
                      [NlpcDriver(overlay, name, timeout_s) for name in NLPC_NAMES])
        backend.overlay = overlay
        for engine in range(2):
            result = backend._nlpc_jobs([(engine, PROBE, np.zeros((1, 1), np.float32), None, {})])[0]
            if int(result.view(np.uint32)[0, 0]) != ABI:
                raise RuntimeError(f"Engine {engine + 4}: NLPC ABI mismatch")
        return backend

    def _nlpc_jobs(self, calls):
        if self.faulted:
            raise RuntimeError('FPGA backend faulted; open a new backend before continuing')
        jobs = []
        failed = False
        try:
            for engine, op, a, b, kw in calls:
                jobs.append((engine, op, self.nlpc[engine].prepare(op, a, b, **kw)))
            for engine, op, job in jobs:
                self.nlpc[engine].start(job)
            results = []
            for engine, op, job in jobs:
                results.append(self.nlpc[engine].wait(job))
                self.counts[f"engine_{engine+4}_op_{op}"] += 1
            return results
        except BaseException:
            failed = True
            raise
        finally:
            for engine, op, job in jobs:
                self.nlpc[engine].release(job)
            if failed:
                self._reset_after_fault()

    def vector(self, op, a, b=None, **kwargs):
        """Divide independent rows between NLPC 4 and 5; preserve causal indices."""
        a = np.asarray(a)
        if a.ndim != 2 or a.shape[0] == 0:
            raise ValueError("NLPC input must have at least one row")
        result = np.empty_like(a)
        for start in range(0, len(a), 2 * MAX_ROWS):
            stop = min(start + 2 * MAX_ROWS, len(a))
            middle = start + (stop - start + 1) // 2
            spans = [(start, middle), (middle, stop)]
            calls = []
            for engine, (lo, hi) in enumerate(spans):
                if hi == lo:
                    continue
                sliced_b = b if op == RMS_NORM or b is None else b[lo:hi]
                kw = dict(kwargs)
                if op == CAUSAL_SOFTMAX:
                    kw['row_offset'] = kw.get('row_offset', 0) + lo
                calls.append((engine, op, a[lo:hi], sliced_b, kw))
            for (lo, hi), output in zip(spans, self._nlpc_jobs(calls)):
                result[lo:hi] = output
        return result

    def matmul(self, x, weights_kc, *, name):
        """INT8 LPC tiles; all C partials accumulated by NLPC integer add."""
        if self.faulted:
            raise RuntimeError('FPGA backend faulted; open a new backend before continuing')
        prepared = prepare_mvm(x, weights_kc)
        tokens, channels = prepared.activations.shape
        outputs = prepared.weights.shape[1]
        integer_sum = np.empty((tokens, outputs), np.int32)
        started = time.monotonic()
        for t in range(0, tokens, 32):
            # One output tile per LPC. Launch all four before waiting.
            for group_k in range(0, outputs, 128):
                ks = list(range(group_k, min(group_k+128, outputs), 32))
                accumulators = [None] * len(ks)
                for c in range(0, channels, 32):
                    jobs = []
                    failed = False
                    try:
                        for engine, k in enumerate(ks):
                            job = self.lpc[engine].prepare_mvm_batch(
                                prepared.activations[t:t+32, c:c+32],
                                prepared.weights[c:c+32, k:k+32])
                            jobs.append((engine, job))
                        for engine, job in jobs:
                            self.lpc[engine].start_job(job)
                        partials = []
                        for engine, job in jobs:
                            partials.append(self.lpc[engine].wait_job(job))
                            self.counts[f"engine_{engine}_mvm"] += 1
                    except BaseException:
                        failed = True
                        raise
                    finally:
                        for engine, job in jobs:
                            self.lpc[engine].release_job(job, abort=True)
                        if failed:
                            self._reset_after_fault()
                    if c == 0:
                        accumulators = partials
                    else:
                        for first in range(0, len(ks), 2):
                            calls = [(j-first, ADD_I32, accumulators[j], partials[j], {})
                                     for j in range(first, min(first+2, len(ks)))]
                            for j, output in enumerate(self._nlpc_jobs(calls), first):
                                accumulators[j] = output
                for k, acc in zip(ks, accumulators):
                    integer_sum[t:t+acc.shape[0], k:k+acc.shape[1]] = acc
        self.stages.append({"name": name, "input_shape": list(x.shape),
                            "weight_shape": list(weights_kc.shape),
                            "wall_seconds": time.monotonic()-started})
        return prepared.reconstruct(integer_sum)


WEIGHT_KEYS = ('input_layernorm.weight', 'post_attention_layernorm.weight',
               'self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
               'self_attn.o_proj.weight', 'self_attn.q_norm.weight', 'self_attn.k_norm.weight',
               'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight')


def validate_bundle(data, metadata):
    if metadata.get('format') != 'qwen6-bundle-v1':
        raise ValueError("Unsupported FPGA bundle format; use export_fpga_bundle.py")
    cfg = metadata['config']
    if metadata.get('block_index') != 0 or cfg.get('model_type') != 'qwen3':
        raise ValueError("Only Qwen3 block index 0 is supported")
    if cfg.get('attention_bias', False) or cfg.get('rope_scaling') or cfg.get('sliding_window'):
        raise ValueError("Bias, scaled RoPE and sliding-window variants are not supported")
    if cfg.get('hidden_act') != 'silu':
        raise ValueError("Only SiLU Qwen3 MLP is supported")
    epsilon = float(cfg.get('rms_norm_eps', 0))
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("rms_norm_eps must be positive and finite")
    h, d, q, kv, ff = (int(cfg[k]) for k in ('hidden_size', 'head_dim', 'num_attention_heads',
                                           'num_key_value_heads', 'intermediate_size'))
    if min(h, d, q, kv, ff) < 1 or q % kv or d % 2 or max(h, d, ff) > MAX_COLS:
        raise ValueError("Unsupported model dimensions")
    x = data['input']
    if x.ndim != 2 or x.shape[1] != h or not 1 <= len(x) <= 128:
        raise ValueError("Block input must be [1..128 tokens, hidden_size]")
    if len(metadata.get('token_ids', [])) != len(x) or metadata.get('position_ids') != list(range(len(x))):
        raise ValueError("Bundle token IDs/positions do not match the prefill input")
    shapes = [(h,), (h,), (q*d, h), (kv*d, h), (kv*d, h), (h, q*d),
              (d,), (d,), (ff, h), (ff, h), (h, ff)]
    for key, shape in zip(WEIGHT_KEYS, shapes):
        if data[key].shape != shape:
            raise ValueError(f"{key}: expected {shape}, got {data[key].shape}")
    if data['rotary'].shape != (len(x), 2*d):
        raise ValueError("Expected one cos/sin row per token")
    for key in ('input', 'rotary') + WEIGHT_KEYS:
        if data[key].dtype != np.float32 or not np.isfinite(data[key]).all():
            raise ValueError(f"{key} must contain finite float32 containers")
        if not np.array_equal(round_bf16(data[key]), data[key]):
            raise ValueError(f"{key} must be BF16-rounded")


def run_block(backend, data, metadata):
    validate_bundle(data, metadata)
    cfg = metadata['config']
    h, d, heads, kvheads = (int(cfg[k]) for k in ('hidden_size', 'head_dim', 'num_attention_heads',
                                               'num_key_value_heads'))
    x = data['input']
    t = len(x)
    eps = float(cfg['rms_norm_eps'])
    def linear(value, key):
        return backend.matmul(value, data[key+'.weight'], name=key)
    normalized = backend.vector(RMS_NORM, x, data['input_layernorm.weight'], epsilon=eps)
    q = linear(normalized, 'self_attn.q_proj').reshape(t*heads, d)
    k = linear(normalized, 'self_attn.k_proj').reshape(t*kvheads, d)
    v = linear(normalized, 'self_attn.v_proj').reshape(t, kvheads, d).transpose(1, 0, 2)
    q = backend.vector(RMS_NORM, q, data['self_attn.q_norm.weight'], epsilon=eps)
    k = backend.vector(RMS_NORM, k, data['self_attn.k_norm.weight'], epsilon=eps)
    q = backend.vector(ROPE, q, np.repeat(data['rotary'], heads, axis=0)).reshape(t, heads, d).transpose(1, 0, 2)
    k = backend.vector(ROPE, k, np.repeat(data['rotary'], kvheads, axis=0)).reshape(t, kvheads, d).transpose(1, 0, 2)
    context = np.empty((heads, t, d), np.float32)
    for head in range(heads):
        kvhead = head // (heads // kvheads)
        scores = backend.matmul(q[head], k[kvhead], name=f'attention_qk_head_{head}')
        probabilities = backend.vector(CAUSAL_SOFTMAX, scores, scale=d**-.5)
        # Quantizing V across the entire sequence would let future tokens alter
        # earlier tokens' INT8 scales, even with a causal mask. Each query uses
        # only its visible prefix for both quantization and the FPGA product.
        for query in range(t):
            context[head, query:query+1] = backend.matmul(
                probabilities[query:query+1, :query+1], v[kvhead, :query+1].T,
                name=f'attention_pv_head_{head}_query_{query}')
    projected = linear(context.transpose(1, 0, 2).reshape(t, heads*d), 'self_attn.o_proj')
    residual = backend.vector(RESIDUAL_ADD, x, projected)
    normalized = backend.vector(RMS_NORM, residual, data['post_attention_layernorm.weight'], epsilon=eps)
    gate = linear(normalized, 'mlp.gate_proj')
    up = linear(normalized, 'mlp.up_proj')
    activated = backend.vector(SILU_MUL, gate, up)
    down = linear(activated, 'mlp.down_proj')
    output = backend.vector(RESIDUAL_ADD, residual, down)
    return {'output': output, 'key_cache': k, 'value_cache': v}


def run_from_args(args):
    if args.all_blocks or args.block_index not in (None, 0):
        raise ValueError("fpga6 implements only block 1 (--block-index 0), prefill only")
    if args.dtype != 'bfloat16' or args.weight_bits != 8 or args.act_bits != 8:
        raise ValueError("fpga6 requires BF16 model operands and --weight-bits 8 --act-bits 8")
    if not args.fpga_bundle or not args.fpga_bitstream or not args.fpga_driver:
        raise ValueError("fpga6 requires --fpga-bundle, --fpga-bitstream and --fpga-driver")
    with np.load(args.fpga_bundle, allow_pickle=False) as bundle:
        metadata = json.loads(str(bundle['metadata']))
        data = {key: bundle[key] for key in ('input', 'rotary') + WEIGHT_KEYS}
    validate_bundle(data, metadata)
    if metadata.get('prompt') != args.prompt:
        raise ValueError("--prompt must match the prompt used to export this bundle")
    if len(data['input']) > args.max_input_tokens:
        raise ValueError("Bundle exceeds --max-input-tokens; re-export it explicitly")
    backend = Fpga6Backend.open(args.fpga_bitstream, args.fpga_driver, args.fpga_timeout)
    start = time.monotonic()
    result = run_block(backend, data, metadata)
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out/'fpga_block1_prefill.npz', **result)
    report = {'execution_target': 'fpga6', 'physical_hardware': True, 'block_index': 0,
              'phase': 'prefill', 'tokens': len(data['input']), 'decode_executed': False,
              'cpu_model_arithmetic_fallback': False, 'lpc_dtype': 'INT8',
              'accumulator_dtype': 'INT32', 'nlpc_arithmetic': 'FP32 with BF16 output rounding',
              'ps_operations': ['BF16/INT8 conversion', 'scale reconstruction', 'layout/copies', 'DMA scheduling'],
              'engine_operations': dict(backend.counts), 'projection_stages': backend.stages,
              'wall_seconds': time.monotonic()-start, 'bundle_metadata': metadata,
              'bitstream': str(Path(args.fpga_bitstream).resolve())}
    (out/'fpga_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f"FPGA block 1 prefill completed: {len(data['input'])} tokens. Report: {out/'fpga_summary.json'}")
    return 0
