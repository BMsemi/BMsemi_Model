from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np


def normalize_float_format(float_format: str | None) -> str:
    fmt = "bf16" if float_format is None else str(float_format).lower()
    aliases = {
        "bfloat16": "bf16",
        "torch.bfloat16": "bf16",
        "float32": "fp32",
        "torch.float32": "fp32",
        "none": "fp32",
    }
    fmt = aliases.get(fmt, fmt)
    if fmt not in {"bf16", "fp32"}:
        raise ValueError(f"Unsupported float_format {float_format!r}; use 'bf16' or 'fp32'.")
    return fmt


def round_float_format(x: np.ndarray, float_format: str | None = "bf16") -> np.ndarray:
    """Round runtime-module boundary values to BF16 or FP32, returned as float32."""

    fmt = normalize_float_format(float_format)
    arr = np.asarray(x, dtype=np.float32)
    if fmt == "fp32" or arr.size == 0:
        return arr.astype(np.float32, copy=True)

    out = np.ascontiguousarray(arr, dtype=np.float32).copy()
    u = out.view(np.uint32)
    finite = np.isfinite(out)
    rounded = u[finite] + np.uint32(0x7FFF) + ((u[finite] >> np.uint32(16)) & np.uint32(1))
    u[finite] = rounded & np.uint32(0xFFFF0000)
    return out


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "shape": [int(x) for x in value.shape],
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _shape(value: Any) -> list[int] | list[list[int]]:
    if isinstance(value, np.ndarray):
        return [int(x) for x in value.shape]
    if isinstance(value, (list, tuple)):
        return [_shape(v) for v in value]
    return []


@dataclass(frozen=True)
class RuntimeModuleCall:
    index: int
    module: str
    stage: str
    location: str
    output_shape: list[int] | list[list[int]]
    detail: dict[str, Any]


class ModuleCallRecorder:
    """Small execution recorder for non-PE runtime modules."""

    def __init__(self, *, keep_calls: bool = True):
        self.keep_calls = bool(keep_calls)
        self.calls: list[RuntimeModuleCall] = []
        self.counts: Counter[str] = Counter()
        self._next_index = 0

    def record(
        self,
        module: str,
        stage: str,
        output: Any,
        *,
        location: str = "outside_pe",
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.counts[str(module)] += 1
        if not self.keep_calls:
            self._next_index += 1
            return
        self.calls.append(
            RuntimeModuleCall(
                index=self._next_index,
                module=str(module),
                stage=str(stage),
                location=str(location),
                output_shape=_shape(output),
                detail=_json_value(detail or {}),
            )
        )
        self._next_index += 1

    def summary(self, *, max_calls: int = 200) -> dict[str, Any]:
        ordered_modules: list[str] = []
        for call in self.calls:
            if call.module not in ordered_modules:
                ordered_modules.append(call.module)
        if not ordered_modules:
            ordered_modules = list(self.counts.keys())
        kept = self.calls[: max(0, int(max_calls))]
        return {
            "module_location": "outside_pe",
            "execution_order_first_seen": ordered_modules,
            "call_counts": {name: int(self.counts[name]) for name in sorted(self.counts)},
            "calls": [call.__dict__ for call in kept],
            "truncated_calls": max(0, len(self.calls) - len(kept)),
        }


class BF16ArithmeticModule:
    """Explicit non-PE arithmetic boundary used by the runtime modules."""

    name = "BF16 arithmetic boundary"
    location = "outside_pe"

    def __init__(self, float_format: str = "bf16", recorder: ModuleCallRecorder | None = None):
        self.float_format = normalize_float_format(float_format)
        self.recorder = recorder

    def __call__(self, x: np.ndarray, *, stage: str, detail: dict[str, Any] | None = None) -> np.ndarray:
        y = round_float_format(x, self.float_format)
        if self.recorder is not None:
            meta = {"float_format": self.float_format}
            if detail:
                meta.update(detail)
            self.recorder.record(self.name, stage, y, location=self.location, detail=meta)
        return y


class RuntimeModule:
    location = "outside_pe"

    def __init__(
        self,
        name: str,
        *,
        arithmetic: BF16ArithmeticModule | None = None,
        recorder: ModuleCallRecorder | None = None,
    ):
        self.name = str(name)
        self.arithmetic = arithmetic
        self.recorder = recorder

    def _finish(
        self,
        output: np.ndarray,
        *,
        stage: str,
        detail: dict[str, Any] | None = None,
        round_output: bool = True,
    ) -> np.ndarray:
        y = np.asarray(output, dtype=np.float32)
        if round_output and self.arithmetic is not None:
            y = self.arithmetic(y, stage=f"{stage}:bf16", detail={"source_module": self.name})
        if self.recorder is not None:
            self.recorder.record(self.name, stage, y, location=self.location, detail=detail or {})
        return y


class EmbeddingLookupModule(RuntimeModule):
    def __init__(
        self,
        embedding_table: np.ndarray,
        *,
        arithmetic: BF16ArithmeticModule | None = None,
        recorder: ModuleCallRecorder | None = None,
    ):
        super().__init__("Embedding SRAM lookup", arithmetic=arithmetic, recorder=recorder)
        table = np.asarray(embedding_table, dtype=np.float32)
        if table.ndim != 2:
            raise ValueError(f"embedding_table must be 2D, got shape {table.shape}")
        self.embedding_table = table

    @property
    def vocab_size(self) -> int:
        return int(self.embedding_table.shape[0])

    def __call__(self, token_ids: np.ndarray, *, stage: str = "embedding") -> np.ndarray:
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.size == 0:
            out = np.zeros((*ids.shape, self.embedding_table.shape[1]), dtype=np.float32)
        else:
            out = self.embedding_table[np.mod(ids, self.vocab_size)]
        return self._finish(out, stage=stage, detail={"tokens": int(ids.size), "vocab_size": self.vocab_size})


class HiddenVectorBufferModule(RuntimeModule):
    def __init__(
        self,
        *,
        arithmetic: BF16ArithmeticModule | None = None,
        recorder: ModuleCallRecorder | None = None,
    ):
        super().__init__("hidden vector buffer", arithmetic=arithmetic, recorder=recorder)
        self.last: np.ndarray | None = None

    def __call__(self, hidden: np.ndarray, *, stage: str = "hidden_buffer") -> np.ndarray:
        self.last = np.asarray(hidden, dtype=np.float32).copy()
        return self._finish(self.last, stage=stage, detail={"buffered": True})


class RMSNormModule(RuntimeModule):
    def __init__(
        self,
        eps: float = 1.0e-6,
        weight: np.ndarray | None = None,
        *,
        name: str = "RMSNorm",
        arithmetic: BF16ArithmeticModule | None = None,
        recorder: ModuleCallRecorder | None = None,
    ):
        super().__init__(name, arithmetic=arithmetic, recorder=recorder)
        self.eps = float(eps)
        self.weight = None if weight is None else np.asarray(weight, dtype=np.float32)

    def __call__(self, x: np.ndarray, *, stage: str = "rmsnorm") -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        rms = np.sqrt(np.mean(arr * arr, axis=-1, keepdims=True) + self.eps)
        out = arr / rms
        if self.weight is not None:
            out = out * self.weight
        return self._finish(out, stage=stage, detail={"eps": self.eps})


class QKNormModule(RuntimeModule):
    def __init__(
        self,
        eps: float = 1.0e-6,
        *,
        arithmetic: BF16ArithmeticModule | None = None,
        recorder: ModuleCallRecorder | None = None,
    ):
        super().__init__("q_norm / k_norm", arithmetic=arithmetic, recorder=recorder)
        self.q_norm = RMSNormModule(eps=eps, name="q_norm", arithmetic=arithmetic, recorder=recorder)
        self.k_norm = RMSNormModule(eps=eps, name="k_norm", arithmetic=arithmetic, recorder=recorder)

    def __call__(self, q: np.ndarray, k: np.ndarray, *, stage: str = "qk_norm") -> tuple[np.ndarray, np.ndarray]:
        q_out = self.q_norm(q, stage=f"{stage}:q")
        k_out = self.k_norm(k, stage=f"{stage}:k")
        if self.recorder is not None:
            self.recorder.record(self.name, stage, (q_out, k_out), location=self.location, detail={"eps": self.q_norm.eps})
        return q_out, k_out


class RoPEModule(RuntimeModule):
    def __init__(
        self,
        base: float = 10000.0,
        *,
        arithmetic: BF16ArithmeticModule | None = None,
        recorder: ModuleCallRecorder | None = None,
    ):
        super().__init__("RoPE positional encoding", arithmetic=arithmetic, recorder=recorder)
        self.base = float(base)

    def __call__(
        self,
        q: np.ndarray,
        k: np.ndarray,
        *,
        positions: np.ndarray | None = None,
        stage: str = "rope",
    ) -> tuple[np.ndarray, np.ndarray]:
        q_arr = np.asarray(q, dtype=np.float32)
        k_arr = np.asarray(k, dtype=np.float32)
        if positions is None:
            positions_arr = np.arange(q_arr.shape[-2], dtype=np.float32)
        else:
            positions_arr = np.asarray(positions, dtype=np.float32).reshape(-1)
        q_out = self._rotate(q_arr, positions_arr)
        k_out = self._rotate(k_arr, positions_arr)
        if self.arithmetic is not None:
            q_out = self.arithmetic(q_out, stage=f"{stage}:q_bf16", detail={"source_module": self.name})
            k_out = self.arithmetic(k_out, stage=f"{stage}:k_bf16", detail={"source_module": self.name})
        if self.recorder is not None:
            self.recorder.record(
                self.name,
                stage,
                (q_out, k_out),
                location=self.location,
                detail={"base": self.base, "positions": int(positions_arr.size)},
            )
        return q_out, k_out

    def _rotate(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        if x.ndim < 2:
            raise ValueError(f"RoPE input must have sequence and channel axes, got shape {x.shape}")
        seq_len = int(x.shape[-2])
        if positions.size != seq_len:
            raise ValueError(f"RoPE positions length {positions.size} does not match sequence length {seq_len}")
        dim = int(x.shape[-1])
        rot_dim = dim - (dim % 2)
        if rot_dim == 0:
            return x.copy()
        out = x.copy()
        pair_count = rot_dim // 2
        inv_freq = self.base ** (-np.arange(0, rot_dim, 2, dtype=np.float32) / float(rot_dim))
        angles = positions.reshape((1,) * (x.ndim - 2) + (seq_len, 1)) * inv_freq.reshape(
            (1,) * (x.ndim - 1) + (pair_count,)
        )
        cos = np.cos(angles).astype(np.float32)
        sin = np.sin(angles).astype(np.float32)
        even = x[..., :rot_dim:2]
        odd = x[..., 1:rot_dim:2]
        out[..., :rot_dim:2] = even * cos - odd * sin
        out[..., 1:rot_dim:2] = even * sin + odd * cos
        return out


class KVCacheControlModule:
    name = "KV-cache indexing/control"
    location = "outside_pe"

    def __init__(self, cache: Any, *, recorder: ModuleCallRecorder | None = None):
        self.cache = cache
        self.recorder = recorder

    def append_and_read(
        self,
        *,
        layer_idx: int,
        token_idx: int,
        key: np.ndarray,
        value: np.ndarray,
        now_s: float | None = None,
        stage: str = "kv_cache",
    ) -> Any:
        self.cache.append(layer_idx, token_idx, key, value, now_s=now_s)
        read = self.cache.read_layer(layer_idx, upto_token_idx=token_idx, now_s=now_s)
        if self.recorder is not None:
            self.recorder.record(
                self.name,
                stage,
                (read.keys, read.values),
                location=self.location,
                detail={
                    "layer_idx": int(layer_idx),
                    "token_idx": int(token_idx),
                    "context_tokens": int(read.token_ids.shape[0]),
                },
            )
        return read


class CausalMaskModule(RuntimeModule):
    def __init__(
        self,
        mask_value: float = -1.0e30,
        *,
        arithmetic: BF16ArithmeticModule | None = None,
        recorder: ModuleCallRecorder | None = None,
    ):
        super().__init__("causal mask", arithmetic=arithmetic, recorder=recorder)
        self.mask_value = float(mask_value)

    def __call__(
        self,
        scores: np.ndarray,
        *,
        query_positions: np.ndarray | None = None,
        key_positions: np.ndarray | None = None,
        stage: str = "causal_mask",
    ) -> np.ndarray:
        arr = np.asarray(scores, dtype=np.float32)
        q_len = int(arr.shape[-2])
        k_len = int(arr.shape[-1])
        q_pos = np.arange(q_len, dtype=np.int64) if query_positions is None else np.asarray(query_positions, dtype=np.int64)
        k_pos = np.arange(k_len, dtype=np.int64) if key_positions is None else np.asarray(key_positions, dtype=np.int64)
        q_pos = q_pos.reshape(-1)
        k_pos = k_pos.reshape(-1)
        if q_pos.size != q_len:
            raise ValueError(f"query_positions length {q_pos.size} does not match score query length {q_len}")
        if k_pos.size != k_len:
            raise ValueError(f"key_positions length {k_pos.size} does not match score key length {k_len}")
        mask = k_pos.reshape(1, k_len) <= q_pos.reshape(q_len, 1)
        masked = np.where(mask.reshape((1,) * (arr.ndim - 2) + (q_len, k_len)), arr, self.mask_value)
        return self._finish(masked, stage=stage, detail={"masked_entries": int(mask.size - np.count_nonzero(mask))})


class SoftmaxModule(RuntimeModule):
    def __init__(
        self,
        *,
        arithmetic: BF16ArithmeticModule | None = None,
        recorder: ModuleCallRecorder | None = None,
    ):
        super().__init__("softmax", arithmetic=arithmetic, recorder=recorder)

    def __call__(self, x: np.ndarray, *, stage: str = "softmax") -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        finite = np.isfinite(arr)
        safe = np.where(finite, arr, -np.inf)
        row_max = np.max(safe, axis=-1, keepdims=True)
        row_max = np.where(np.isfinite(row_max), row_max, 0.0)
        shifted = np.where(finite, arr - row_max, -np.inf)
        exp_x = np.where(finite, np.exp(np.clip(shifted, -80.0, 80.0)), 0.0)
        denom = np.sum(exp_x, axis=-1, keepdims=True)
        probs = np.divide(exp_x, denom, out=np.zeros_like(exp_x), where=denom > 0.0)
        return self._finish(probs, stage=stage, detail={"axis": -1})


class SiLUModule(RuntimeModule):
    def __init__(
        self,
        *,
        arithmetic: BF16ArithmeticModule | None = None,
        recorder: ModuleCallRecorder | None = None,
    ):
        super().__init__("SiLU activation", arithmetic=arithmetic, recorder=recorder)

    def __call__(self, x: np.ndarray, *, stage: str = "silu") -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        clipped = np.clip(arr, -80.0, 80.0)
        sigmoid = 1.0 / (1.0 + np.exp(-clipped))
        return self._finish(arr * sigmoid, stage=stage, detail={"formula": "x * sigmoid(x)"})


class SwiGLUMultiplyModule(RuntimeModule):
    def __init__(
        self,
        silu: SiLUModule,
        *,
        arithmetic: BF16ArithmeticModule | None = None,
        recorder: ModuleCallRecorder | None = None,
    ):
        super().__init__("SwiGLU multiply", arithmetic=arithmetic, recorder=recorder)
        self.silu = silu

    def __call__(self, gate_up: np.ndarray, *, stage: str = "swiglu") -> np.ndarray:
        arr = np.asarray(gate_up, dtype=np.float32)
        if arr.shape[-1] % 2 != 0:
            raise ValueError(f"SwiGLU expects an even last dimension, got {arr.shape[-1]}")
        gate, up = np.split(arr, 2, axis=-1)
        activated_gate = self.silu(gate, stage=f"{stage}:silu_gate")
        out = activated_gate * up
        return self._finish(out, stage=stage, detail={"formula": "silu(gate) * up"})


class ResidualAddModule(RuntimeModule):
    def __init__(
        self,
        *,
        arithmetic: BF16ArithmeticModule | None = None,
        recorder: ModuleCallRecorder | None = None,
    ):
        super().__init__("residual add", arithmetic=arithmetic, recorder=recorder)

    def __call__(self, residual: np.ndarray, branch: np.ndarray, *, stage: str = "residual_add") -> np.ndarray:
        out = np.asarray(residual, dtype=np.float32) + np.asarray(branch, dtype=np.float32)
        return self._finish(out, stage=stage, detail={"inputs": 2})


class TokenSamplingModule:
    name = "token sampling / argmax / top-k / top-p"
    location = "outside_pe"

    def __init__(self, *, recorder: ModuleCallRecorder | None = None, rng: np.random.Generator | None = None):
        self.recorder = recorder
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def __call__(
        self,
        logits: np.ndarray,
        *,
        strategy: str = "argmax",
        top_k: int | None = None,
        top_p: float | None = None,
        stage: str = "sampling",
    ) -> np.ndarray:
        arr = np.asarray(logits, dtype=np.float32)
        flat = arr.reshape(-1, arr.shape[-1])
        strategy_norm = str(strategy).lower()
        if strategy_norm == "argmax" and top_k is None and top_p is None:
            tokens = np.argmax(flat, axis=-1).astype(np.int64)
        else:
            tokens = np.array(
                [self._sample_row(row, strategy=strategy_norm, top_k=top_k, top_p=top_p) for row in flat],
                dtype=np.int64,
            )
        tokens = tokens.reshape(arr.shape[:-1])
        if self.recorder is not None:
            self.recorder.record(
                self.name,
                stage,
                tokens,
                location=self.location,
                detail={
                    "strategy": strategy_norm,
                    "top_k": None if top_k is None else int(top_k),
                    "top_p": None if top_p is None else float(top_p),
                },
            )
        return tokens

    def _sample_row(
        self,
        row: np.ndarray,
        *,
        strategy: str,
        top_k: int | None,
        top_p: float | None,
    ) -> int:
        logits = np.asarray(row, dtype=np.float32)
        allowed = np.ones((logits.shape[0],), dtype=bool)
        if top_k is not None and int(top_k) > 0:
            k = min(int(top_k), logits.shape[0])
            threshold = np.partition(logits, -k)[-k]
            allowed &= logits >= threshold
        probs = SoftmaxModule()(logits[None, :], stage="sampling_softmax")[0]
        if top_p is not None:
            p = min(max(float(top_p), 0.0), 1.0)
            order = np.argsort(probs)[::-1]
            cumulative = np.cumsum(probs[order])
            nucleus = cumulative <= p
            if nucleus.size:
                nucleus[0] = True
            keep = np.zeros_like(allowed)
            keep[order[nucleus]] = True
            allowed &= keep
        if not np.any(allowed):
            return int(np.argmax(logits))
        masked_probs = np.where(allowed, probs, 0.0)
        total = float(np.sum(masked_probs))
        if total <= 0.0:
            return int(np.argmax(logits))
        masked_probs = masked_probs / total
        return int(self.rng.choice(np.arange(logits.shape[0]), p=masked_probs))


def runtime_module_catalog() -> list[dict[str, str]]:
    return [
        {"module": "Embedding SRAM lookup", "location": "outside_pe", "role": "token id to hidden vector read"},
        {"module": "hidden vector buffer", "location": "outside_pe", "role": "activation staging between modules"},
        {"module": "RMSNorm", "location": "outside_pe", "role": "pre-attention, pre-FFN, and final hidden-state normalization"},
        {"module": "q_norm / k_norm", "location": "outside_pe", "role": "separate query/key head normalization"},
        {"module": "RoPE positional encoding", "location": "outside_pe", "role": "sin/cos pair rotation for Q and K"},
        {"module": "KV-cache indexing/control", "location": "outside_pe", "role": "token-slot append/read control"},
        {"module": "causal mask", "location": "outside_pe", "role": "hide future tokens before softmax"},
        {"module": "softmax", "location": "outside_pe", "role": "attention probability normalization"},
        {"module": "SiLU activation", "location": "outside_pe", "role": "gate nonlinearity"},
        {"module": "SwiGLU multiply", "location": "outside_pe", "role": "silu(gate) times up branch"},
        {"module": "residual add", "location": "outside_pe", "role": "skip connection accumulation"},
        {"module": "token sampling / argmax / top-k / top-p", "location": "outside_pe", "role": "choose next token from logits"},
        {"module": "BF16 arithmetic boundary", "location": "outside_pe", "role": "direct BF16 rounding for non-PE module outputs"},
    ]
