from __future__ import annotations

import ctypes
import json
import math
import mmap
import os
import struct
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _safe_print(*args, **kwargs) -> None:
    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        return


class PlBackendUnavailable(RuntimeError):
    """Raised when the real PL DMA/MMIO backend cannot be opened."""


@dataclass
class PlMvmStats:
    mvm_calls: int = 0
    correction_calls: int = 0
    dma_transactions: int = 0
    primary_dma_transactions: int = 0
    correction_dma_transactions: int = 0
    dma_tx_words: int = 0
    dma_rx_words: int = 0
    estimated_macs: int = 0
    elapsed_s: float = 0.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    return int(text, 0)


class AxiX2MvmCmaBackend:
    """
    ctypes wrapper for the ZCU106 AXI/MMIO MVM accelerator.

    The PL block computes one signed INT8 MVM per call:
    vector[rows] @ weights[rows, cols] -> int32[cols], rows/cols <= 32.
    """

    def __init__(
        self,
        *,
        lib_path: str | Path | None = None,
        source_path: str | Path | None = None,
        compile_if_missing: bool = True,
        accel_base: int | str | None = None,
        reserved_base: int | str | None = None,
        accel_uio_path: str | None = None,
    ) -> None:
        self.root = Path(__file__).resolve().parent
        self.source_path = Path(source_path) if source_path is not None else self.root / "axi_x2_mvm_cma.c"
        self.lib_path = Path(lib_path or os.environ.get("YOLO_PL_C_LIB", "") or self.root / "libaxi_x2_mvm_cma.so")
        self.accel_base = _optional_int(accel_base if accel_base is not None else os.environ.get("YOLO_PL_ACCEL_BASE"))
        self.reserved_base = _optional_int(
            reserved_base if reserved_base is not None else os.environ.get("YOLO_PL_RESERVED_BASE")
        )
        self.accel_uio_path = accel_uio_path or os.environ.get("YOLO_PL_ACCEL_UIO", "/dev/uio4")
        self.stats = PlMvmStats()
        self.max_rows_per_call = max(1, min(32, int(os.environ.get("YOLO_PL_MAX_ROWS", "32") or "32")))
        self.max_batch_per_call = max(1, min(32, int(os.environ.get("YOLO_PL_BATCH", "32") or "32")))
        self.s2mm_lane_workaround = os.environ.get("YOLO_PL_S2MM_LANE_WORKAROUND", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        default_cols = "16" if self.s2mm_lane_workaround else "2"
        default_output_words = "1024" if self.s2mm_lane_workaround else "2"
        self.max_cols_per_call = max(1, min(32, int(os.environ.get("YOLO_PL_MAX_COLS", default_cols) or default_cols)))
        self.max_output_words_per_call = max(
            1,
            min(1024, int(os.environ.get("YOLO_PL_MAX_OUTPUT_WORDS", default_output_words) or default_output_words)),
        )
        evidence_dir = os.environ.get("YOLO_PL_EVIDENCE_DIR", "").strip()
        self.evidence_dir = Path(evidence_dir) if evidence_dir else Path.cwd()
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.evidence_dir / "pl_datapath_events.jsonl"
        self.summary_path = self.evidence_dir / "pl_datapath_summary.json"
        self.report_path = self.evidence_dir / "pl_datapath_report.md"
        self.layer_records: list[dict[str, Any]] = []
        self.active_layer: dict[str, Any] | None = None
        self._transaction_role = "primary"
        self._evidence_interval = max(0, int(os.environ.get("YOLO_PL_EVIDENCE_INTERVAL", "5000") or "0"))
        self._evidence_first_transactions = max(0, int(os.environ.get("YOLO_PL_EVIDENCE_FIRST_TX", "3") or "0"))
        self._last_evidence_transaction = 0

        needs_compile = not self.lib_path.exists()
        if self.lib_path.exists() and self.source_path.exists():
            needs_compile = self.source_path.stat().st_mtime > self.lib_path.stat().st_mtime
        if needs_compile and compile_if_missing:
            self._compile_library()
        if not self.lib_path.exists():
            raise PlBackendUnavailable(f"PL backend library not found: {self.lib_path}")

        try:
            self.lib = ctypes.CDLL(str(self.lib_path))
        except OSError as exc:
            raise PlBackendUnavailable(f"Could not load {self.lib_path}: {exc}") from exc

        self.lib.axi_x2_open.argtypes = []
        self.lib.axi_x2_open.restype = ctypes.c_int
        self.lib.axi_x2_close.argtypes = []
        self.lib.axi_x2_close.restype = None
        self.lib.axi_x2_last_error.argtypes = []
        self.lib.axi_x2_last_error.restype = ctypes.c_char_p
        self.lib.axi_x2_run_i8.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.lib.axi_x2_run_i8.restype = ctypes.c_int
        self.lib.axi_x2_run_i8_batch.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.lib.axi_x2_run_i8_batch.restype = ctypes.c_int
        try:
            self.lib.axi_x2_get_stats.argtypes = [
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
            ]
            self.lib.axi_x2_get_stats.restype = ctypes.c_int
            self._has_c_stats = True
        except AttributeError:
            self._has_c_stats = False

        self._activate_environment()
        rc = int(self.lib.axi_x2_open())
        if rc != 0:
            raise PlBackendUnavailable(self._last_error())
        self._emit_event(
            "pl_backend_open",
            {
                "lib_path": self.lib_path,
                "source_path": self.source_path,
                "uio_path": self.accel_uio_path,
                "accel_base": self.accel_base,
                "reserved_base": self.reserved_base,
                "alloc_mode": os.environ.get("YOLO_PL_ALLOC", "reserved"),
                "max_rows_per_call": self.max_rows_per_call,
                "max_batch_per_call": self.max_batch_per_call,
                "max_cols_per_call": self.max_cols_per_call,
                "max_output_words_per_call": self.max_output_words_per_call,
                "s2mm_lane_workaround": self.s2mm_lane_workaround,
                "tx_chunk_words_env": os.environ.get("YOLO_PL_TX_CHUNK_WORDS", "full"),
                "accelerator_registers": self._read_accel_regs(),
                "dma_registers": self._read_dma_regs(),
            },
        )
        self.write_summary()

    def _compile_library(self) -> None:
        if not self.source_path.exists():
            raise PlBackendUnavailable(f"PL backend source not found: {self.source_path}")
        cmd = [
            "gcc",
            "-O2",
            "-Wall",
            "-Wextra",
            "-fPIC",
            "-shared",
            str(self.source_path),
            "-o",
            str(self.lib_path),
            "-lcma",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except Exception as exc:
            raise PlBackendUnavailable(f"Could not compile PL backend library with gcc: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise PlBackendUnavailable(f"PL backend library compile failed: {detail}")

    def _activate_environment(self) -> None:
        if self.accel_base is None:
            os.environ.pop("YOLO_PL_ACCEL_BASE", None)
        else:
            os.environ["YOLO_PL_ACCEL_BASE"] = f"0x{self.accel_base:X}"
        if self.reserved_base is not None:
            os.environ["YOLO_PL_RESERVED_BASE"] = f"0x{self.reserved_base:X}"
        if self.accel_uio_path:
            os.environ["YOLO_PL_ACCEL_UIO"] = self.accel_uio_path

    def _last_error(self) -> str:
        raw = self.lib.axi_x2_last_error()
        return raw.decode("utf-8", errors="replace") if raw else "unknown PL backend error"

    def _emit_event(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "event": event,
            "timestamp_s": time.time(),
            "perf_counter_s": time.perf_counter(),
            **payload,
        }
        try:
            with self.events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")
        except OSError:
            pass

    def _read_accel_regs(self) -> dict[str, Any]:
        regs = {
            "ctrl": 0x00,
            "status": 0x04,
            "config": 0x08,
            "expected": 0x0C,
            "rx_words": 0x10,
            "tx_words": 0x14,
            "cycles": 0x18,
            "error": 0x1C,
            "magic": 0x20,
            "version": 0x24,
            "debug_stream0": 0x2C,
            "debug_stream1": 0x30,
            "debug_stream2": 0x34,
            "debug_stream3": 0x38,
            "debug_weight0": 0x3C,
            "debug_weight1": 0x40,
            "debug_vector0": 0x44,
            "debug_vector1": 0x48,
            "debug_result0": 0x4C,
            "debug_result1": 0x50,
            "debug_result2": 0x54,
            "debug_result3": 0x58,
            "debug_axis0": 0x5C,
            "debug_axis1": 0x60,
            "debug_axis2": 0x64,
            "debug_axis3": 0x68,
            "debug_axis_meta": 0x6C,
            "writer_addr_lo": 0x70,
            "writer_addr_hi": 0x74,
            "writer_count": 0x78,
            "writer_status": 0x7C,
            "reader_addr_lo": 0x80,
            "reader_addr_hi": 0x84,
            "reader_count": 0x88,
            "reader_status": 0x8C,
        }
        try:
            self._activate_environment()
            if self.accel_base is None:
                fd = os.open(self.accel_uio_path, os.O_RDWR | os.O_SYNC)
                offset = 0
            else:
                fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
                offset = self.accel_base
            try:
                with mmap.mmap(fd, 0x1000, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=offset) as mm:
                    return {
                        name: int(struct.unpack_from("<I", mm, offset)[0])
                        for name, offset in regs.items()
                    }
            finally:
                os.close(fd)
        except OSError as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _read_dma_regs(self) -> dict[str, Any]:
        regs = {
            "mm2s_dmacr": 0x00,
            "mm2s_dmasr": 0x04,
            "s2mm_dmacr": 0x30,
            "s2mm_dmasr": 0x34,
        }
        try:
            fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
            try:
                with mmap.mmap(fd, 0x10000, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=0xA0010000) as mm:
                    return {
                        name: int(struct.unpack_from("<I", mm, offset)[0])
                        for name, offset in regs.items()
                    }
            finally:
                os.close(fd)
        except OSError as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _read_c_stats(self) -> dict[str, Any]:
        if not self._has_c_stats:
            return {}
        self._activate_environment()
        status = ctypes.c_uint32()
        expected = ctypes.c_uint32()
        rx_words = ctypes.c_uint32()
        tx_words = ctypes.c_uint32()
        cycles = ctypes.c_uint32()
        error_code = ctypes.c_uint32()
        rc = int(
            self.lib.axi_x2_get_stats(
                ctypes.byref(status),
                ctypes.byref(expected),
                ctypes.byref(rx_words),
                ctypes.byref(tx_words),
                ctypes.byref(cycles),
                ctypes.byref(error_code),
            )
        )
        if rc != 0:
            return {"error": self._last_error()}
        return {
            "status": int(status.value),
            "expected": int(expected.value),
            "rx_words": int(rx_words.value),
            "tx_words": int(tx_words.value),
            "cycles": int(cycles.value),
            "error": int(error_code.value),
        }

    def _transaction_words(self, batch: int, rows: int, cols: int) -> tuple[int, int]:
        vector_words = math.ceil(int(rows) / 4.0)
        weight_words = math.ceil((int(rows) * int(cols)) / 4.0)
        tx_words = weight_words + (int(batch) * vector_words)
        rx_words = int(batch) * int(cols)
        return tx_words, rx_words

    def _maybe_emit_transaction_progress(self) -> None:
        if not self._evidence_interval:
            return
        if self.stats.dma_transactions - self._last_evidence_transaction < self._evidence_interval:
            return
        self._last_evidence_transaction = self.stats.dma_transactions
        self._emit_event(
            "pl_dma_transaction_progress",
            {
                "active_layer": self.active_layer.get("layer_name") if self.active_layer else "",
                "stats": asdict(self.stats),
                "accelerator_registers": self._read_accel_regs(),
                "c_stats": self._read_c_stats(),
            },
        )
        self.write_summary()

    def _should_emit_transaction_detail(self, transaction_index: int) -> bool:
        if transaction_index <= self._evidence_first_transactions:
            return True
        return bool(self._evidence_interval and transaction_index % self._evidence_interval == 0)

    def _stats_delta(self, start: PlMvmStats) -> dict[str, Any]:
        current = asdict(self.stats)
        baseline = asdict(start)
        return {key: current[key] - baseline[key] for key in current}

    def write_summary(self) -> None:
        accelerator_registers = self._read_accel_regs()
        version = int(accelerator_registers.get("version", 0) or 0)
        if version >= 0x00020013:
            datapath = [
                "PS Python runtime",
                "ctypes libaxi_x2_mvm_cma.so",
                f"AXI-Lite/MMIO accelerator registers via {self.accel_uio_path}",
                "reserved low-DDR input words at 16-byte stride",
                "PL AXI single-beat input reader",
                "PL axi_x2_mvm_accel PE/MVM engine",
                "PL AXI result writer to contiguous DDR output",
            ]
            input_transport = "pl_axi_reader_stride16"
        else:
            datapath = [
                "PS Python runtime",
                "ctypes libaxi_x2_mvm_cma.so",
                f"AXI-Lite/MMIO accelerator registers via {self.accel_uio_path}",
                "/dev/mem AXI DMA registers at 0xA0010000",
                "reserved low-DDR DMA buffers",
                "PL axi_x2_mvm_accel PE/MVM engine",
            ]
            input_transport = "axi_dma_mm2s"
        summary = {
            "backend": "pl_dma",
            "input_transport": input_transport,
            "datapath": datapath,
            "proof": {
                "software_crossbar_fallback_enabled": False,
                "magic_expected_hex": "0x58324d56",
                "pl_device_files": [self.accel_uio_path, "/dev/mem"],
            },
            "lib_path": str(self.lib_path),
            "stats": asdict(self.stats),
            "limits": {
                "max_rows_per_call": self.max_rows_per_call,
                "max_batch_per_call": self.max_batch_per_call,
                "max_cols_per_call": self.max_cols_per_call,
                "max_output_words_per_call": self.max_output_words_per_call,
                "s2mm_lane_workaround": self.s2mm_lane_workaround,
                "tx_chunk_words_env": os.environ.get("YOLO_PL_TX_CHUNK_WORDS", "full"),
            },
            "active_layer": self.active_layer,
            "layer_count": len(self.layer_records),
            "layers": self.layer_records,
            "accelerator_registers": accelerator_registers,
            "dma_registers": self._read_dma_regs(),
            "c_stats": self._read_c_stats(),
            "events_path": str(self.events_path),
            "report_path": str(self.report_path),
        }
        try:
            self.summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
            self.report_path.write_text(self._format_report(summary), encoding="utf-8")
        except OSError:
            pass

    def _format_report(self, summary: dict[str, Any]) -> str:
        regs = summary.get("accelerator_registers", {})
        stats = summary.get("stats", {})
        active = summary.get("active_layer") or {}
        lines = [
            "# PL-DMA Crossbar Datapath Evidence",
            "",
            "This file is generated by the board-side YOLO run. It is separate from the Streamlit UI.",
            "",
            "## Datapath",
            "",
            " -> ".join(summary.get("datapath", [])) + " -> PS graph continuation and YOLO postprocess.",
            "",
            "## Proof Points",
            "",
            f"- PL backend: `{summary.get('backend')}`",
            f"- Software crossbar fallback enabled: `{summary.get('proof', {}).get('software_crossbar_fallback_enabled')}`",
            f"- Accelerator magic: `0x{int(regs.get('magic', 0)):08x}`",
            f"- Accelerator version: `0x{int(regs.get('version', 0)):08x}`",
            f"- Accelerator error register: `0x{int(regs.get('error', 0)):08x}`",
            f"- Input transport: `{summary.get('input_transport', '')}`",
            f"- PL reader address: `0x{int(regs.get('reader_addr_lo', 0)):08x}`",
            f"- PL reader words: `{int(regs.get('reader_count', 0))}`",
            f"- PL reader status: `0x{int(regs.get('reader_status', 0)):08x}`",
            f"- PE result debug: `0x{int(regs.get('debug_result0', 0)):08x}` "
            f"`0x{int(regs.get('debug_result1', 0)):08x}` "
            f"`0x{int(regs.get('debug_result2', 0)):08x}` "
            f"`0x{int(regs.get('debug_result3', 0)):08x}`",
            f"- Accepted AXIS debug: `0x{int(regs.get('debug_axis0', 0)):08x}` "
            f"`0x{int(regs.get('debug_axis1', 0)):08x}` "
            f"`0x{int(regs.get('debug_axis2', 0)):08x}` "
            f"`0x{int(regs.get('debug_axis3', 0)):08x}` "
            f"meta `0x{int(regs.get('debug_axis_meta', 0)):08x}`",
            f"- DMA transactions issued: `{stats.get('dma_transactions', 0)}`",
            f"- Estimated MACs sent to PL: `{stats.get('estimated_macs', 0)}`",
            f"- Events log: `{self.events_path}`",
            "",
        ]
        if active:
            lines.extend(
                [
                    "## Active Layer",
                    "",
                    f"- Layer: `{active.get('layer_name')}`",
                    f"- Input dtype: `{active.get('input_dtype')}`",
                    f"- Vectors: `{active.get('num_vectors')}`",
                    f"- K: `{active.get('k_total')}`",
                    f"- Output channels: `{active.get('out_channels')}`",
                    f"- Planned primary DMA transactions: `{active.get('planned_primary_dma_transactions')}`",
                    "",
                ]
            )
        lines.extend(["## Completed Layers", "", "| Layer | dtype | vectors | K | C | primary tx | correction tx | elapsed s |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for layer in summary.get("layers", []):
            delta = layer.get("stats_delta", {})
            lines.append(
                "| {layer} | {dtype} | {vectors} | {k} | {channels} | {primary} | {correction} | {elapsed:.3f} |".format(
                    layer=layer.get("layer_name", ""),
                    dtype=layer.get("input_dtype", ""),
                    vectors=layer.get("num_vectors", 0),
                    k=layer.get("k_total", 0),
                    channels=layer.get("out_channels", 0),
                    primary=delta.get("primary_dma_transactions", 0),
                    correction=delta.get("correction_dma_transactions", 0),
                    elapsed=float(layer.get("elapsed_s", 0.0) or 0.0),
                )
            )
        lines.append("")
        return "\n".join(lines)

    def close(self) -> None:
        if hasattr(self, "lib"):
            self.lib.axi_x2_close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _run_mvm_i8_raw(self, vector: np.ndarray, weights: np.ndarray) -> np.ndarray:
        vec = np.ascontiguousarray(vector, dtype=np.int8).reshape(-1)
        mat = np.ascontiguousarray(weights, dtype=np.int8)
        if mat.ndim != 2:
            raise ValueError(f"weights must be rank 2, got {mat.shape}")
        rows, cols = int(mat.shape[0]), int(mat.shape[1])
        if vec.shape[0] != rows:
            raise ValueError(f"vector length {vec.shape[0]} does not match weight rows {rows}")
        if not (1 <= rows <= 32 and 1 <= cols <= 32):
            raise ValueError(f"PL MVM supports rows/cols 1..32, got rows={rows} cols={cols}")

        out = np.zeros((cols,), dtype=np.int32)
        self._activate_environment()
        rc = int(
            self.lib.axi_x2_run_i8(
                vec.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                mat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                ctypes.c_uint32(rows),
                ctypes.c_uint32(cols),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            )
        )
        if rc != 0:
            raise RuntimeError(self._last_error())
        self.stats.mvm_calls += 1
        return out

    def _run_mvm_i8_batch_raw(self, vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
        vecs = np.ascontiguousarray(vectors, dtype=np.int8)
        mat = np.ascontiguousarray(weights, dtype=np.int8)
        if vecs.ndim != 2:
            raise ValueError(f"vectors must be rank 2, got {vecs.shape}")
        if mat.ndim != 2:
            raise ValueError(f"weights must be rank 2, got {mat.shape}")
        batch, rows = int(vecs.shape[0]), int(vecs.shape[1])
        cols = int(mat.shape[1])
        if mat.shape[0] != rows:
            raise ValueError(f"vector width {rows} does not match weight rows {mat.shape[0]}")
        if not (1 <= batch <= 32 and 1 <= rows <= 32 and 1 <= cols <= 32):
            raise ValueError(f"PL batch MVM supports batch/rows/cols 1..32, got batch={batch} rows={rows} cols={cols}")
        if batch * cols > self.max_output_words_per_call:
            raise ValueError(
                f"Raw PL call would return {batch * cols} words; current DMA backend limit is "
                f"{self.max_output_words_per_call}. Use run_mvm_i8_batch tiling."
            )

        out = np.zeros((batch, cols), dtype=np.int32)
        tx_words, rx_words = self._transaction_words(batch, rows, cols)
        transaction_index = self.stats.dma_transactions + 1
        tx_record = {
            "transaction_index": transaction_index,
            "role": self._transaction_role,
            "active_layer": self.active_layer.get("layer_name") if self.active_layer else "",
            "batch": batch,
            "rows": rows,
            "cols": cols,
            "tx_words": tx_words,
            "rx_words": rx_words,
            "estimated_macs": batch * rows * cols,
            "accelerator_registers_before": self._read_accel_regs(),
            "c_stats_before": self._read_c_stats(),
        }
        if self.active_layer is not None:
            self.active_layer["inflight_transaction"] = tx_record
        if self._should_emit_transaction_detail(transaction_index):
            self._emit_event("pl_dma_transaction_start", tx_record)
            self.write_summary()
        self._activate_environment()
        rc = int(
            self.lib.axi_x2_run_i8_batch(
                vecs.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                mat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                ctypes.c_uint32(batch),
                ctypes.c_uint32(rows),
                ctypes.c_uint32(cols),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            )
        )
        if rc != 0:
            raise RuntimeError(self._last_error())
        self.stats.mvm_calls += batch
        self.stats.dma_transactions += 1
        self.stats.dma_tx_words += tx_words
        self.stats.dma_rx_words += rx_words
        self.stats.estimated_macs += batch * rows * cols
        if self._transaction_role == "correction":
            self.stats.correction_dma_transactions += 1
        else:
            self.stats.primary_dma_transactions += 1
        if self.active_layer is not None:
            complete_record = {
                **tx_record,
                "accelerator_registers_after": self._read_accel_regs(),
                "c_stats_after": self._read_c_stats(),
            }
            self.active_layer["last_completed_transaction"] = complete_record
            if self.active_layer.get("inflight_transaction", {}).get("transaction_index") == transaction_index:
                self.active_layer.pop("inflight_transaction", None)
            if self._should_emit_transaction_detail(transaction_index):
                self._emit_event("pl_dma_transaction_complete", complete_record)
                self.write_summary()
        self._maybe_emit_transaction_progress()
        return out

    def run_mvm_i8(self, vector: np.ndarray, weights: np.ndarray) -> np.ndarray:
        vec = np.ascontiguousarray(vector, dtype=np.int8).reshape(-1)
        mat = np.ascontiguousarray(weights, dtype=np.int8)
        if mat.ndim != 2:
            raise ValueError(f"weights must be rank 2, got {mat.shape}")
        if vec.shape[0] != mat.shape[0]:
            raise ValueError(f"vector length {vec.shape[0]} does not match weight rows {mat.shape[0]}")
        return self.run_mvm_i8_batch(vec.reshape(1, -1), mat)[0]

    def _run_mvm_i8_batch_s2mm_workaround(self, vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Work around the live ZCU106 S2MM lane issue.

        Full-width S2MM writes keep words 2 and 3 from each 4-word group and
        drop words 0 and 1. Put real columns in physical lanes 2/3, 6/7, ...
        and gather the written lanes back into a dense logical result.
        """
        vecs = np.ascontiguousarray(vectors, dtype=np.int8)
        mat = np.ascontiguousarray(weights, dtype=np.int8)
        out = np.zeros((vecs.shape[0], mat.shape[1]), dtype=np.int32)
        logical_col_step = max(1, min(self.max_cols_per_call, 16))
        for row0 in range(0, mat.shape[0], self.max_rows_per_call):
            row1 = min(row0 + self.max_rows_per_call, mat.shape[0])
            for c0 in range(0, mat.shape[1], logical_col_step):
                c1 = min(c0 + logical_col_step, mat.shape[1])
                logical_cols = c1 - c0
                groups = (logical_cols + 1) // 2
                physical_cols = groups * 4
                if physical_cols > 32:
                    raise ValueError(f"Internal S2MM workaround produced too many physical columns: {physical_cols}")

                padded = np.zeros((row1 - row0, physical_cols), dtype=np.int8)
                for group in range(groups):
                    src0 = c0 + (group * 2)
                    if src0 < c1:
                        padded[:, (group * 4) + 2] = mat[row0:row1, src0]
                    if src0 + 1 < c1:
                        padded[:, (group * 4) + 3] = mat[row0:row1, src0 + 1]

                batch_step = max(1, min(self.max_batch_per_call, self.max_output_words_per_call // physical_cols))
                for batch0 in range(0, vecs.shape[0], batch_step):
                    batch1 = min(batch0 + batch_step, vecs.shape[0])
                    raw = self._run_mvm_i8_batch_raw(vecs[batch0:batch1, row0:row1], padded)
                    gathered_cols: list[np.ndarray] = []
                    for group in range(groups):
                        gathered_cols.append(raw[:, group * 4])
                        if len(gathered_cols) < logical_cols:
                            gathered_cols.append(raw[:, (group * 4) + 1])
                    out[batch0:batch1, c0:c1] += np.stack(gathered_cols[:logical_cols], axis=1).astype(
                        np.int32,
                        copy=False,
                    )
        return out

    def run_mvm_i8_batch(self, vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
        vecs = np.ascontiguousarray(vectors, dtype=np.int8)
        mat = np.ascontiguousarray(weights, dtype=np.int8)
        if vecs.ndim != 2:
            raise ValueError(f"vectors must be rank 2, got {vecs.shape}")
        if mat.ndim != 2:
            raise ValueError(f"weights must be rank 2, got {mat.shape}")
        if vecs.shape[1] != mat.shape[0]:
            raise ValueError(f"vector width {vecs.shape[1]} does not match weight rows {mat.shape[0]}")

        if self.s2mm_lane_workaround:
            return self._run_mvm_i8_batch_s2mm_workaround(vecs, mat)

        out = np.zeros((vecs.shape[0], mat.shape[1]), dtype=np.int32)
        col_step = max(1, min(self.max_cols_per_call, self.max_output_words_per_call))
        for c0 in range(0, mat.shape[1], col_step):
            c1 = min(c0 + col_step, mat.shape[1])
            col_count = max(1, c1 - c0)
            batch_step = max(1, min(self.max_batch_per_call, self.max_output_words_per_call // col_count))
            for batch0 in range(0, vecs.shape[0], batch_step):
                batch1 = min(batch0 + batch_step, vecs.shape[0])
                for row0 in range(0, mat.shape[0], self.max_rows_per_call):
                    row1 = min(row0 + self.max_rows_per_call, mat.shape[0])
                    out[batch0:batch1, c0:c1] += self._run_mvm_i8_batch_raw(
                        vecs[batch0:batch1, row0:row1],
                        mat[row0:row1, c0:c1],
                    )
        return out

    def run_mvm_u8_i8(self, vector_u8: np.ndarray, weights_i8: np.ndarray) -> np.ndarray:
        """
        Compute uint8 activations times signed int8 weights using signed INT8 PL MVMs.

        Values >= 128 are sent as their wrapped int8 byte value, then corrected
        with a second PL MVM over a 0/1 high-byte mask.
        """
        vec_u8 = np.ascontiguousarray(vector_u8, dtype=np.uint8).reshape(-1)
        mat_i8 = np.ascontiguousarray(weights_i8, dtype=np.int8)
        previous_role = self._transaction_role
        self._transaction_role = "primary"
        try:
            result = self.run_mvm_i8(vec_u8.view(np.int8), mat_i8)
            high = vec_u8 >= 128
            if bool(np.any(high)):
                self._transaction_role = "correction"
                correction = self.run_mvm_i8(high.astype(np.int8), mat_i8)
                self.stats.correction_calls += 1
                result = result + (correction * np.int32(256))
        finally:
            self._transaction_role = previous_role
        return result.astype(np.int32, copy=False)

    def run_mvm_u8_i8_batch(self, vectors_u8: np.ndarray, weights_i8: np.ndarray) -> np.ndarray:
        vecs_u8 = np.ascontiguousarray(vectors_u8, dtype=np.uint8)
        mat_i8 = np.ascontiguousarray(weights_i8, dtype=np.int8)
        if vecs_u8.ndim != 2:
            raise ValueError(f"vectors_u8 must be rank 2, got {vecs_u8.shape}")
        previous_role = self._transaction_role
        self._transaction_role = "primary"
        try:
            result = self.run_mvm_i8_batch(vecs_u8.view(np.int8), mat_i8)
            high = vecs_u8 >= 128
            if bool(np.any(high)):
                self._transaction_role = "correction"
                correction = self.run_mvm_i8_batch(high.astype(np.int8), mat_i8)
                self.stats.correction_calls += int(vecs_u8.shape[0])
                result = result + (correction * np.int32(256))
        finally:
            self._transaction_role = previous_role
        return result.astype(np.int32, copy=False)

    def _start_layer_record(
        self,
        *,
        layer_name: str,
        input_dtype: str,
        num_positions: int,
        k_total: int,
        out_channels: int,
        weight_tiles_per_vector: int,
        position_batches: int,
        planned_primary_dma_transactions: int,
    ) -> tuple[dict[str, Any], PlMvmStats]:
        start_stats = PlMvmStats(**asdict(self.stats))
        record = {
            "layer_name": layer_name,
            "input_dtype": input_dtype,
            "num_vectors": num_positions,
            "k_total": k_total,
            "out_channels": out_channels,
            "max_rows_per_call": self.max_rows_per_call,
            "max_batch_per_call": self.max_batch_per_call,
            "weight_tiles_per_vector": weight_tiles_per_vector,
            "position_batches": position_batches,
            "planned_primary_dma_transactions": planned_primary_dma_transactions,
            "planned_macs": int(num_positions) * int(k_total) * int(out_channels),
            "start_perf_counter_s": time.perf_counter(),
            "start_accelerator_registers": self._read_accel_regs(),
            "start_c_stats": self._read_c_stats(),
        }
        self.active_layer = record
        self._emit_event("pl_layer_start", record)
        self.write_summary()
        return record, start_stats

    def _finish_layer_record(self, record: dict[str, Any], start_stats: PlMvmStats, *, error: str = "") -> None:
        record["elapsed_s"] = time.perf_counter() - float(record["start_perf_counter_s"])
        record["stats_delta"] = self._stats_delta(start_stats)
        record["end_accelerator_registers"] = self._read_accel_regs()
        record["end_c_stats"] = self._read_c_stats()
        if error:
            record["error"] = error
            self._emit_event("pl_layer_error", record)
        else:
            self.layer_records.append(dict(record))
            self._emit_event("pl_layer_complete", record)
        self.active_layer = None
        self.write_summary()

    def matmul_u8_i8(
        self,
        activations_u8: np.ndarray,
        weights_i8: np.ndarray,
        *,
        layer_name: str = "layer",
    ) -> np.ndarray:
        acts = np.ascontiguousarray(activations_u8, dtype=np.uint8)
        weights = np.ascontiguousarray(weights_i8, dtype=np.int8)
        if acts.ndim != 2 or weights.ndim != 2:
            raise ValueError(f"Expected rank-2 activations and weights, got {acts.shape} and {weights.shape}")
        if acts.shape[1] != weights.shape[0]:
            raise ValueError(f"Activation width {acts.shape[1]} does not match weight rows {weights.shape[0]}")

        num_positions, k_total = int(acts.shape[0]), int(acts.shape[1])
        out_channels = int(weights.shape[1])
        out = np.zeros((num_positions, out_channels), dtype=np.int32)
        progress_interval = max(0, int(os.environ.get("YOLO_PL_PROGRESS_INTERVAL", "256") or "0"))
        started = time.perf_counter()

        weight_tiles: list[tuple[int, int, int, int, np.ndarray]] = []
        row_tile = self.max_rows_per_call
        col_tile = max(1, min(self.max_cols_per_call, self.max_output_words_per_call))
        for k0 in range(0, k_total, row_tile):
            k1 = min(k0 + row_tile, k_total)
            for c0 in range(0, out_channels, col_tile):
                c1 = min(c0 + col_tile, out_channels)
                weight_tiles.append((k0, k1, c0, c1, np.ascontiguousarray(weights[k0:k1, c0:c1], dtype=np.int8)))

        position_batches = (num_positions + self.max_batch_per_call - 1) // self.max_batch_per_call
        total_tile_calls = position_batches * len(weight_tiles)
        _safe_print(
            f"[PL-DMA] {layer_name}: {num_positions:,} vectors, K={k_total}, C={out_channels}, "
            f"{len(weight_tiles)} MVM tiles/vector, batch={self.max_batch_per_call} "
            f"({total_tile_calls:,} primary DMA transactions)",
            flush=True,
        )

        completed = 0
        record, start_stats = self._start_layer_record(
            layer_name=layer_name,
            input_dtype="uint8",
            num_positions=num_positions,
            k_total=k_total,
            out_channels=out_channels,
            weight_tiles_per_vector=len(weight_tiles),
            position_batches=position_batches,
            planned_primary_dma_transactions=total_tile_calls,
        )
        try:
            for pos0 in range(0, num_positions, self.max_batch_per_call):
                pos1 = min(pos0 + self.max_batch_per_call, num_positions)
                vec_batch = acts[pos0:pos1]
                for k0, k1, c0, c1, tile in weight_tiles:
                    out[pos0:pos1, c0:c1] += self.run_mvm_u8_i8_batch(vec_batch[:, k0:k1], tile)
                    completed += 1
                if progress_interval and pos1 % progress_interval == 0:
                    elapsed = time.perf_counter() - started
                    _safe_print(
                        f"[PL-DMA] {layer_name}: {pos1:,}/{num_positions:,} vectors "
                        f"({completed:,}/{total_tile_calls:,} tile calls, {elapsed:.1f}s)",
                        flush=True,
                    )
        except Exception as exc:
            self._finish_layer_record(record, start_stats, error=str(exc))
            raise

        self.stats.elapsed_s += time.perf_counter() - started
        self._finish_layer_record(record, start_stats)
        return out

    def matmul_i8_i8(
        self,
        activations_i8: np.ndarray,
        weights_i8: np.ndarray,
        *,
        layer_name: str = "layer",
    ) -> np.ndarray:
        acts = np.ascontiguousarray(activations_i8, dtype=np.int8)
        weights = np.ascontiguousarray(weights_i8, dtype=np.int8)
        if acts.ndim != 2 or weights.ndim != 2:
            raise ValueError(f"Expected rank-2 activations and weights, got {acts.shape} and {weights.shape}")
        if acts.shape[1] != weights.shape[0]:
            raise ValueError(f"Activation width {acts.shape[1]} does not match weight rows {weights.shape[0]}")

        num_positions, k_total = int(acts.shape[0]), int(acts.shape[1])
        out_channels = int(weights.shape[1])
        out = np.zeros((num_positions, out_channels), dtype=np.int32)
        progress_interval = max(0, int(os.environ.get("YOLO_PL_PROGRESS_INTERVAL", "256") or "0"))
        started = time.perf_counter()

        weight_tiles: list[tuple[int, int, int, int, np.ndarray]] = []
        row_tile = self.max_rows_per_call
        col_tile = max(1, min(self.max_cols_per_call, self.max_output_words_per_call))
        for k0 in range(0, k_total, row_tile):
            k1 = min(k0 + row_tile, k_total)
            for c0 in range(0, out_channels, col_tile):
                c1 = min(c0 + col_tile, out_channels)
                weight_tiles.append((k0, k1, c0, c1, np.ascontiguousarray(weights[k0:k1, c0:c1], dtype=np.int8)))

        position_batches = (num_positions + self.max_batch_per_call - 1) // self.max_batch_per_call
        total_tile_calls = position_batches * len(weight_tiles)
        _safe_print(
            f"[PL-DMA] {layer_name}: {num_positions:,} signed vectors, K={k_total}, C={out_channels}, "
            f"{len(weight_tiles)} MVM tiles/vector, batch={self.max_batch_per_call} "
            f"({total_tile_calls:,} primary DMA transactions)",
            flush=True,
        )

        completed = 0
        record, start_stats = self._start_layer_record(
            layer_name=layer_name,
            input_dtype="int8",
            num_positions=num_positions,
            k_total=k_total,
            out_channels=out_channels,
            weight_tiles_per_vector=len(weight_tiles),
            position_batches=position_batches,
            planned_primary_dma_transactions=total_tile_calls,
        )
        previous_role = self._transaction_role
        self._transaction_role = "primary"
        try:
            for pos0 in range(0, num_positions, self.max_batch_per_call):
                pos1 = min(pos0 + self.max_batch_per_call, num_positions)
                vec_batch = acts[pos0:pos1]
                for k0, k1, c0, c1, tile in weight_tiles:
                    out[pos0:pos1, c0:c1] += self.run_mvm_i8_batch(vec_batch[:, k0:k1], tile)
                    completed += 1
                if progress_interval and pos1 % progress_interval == 0:
                    elapsed = time.perf_counter() - started
                    _safe_print(
                        f"[PL-DMA] {layer_name}: {pos1:,}/{num_positions:,} signed vectors "
                        f"({completed:,}/{total_tile_calls:,} tile calls, {elapsed:.1f}s)",
                        flush=True,
                    )
        except Exception as exc:
            self._finish_layer_record(record, start_stats, error=str(exc))
            raise
        finally:
            self._transaction_role = previous_role

        self.stats.elapsed_s += time.perf_counter() - started
        self._finish_layer_record(record, start_stats)
        return out
