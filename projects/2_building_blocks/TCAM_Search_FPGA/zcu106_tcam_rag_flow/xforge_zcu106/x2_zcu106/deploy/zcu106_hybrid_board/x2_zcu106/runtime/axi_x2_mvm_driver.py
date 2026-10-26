from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from numpy.typing import NDArray


REG_CTRL = 0x00
REG_STATUS = 0x04
REG_CONFIG = 0x08
REG_EXPECTED = 0x0C
REG_RX_WORDS = 0x10
REG_TX_WORDS = 0x14
REG_CYCLES = 0x18
REG_ERROR = 0x1C
REG_MAGIC = 0x20
REG_VERSION = 0x24
REG_WR_ADDR_LO = 0x70
REG_WR_ADDR_HI = 0x74
REG_WR_COUNT = 0x78
REG_WR_STATUS = 0x7C
REG_RD_ADDR_LO = 0x80
REG_RD_ADDR_HI = 0x84
REG_RD_COUNT = 0x88
REG_RD_STATUS = 0x8C
REG_TDC_ORDER = 0x90
REG_TDC_ARRIVAL_MIN = 0x94
REG_TDC_ARRIVAL_MAX = 0x98
REG_TDC_ARRIVAL_SPREAD = 0x9C
REG_TDC_PE_COMPLETION = 0xA0
REG_TDC_CONFIG = 0xA4
REG_TDC_TIMING_MIN = 0xA8
REG_TDC_TIMING_MAX = 0xAC

CTRL_START = 1 << 0
CTRL_ABORT = 1 << 1
CTRL_IRQ_ENABLE = 1 << 2
CTRL_CLEAR = 1 << 8

STATUS_IDLE = 1 << 0
STATUS_DONE = 1 << 1
STATUS_ERROR = 1 << 2

MAGIC_X2MV = 0x58324D56
MIN_CONTINUOUS_TDC_VERSION = 0x00020022
MIN_DUAL_PE_VERSION = 0x00020024


def _as_i8_vector(vector: NDArray[np.integer]) -> NDArray[np.int8]:
    arr = np.asarray(vector)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1-D vector, got shape {arr.shape}")
    if not 1 <= arr.shape[0] <= 32:
        raise ValueError(f"Vector length must be 1..32, got {arr.shape[0]}")
    return arr.astype(np.int8, copy=False)


def _as_i8_batch(vectors: NDArray[np.integer], *, max_batch: int = 32) -> NDArray[np.int8]:
    arr = np.asarray(vectors)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 1-D vector or 2-D batch, got shape {arr.shape}")
    if not 1 <= arr.shape[0] <= max_batch:
        raise ValueError(f"Batch size must be 1..{max_batch}, got {arr.shape[0]}")
    if not 1 <= arr.shape[1] <= 32:
        raise ValueError(f"Vector length must be 1..32, got {arr.shape[1]}")
    return arr.astype(np.int8, copy=False)


def _as_i8_matrix(weights: NDArray[np.integer], rows: int) -> NDArray[np.int8]:
    arr = np.asarray(weights)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D weight matrix, got shape {arr.shape}")
    if arr.shape[0] != rows:
        raise ValueError(f"Weight rows {arr.shape[0]} must match vector rows {rows}")
    if not 1 <= arr.shape[1] <= 32:
        raise ValueError(f"Weight columns must be 1..32, got {arr.shape[1]}")
    return arr.astype(np.int8, copy=False)


def _pack_i8_words(values: NDArray[np.int8]) -> NDArray[np.uint32]:
    flat = np.asarray(values, dtype=np.int8).reshape(-1)
    padded_len = ((flat.size + 3) // 4) * 4
    padded = np.zeros(padded_len, dtype=np.uint8)
    padded[: flat.size] = flat.view(np.uint8)
    return np.frombuffer(padded.tobytes(), dtype="<u4").astype(np.uint32, copy=False)


def pack_mvm_stream(vector: NDArray[np.integer], weights: NDArray[np.integer]) -> NDArray[np.uint32]:
    return pack_mvm_batch_stream(_as_i8_vector(vector).reshape(1, -1), weights)


def pack_mvm_batch_stream(
    vectors: NDArray[np.integer],
    weights: NDArray[np.integer],
) -> NDArray[np.uint32]:
    """
    Pack the exact stream consumed by ``x2_mvm_axis_core``.

    Each vector is padded independently to a 32-bit word boundary. This is
    important when ``rows`` is not a multiple of four: padding may not be
    shared between adjacent batch elements.
    """

    batch = _as_i8_batch(vectors)
    mat = _as_i8_matrix(weights, rows=batch.shape[1])
    parts = [_pack_i8_words(mat.reshape(-1))]
    parts.extend(_pack_i8_words(vector) for vector in batch)
    return np.concatenate(parts)


@dataclass(frozen=True)
class AxiX2MvmStats:
    status: int
    expected_words: int
    rx_words: int
    tx_words: int
    cycles: int
    error_code: int
    tdc_model_enabled: bool
    tdc_reference_levels: int
    tdc_timing_min_ps: int
    tdc_timing_max_ps: int
    tdc_timing_valid: bool
    tdc_first_ready_column: int
    tdc_last_ready_column: int
    tdc_arrival_min_ps: int
    tdc_arrival_max_ps: int
    tdc_arrival_spread_ps: int
    tdc_pe_completion_time_ps: int


@dataclass
class AxiX2MvmJob:
    tx: object
    rx: object
    tx_word_count: int
    output_words: int
    batch: int
    cols: int
    started: bool = False
    deadline: float = 0.0
    released: bool = False


class AxiX2MvmPynq:
    """
    PS-side PYNQ driver for zcu106_axi_x2_mvm.bit.

    The PL block performs one signed INT8 matrix-vector multiply:
    vector[rows] @ weights[rows, cols] -> int32[cols].
    """

    def __init__(
        self,
        bit_path: str | Path | None = None,
        *,
        overlay=None,
        dma_name: str = "axi_dma_mvm",
        accel_name: str = "axi_x2_mvm",
        download: bool = True,
        timeout_s: float = 5.0,
    ) -> None:
        if overlay is None:
            if bit_path is None:
                raise ValueError("Provide bit_path or an existing PYNQ Overlay")
            from pynq import Overlay  # type: ignore

            overlay = Overlay(str(Path(bit_path)), download=download)
        from pynq import MMIO, allocate  # type: ignore

        self.overlay = overlay
        self.allocate = allocate
        # Kept for compatibility with overlays that still expose the old DMA
        # object. Version 0x00020022 and newer use the corrected custom AXI
        # reader/writer and do not use the DMA receive channel.
        self.dma = getattr(overlay, dma_name, None)
        self.accel_name = accel_name
        self.timeout_s = float(timeout_s)

        ip_info = overlay.ip_dict.get(accel_name)
        if ip_info is None:
            matches = [name for name in overlay.ip_dict if name.endswith("/" + accel_name) or name == accel_name]
            if not matches:
                raise KeyError(f"Could not find accelerator IP '{accel_name}' in overlay.ip_dict")
            ip_info = overlay.ip_dict[matches[0]]
        self.mmio = MMIO(ip_info["phys_addr"], ip_info["addr_range"])

        magic = self.read_reg(REG_MAGIC)
        if magic != MAGIC_X2MV:
            raise RuntimeError(f"Unexpected accelerator magic 0x{magic:08x}; expected 0x{MAGIC_X2MV:08x}")
        self.version = self.read_reg(REG_VERSION)
        if self.version < MIN_CONTINUOUS_TDC_VERSION:
            raise RuntimeError(
                f"Accelerator version 0x{self.version:08x} predates the corrected 128-bit DDR reader; "
                f"version 0x{MIN_CONTINUOUS_TDC_VERSION:08x} or newer is required."
            )

    def read_reg(self, offset: int) -> int:
        return int(self.mmio.read(int(offset)))

    def write_reg(self, offset: int, value: int) -> None:
        self.mmio.write(int(offset), int(value) & 0xFFFFFFFF)

    def stats(self) -> AxiX2MvmStats:
        tdc_order = self.read_reg(REG_TDC_ORDER)
        tdc_config = self.read_reg(REG_TDC_CONFIG)
        return AxiX2MvmStats(
            status=self.read_reg(REG_STATUS),
            expected_words=self.read_reg(REG_EXPECTED),
            rx_words=self.read_reg(REG_RX_WORDS),
            tx_words=self.read_reg(REG_TX_WORDS),
            cycles=self.read_reg(REG_CYCLES),
            error_code=self.read_reg(REG_ERROR),
            tdc_model_enabled=bool(tdc_config & (1 << 31)),
            tdc_reference_levels=tdc_config & 0xFFFF,
            tdc_timing_min_ps=self.read_reg(REG_TDC_TIMING_MIN),
            tdc_timing_max_ps=self.read_reg(REG_TDC_TIMING_MAX),
            tdc_timing_valid=bool(tdc_order & (1 << 31)),
            tdc_first_ready_column=tdc_order & 0xFF,
            tdc_last_ready_column=(tdc_order >> 8) & 0xFF,
            tdc_arrival_min_ps=self.read_reg(REG_TDC_ARRIVAL_MIN),
            tdc_arrival_max_ps=self.read_reg(REG_TDC_ARRIVAL_MAX),
            tdc_arrival_spread_ps=self.read_reg(REG_TDC_ARRIVAL_SPREAD),
            tdc_pe_completion_time_ps=self.read_reg(REG_TDC_PE_COMPLETION),
        )

    def run_mvm(self, vector: NDArray[np.integer], weights: NDArray[np.integer]) -> NDArray[np.int32]:
        return self.run_mvm_batch(_as_i8_vector(vector).reshape(1, -1), weights)[0]

    def run_mvm_batch(
        self,
        vectors: NDArray[np.integer],
        weights: NDArray[np.integer],
    ) -> NDArray[np.int32]:
        """
        Execute up to 32 vectors against one weight tile.

        The current RTL reads one packed 32-bit input word from every 16-byte
        DDR line and writes four INT32 results per 16-byte DDR line. PYNQ
        buffers are therefore laid out to match those physical transactions.
        """

        job = self.prepare_mvm_batch(vectors, weights)
        try:
            self.start_job(job)
            return self.wait_job(job)
        except Exception:
            self.release_job(job, abort=job.started)
            raise

    def prepare_mvm_batch(
        self,
        vectors: NDArray[np.integer],
        weights: NDArray[np.integer],
    ) -> AxiX2MvmJob:
        """Allocate buffers and configure one PE without starting it."""

        batch_vectors = _as_i8_batch(vectors)
        mat = _as_i8_matrix(weights, rows=batch_vectors.shape[1])
        batch, rows = batch_vectors.shape
        cols = mat.shape[1]
        tx_words = pack_mvm_batch_stream(batch_vectors, mat)
        output_words = batch * cols
        output_lines = (output_words + 3) // 4

        tx = self.allocate(shape=(tx_words.shape[0], 4), dtype=np.uint32)
        rx = self.allocate(shape=(output_lines, 4), dtype=np.uint32)
        job = AxiX2MvmJob(
            tx=tx,
            rx=rx,
            tx_word_count=int(tx_words.shape[0]),
            output_words=output_words,
            batch=batch,
            cols=cols,
        )
        try:
            tx[:] = 0
            tx[:, 0] = tx_words
            rx[:] = 0
            tx.flush()
            rx.flush()

            tx_address = int(tx.physical_address)
            rx_address = int(rx.physical_address)
            if (tx_address >> 32) or (rx_address >> 32):
                raise RuntimeError(
                    "The current RTL exposes 32-bit physical address registers; "
                    f"got tx=0x{tx_address:x}, rx=0x{rx_address:x}"
                )
            if (tx_address & 0xF) or (rx_address & 0xF):
                raise RuntimeError(
                    "X2 input/output buffers must be 16-byte aligned; "
                    f"got tx=0x{tx_address:x}, rx=0x{rx_address:x}"
                )

            self.write_reg(REG_CTRL, CTRL_ABORT | CTRL_CLEAR)
            self.write_reg(REG_CONFIG, rows | (cols << 8) | (batch << 16))
            self.write_reg(REG_RD_ADDR_LO, tx_address)
            self.write_reg(REG_RD_ADDR_HI, 0)
            self.write_reg(REG_WR_ADDR_LO, rx_address)
            self.write_reg(REG_WR_ADDR_HI, 0)
            return job
        except Exception:
            self.release_job(job)
            raise

    def start_job(self, job: AxiX2MvmJob) -> None:
        """Start a prepared job. Multiple PEs may be started before waiting."""

        if job.released:
            raise RuntimeError("Cannot start a released X2 MVM job")
        if job.started:
            raise RuntimeError("X2 MVM job has already been started")
        self.write_reg(REG_CTRL, CTRL_START)
        job.started = True
        job.deadline = time.monotonic() + self.timeout_s

    def wait_job(self, job: AxiX2MvmJob) -> NDArray[np.int32]:
        """Wait for a started job, validate counters, and return its results."""

        if job.released:
            raise RuntimeError("Cannot wait for a released X2 MVM job")
        if not job.started:
            raise RuntimeError("X2 MVM job has not been started")
        try:
            while True:
                status = self.read_reg(REG_STATUS)
                reader_count = self.read_reg(REG_RD_COUNT)
                writer_count = self.read_reg(REG_WR_COUNT)
                if status & STATUS_ERROR:
                    raise RuntimeError(
                        f"Accelerator error_code={self.read_reg(REG_ERROR)}, stats={self.stats()}"
                    )
                if (
                    status & STATUS_DONE
                    and reader_count == job.tx_word_count
                    and writer_count == job.output_words
                ):
                    break
                if time.monotonic() >= job.deadline:
                    raise TimeoutError(
                        f"X2 MVM timed out after {self.timeout_s:.3f}s, "
                        f"reader={reader_count}/{job.tx_word_count}, "
                        f"writer={writer_count}/{job.output_words}, stats={self.stats()}"
                    )
                time.sleep(0.0001)

            job.rx.invalidate()

            stats = self.stats()
            if stats.status & STATUS_ERROR:
                raise RuntimeError(f"Accelerator error_code={stats.error_code}, stats={stats}")
            reader_count = self.read_reg(REG_RD_COUNT)
            writer_count = self.read_reg(REG_WR_COUNT)
            if (
                stats.rx_words != job.tx_word_count
                or stats.tx_words != job.output_words
                or reader_count != job.tx_word_count
                or writer_count != job.output_words
            ):
                raise RuntimeError(
                    "Unexpected transfer count: "
                    f"core_rx={stats.rx_words}, core_tx={stats.tx_words}, "
                    f"reader={reader_count}, writer={writer_count}, "
                    f"expected_rx={job.tx_word_count}, expected_tx={job.output_words}"
                )
            flat = np.asarray(job.rx, dtype=np.uint32).reshape(-1)[: job.output_words]
            result = flat.view(np.int32).reshape(job.batch, job.cols).copy()
        except Exception:
            self.release_job(job, abort=True)
            raise
        self.release_job(job)
        return result

    def release_job(self, job: AxiX2MvmJob, *, abort: bool = False) -> None:
        """Abort if requested and release a job's PYNQ buffers exactly once."""

        if job.released:
            return
        if abort:
            self.write_reg(REG_CTRL, CTRL_ABORT | CTRL_CLEAR)
        job.tx.freebuffer()
        job.rx.freebuffer()
        job.released = True


class DualAxiX2MvmPynq:
    """PS-side controller for two independent 32x32 PL PE arrays."""

    def __init__(
        self,
        bit_path: str | Path | None = None,
        *,
        overlay=None,
        accel_names: tuple[str, str] = ("axi_x2_mvm", "axi_x2_mvm_1"),
        download: bool = True,
        timeout_s: float = 5.0,
    ) -> None:
        if overlay is None:
            if bit_path is None:
                raise ValueError("Provide bit_path or an existing PYNQ Overlay")
            from pynq import Overlay  # type: ignore

            overlay = Overlay(str(Path(bit_path)), download=download)

        self.overlay = overlay
        self.engines = tuple(
            AxiX2MvmPynq(
                overlay=overlay,
                accel_name=name,
                download=False,
                timeout_s=timeout_s,
            )
            for name in accel_names
        )
        for engine in self.engines:
            if engine.version < MIN_DUAL_PE_VERSION:
                raise RuntimeError(
                    f"Accelerator '{engine.accel_name}' reports version 0x{engine.version:08x}; "
                    f"dual-PE version 0x{MIN_DUAL_PE_VERSION:08x} or newer is required."
                )

    def run_mvm(self, vector: NDArray[np.integer], weights: NDArray[np.integer]) -> NDArray[np.int32]:
        return self.run_mvm_batch(_as_i8_vector(vector).reshape(1, -1), weights)[0]

    def run_mvm_batch(
        self,
        vectors: NDArray[np.integer],
        weights: NDArray[np.integer],
    ) -> NDArray[np.int32]:
        """Split up to 64 vectors over both PEs and execute them concurrently."""

        batch_vectors = _as_i8_batch(vectors, max_batch=64)
        mat = _as_i8_matrix(weights, rows=batch_vectors.shape[1])
        active_count = min(len(self.engines), batch_vectors.shape[0])
        vector_chunks = np.array_split(batch_vectors, active_count, axis=0)
        active_engines = self.engines[:active_count]
        jobs: list[tuple[AxiX2MvmPynq, AxiX2MvmJob]] = []

        try:
            for engine, chunk in zip(active_engines, vector_chunks):
                jobs.append((engine, engine.prepare_mvm_batch(chunk, mat)))
            for engine, job in jobs:
                engine.start_job(job)
            results = [engine.wait_job(job) for engine, job in jobs]
            return np.concatenate(results, axis=0)
        except Exception:
            for engine, job in jobs:
                engine.release_job(job, abort=job.started)
            raise


def numpy_mvm_reference(vector: NDArray[np.integer], weights: NDArray[np.integer]) -> NDArray[np.int32]:
    vec = _as_i8_vector(vector).astype(np.int32)
    mat = _as_i8_matrix(weights, rows=vec.shape[0]).astype(np.int32)
    return vec @ mat


def numpy_mvm_batch_reference(
    vectors: NDArray[np.integer],
    weights: NDArray[np.integer],
) -> NDArray[np.int32]:
    batch = _as_i8_batch(vectors).astype(np.int32)
    mat = _as_i8_matrix(weights, rows=batch.shape[1]).astype(np.int32)
    return batch @ mat
