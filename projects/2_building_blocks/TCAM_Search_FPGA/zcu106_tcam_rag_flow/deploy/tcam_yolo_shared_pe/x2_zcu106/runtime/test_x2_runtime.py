from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from axi_x2_mvm_driver import (
    AxiX2MvmPynq,
    DualAxiX2MvmPynq,
    MAGIC_X2MV,
    MIN_CONTINUOUS_TDC_VERSION,
    REG_CYCLES,
    REG_ERROR,
    REG_EXPECTED,
    REG_RD_ADDR_HI,
    REG_RD_ADDR_LO,
    REG_RD_COUNT,
    REG_RX_WORDS,
    REG_STATUS,
    REG_TDC_ARRIVAL_MAX,
    REG_TDC_ARRIVAL_MIN,
    REG_TDC_ARRIVAL_SPREAD,
    REG_TDC_CONFIG,
    REG_TDC_ORDER,
    REG_TDC_PE_COMPLETION,
    REG_TDC_TIMING_MAX,
    REG_TDC_TIMING_MIN,
    REG_TX_WORDS,
    REG_WR_ADDR_HI,
    REG_WR_ADDR_LO,
    REG_WR_COUNT,
    STATUS_DONE,
    numpy_mvm_batch_reference,
    pack_mvm_batch_stream,
)
from x2_runtime import Int8BitPlaneScheduler, NumpyBinaryX2Backend


def _fake_overlay_and_pynq(version: int):
    registers = {
        0x20: MAGIC_X2MV,
        0x24: version,
    }

    class FakeMMIO:
        def __init__(self, _address: int, _length: int) -> None:
            pass

        def read(self, offset: int) -> int:
            return registers.get(offset, 0)

        def write(self, _offset: int, _value: int) -> None:
            pass

    overlay = SimpleNamespace(
        ip_dict={"axi_x2_mvm": {"phys_addr": 0, "addr_range": 0x1000}},
        axi_dma_mvm=object(),
    )
    pynq = SimpleNamespace(MMIO=FakeMMIO, allocate=lambda **_kwargs: None)
    return overlay, pynq


def test_axi_driver_rejects_legacy_ddr_reader_contract() -> None:
    overlay, pynq = _fake_overlay_and_pynq(MIN_CONTINUOUS_TDC_VERSION - 1)
    with patch.dict(sys.modules, {"pynq": pynq}):
        try:
            AxiX2MvmPynq(overlay=overlay)
        except RuntimeError as exc:
            assert "corrected 128-bit DDR reader" in str(exc)
        else:
            raise AssertionError("Legacy PL version was accepted")


def test_matmul_u8_i8() -> None:
    rng = np.random.default_rng(123)
    x = rng.integers(0, 256, size=(5, 77), dtype=np.uint8)
    w = rng.integers(-128, 128, size=(77, 71), dtype=np.int16).astype(np.int8)
    scheduler = Int8BitPlaneScheduler(NumpyBinaryX2Backend(num_devices=8))

    got = scheduler.matmul_u8_i8(x, w)
    expected = x.astype(np.int32) @ w.astype(np.int32)

    np.testing.assert_array_equal(got, expected)


def test_matmul_i8_i8() -> None:
    rng = np.random.default_rng(456)
    x = rng.integers(-128, 128, size=(3, 65), dtype=np.int16).astype(np.int8)
    w = rng.integers(-128, 128, size=(65, 40), dtype=np.int16).astype(np.int8)
    scheduler = Int8BitPlaneScheduler(NumpyBinaryX2Backend(num_devices=4))

    got = scheduler.matmul_i8_i8(x, w)
    expected = x.astype(np.int32) @ w.astype(np.int32)

    np.testing.assert_array_equal(got, expected)


def test_conv2d_tflite_u8_i8() -> None:
    rng = np.random.default_rng(789)
    x = rng.integers(0, 256, size=(1, 6, 5, 3), dtype=np.uint8)
    w = rng.integers(-8, 8, size=(7, 3, 3, 3), dtype=np.int16).astype(np.int8)
    bias = rng.integers(-100, 100, size=(7,), dtype=np.int32)
    scheduler = Int8BitPlaneScheduler(NumpyBinaryX2Backend(num_devices=8))

    got = scheduler.conv2d_tflite_u8_i8(x, w, bias_i32=bias, stride=(2, 1), padding="same")

    padded = np.pad(x.astype(np.int32), ((0, 0), (0, 1), (1, 1), (0, 0)), constant_values=0)
    expected = np.zeros_like(got)
    for oy in range(got.shape[1]):
        for ox in range(got.shape[2]):
            patch = padded[:, oy * 2 : oy * 2 + 3, ox : ox + 3, :]
            for oc in range(w.shape[0]):
                expected[:, oy, ox, oc] = np.sum(patch * w[oc].astype(np.int32), axis=(1, 2, 3)) + bias[oc]

    np.testing.assert_array_equal(got, expected)


def test_depthwise_conv2d_tflite_u8_i8() -> None:
    rng = np.random.default_rng(987)
    x = rng.integers(0, 256, size=(1, 5, 4, 2), dtype=np.uint8)
    w = rng.integers(-8, 8, size=(1, 3, 3, 4), dtype=np.int16).astype(np.int8)
    bias = rng.integers(-100, 100, size=(4,), dtype=np.int32)
    scheduler = Int8BitPlaneScheduler(NumpyBinaryX2Backend(num_devices=8))

    got = scheduler.depthwise_conv2d_tflite_u8_i8(x, w, bias_i32=bias, stride=(1, 1), padding="same")

    padded = np.pad(x.astype(np.int32), ((0, 0), (1, 1), (1, 1), (0, 0)), constant_values=0)
    expected = np.zeros_like(got)
    kh, kw = 3, 3
    in_ch = x.shape[-1]
    multiplier = w.shape[-1] // in_ch
    w_hwcm = w.reshape(kh, kw, in_ch, multiplier).astype(np.int32)
    for oy in range(got.shape[1]):
        for ox in range(got.shape[2]):
            patch = padded[:, oy : oy + kh, ox : ox + kw, :]
            for c in range(in_ch):
                for m in range(multiplier):
                    oc = c * multiplier + m
                    expected[:, oy, ox, oc] = np.sum(patch[:, :, :, c] * w_hwcm[:, :, c, m], axis=(1, 2)) + bias[oc]

    np.testing.assert_array_equal(got, expected)


def test_axi_driver_reads_tdc_timing_registers() -> None:
    registers = {
        REG_STATUS: 0x3,
        REG_EXPECTED: 4,
        REG_RX_WORDS: 4,
        REG_TX_WORDS: 2,
        REG_CYCLES: 20,
        REG_ERROR: 0,
        REG_TDC_ORDER: (1 << 31) | (0 << 8) | 1,
        REG_TDC_ARRIVAL_MIN: 495,
        REG_TDC_ARRIVAL_MAX: 514,
        REG_TDC_ARRIVAL_SPREAD: 19,
        REG_TDC_PE_COMPLETION: 514,
        REG_TDC_CONFIG: (1 << 31) | 256,
        REG_TDC_TIMING_MIN: 0,
        REG_TDC_TIMING_MAX: 576,
    }
    driver = object.__new__(AxiX2MvmPynq)
    driver.read_reg = lambda offset: registers[offset]

    stats = driver.stats()

    assert stats.tdc_timing_valid
    assert stats.tdc_model_enabled
    assert stats.tdc_reference_levels == 256
    assert stats.tdc_timing_min_ps == 0
    assert stats.tdc_timing_max_ps == 576
    assert stats.tdc_first_ready_column == 1
    assert stats.tdc_last_ready_column == 0
    assert stats.tdc_arrival_min_ps == 495
    assert stats.tdc_arrival_max_ps == 514
    assert stats.tdc_arrival_spread_ps == 19
    assert stats.tdc_pe_completion_time_ps == 514


def test_batch_stream_pads_each_vector_independently() -> None:
    vectors = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    weights = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.int8)

    packed = pack_mvm_batch_stream(vectors, weights)

    np.testing.assert_array_equal(
        packed,
        np.array(
            [
                0x0A090807,  # weights [7, 8, 9, 10]
                0x00000C0B,  # weights [11, 12] plus padding
                0x00030201,  # vector 0 plus independent padding
                0x00060504,  # vector 1 plus independent padding
            ],
            dtype=np.uint32,
        ),
    )


def test_numpy_batch_reference() -> None:
    vectors = np.array([[1, 2, 3], [-4, 5, -6]], dtype=np.int8)
    weights = np.array([[7, -8], [9, 10], [-11, 12]], dtype=np.int8)
    expected = vectors.astype(np.int32) @ weights.astype(np.int32)
    np.testing.assert_array_equal(numpy_mvm_batch_reference(vectors, weights), expected)


def test_axi_driver_batch_uses_custom_ddr_contract() -> None:
    vectors = np.array([[1, 2, 3], [-4, 5, -6]], dtype=np.int8)
    weights = np.array([[7, -8, 3], [9, 10, -2], [-11, 12, 5]], dtype=np.int8)
    expected = numpy_mvm_batch_reference(vectors, weights)
    buffers = []

    class FakeBuffer:
        def __init__(self, shape, address: int) -> None:
            self.array = np.zeros(shape, dtype=np.uint32)
            self.physical_address = address
            self.freed = False

        def __getitem__(self, key):
            return self.array[key]

        def __setitem__(self, key, value) -> None:
            self.array[key] = value

        def __array__(self, dtype=None, copy=None):
            result = np.asarray(self.array, dtype=dtype)
            return result.copy() if copy else result

        def flush(self) -> None:
            pass

        def invalidate(self) -> None:
            pass

        def freebuffer(self) -> None:
            self.freed = True

    def allocate(*, shape, dtype):
        assert dtype is np.uint32
        address = 0x10000000 if not buffers else 0x11000000
        result = FakeBuffer(shape, address)
        buffers.append(result)
        return result

    registers = {
        REG_STATUS: 0,
        REG_ERROR: 0,
    }
    writes = []
    driver = object.__new__(AxiX2MvmPynq)
    driver.allocate = allocate
    driver.timeout_s = 1.0
    driver.read_reg = lambda offset: registers.get(offset, 0)

    def write_reg(offset: int, value: int) -> None:
        writes.append((offset, value))
        if offset == 0 and value == 1:
            tx_words = pack_mvm_batch_stream(vectors, weights)
            output_words = expected.size
            buffers[1].array.reshape(-1)[:output_words] = expected.reshape(-1).view(np.uint32)
            registers.update(
                {
                    REG_STATUS: STATUS_DONE | 1,
                    REG_EXPECTED: tx_words.size,
                    REG_RX_WORDS: tx_words.size,
                    REG_TX_WORDS: output_words,
                    REG_RD_COUNT: tx_words.size,
                    REG_WR_COUNT: output_words,
                }
            )

    driver.write_reg = write_reg
    result = driver.run_mvm_batch(vectors, weights)

    np.testing.assert_array_equal(result, expected)
    packed = pack_mvm_batch_stream(vectors, weights)
    assert buffers[0].array.shape == (packed.size, 4)
    np.testing.assert_array_equal(buffers[0].array[:, 0], packed)
    np.testing.assert_array_equal(buffers[0].array[:, 1:], 0)
    assert buffers[1].array.shape == (2, 4)
    assert (REG_RD_ADDR_LO, 0x10000000) in writes
    assert (REG_RD_ADDR_HI, 0) in writes
    assert (REG_WR_ADDR_LO, 0x11000000) in writes
    assert (REG_WR_ADDR_HI, 0) in writes
    assert all(buffer.freed for buffer in buffers)


def test_dual_axi_driver_starts_both_pes_before_waiting() -> None:
    vectors = np.arange(15, dtype=np.int8).reshape(5, 3)
    weights = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int8)
    events = []

    class FakeEngine:
        def __init__(self, name: str) -> None:
            self.name = name

        def prepare_mvm_batch(self, chunk, mat):
            events.append(("prepare", self.name, chunk.shape[0]))
            return SimpleNamespace(
                vectors=chunk.copy(),
                weights=mat.copy(),
                started=False,
                released=False,
            )

        def start_job(self, job) -> None:
            events.append(("start", self.name))
            job.started = True

        def wait_job(self, job):
            assert [event[:2] for event in events[:4]] == [
                ("prepare", "pe0"),
                ("prepare", "pe1"),
                ("start", "pe0"),
                ("start", "pe1"),
            ]
            events.append(("wait", self.name))
            job.released = True
            return numpy_mvm_batch_reference(job.vectors, job.weights)

        def release_job(self, job, *, abort=False) -> None:
            events.append(("release", self.name, abort))
            job.released = True

    driver = object.__new__(DualAxiX2MvmPynq)
    driver.engines = (FakeEngine("pe0"), FakeEngine("pe1"))

    result = driver.run_mvm_batch(vectors, weights)

    np.testing.assert_array_equal(result, numpy_mvm_batch_reference(vectors, weights))
    assert events[0] == ("prepare", "pe0", 3)
    assert events[1] == ("prepare", "pe1", 2)


if __name__ == "__main__":
    test_axi_driver_rejects_legacy_ddr_reader_contract()
    test_matmul_u8_i8()
    test_matmul_i8_i8()
    test_conv2d_tflite_u8_i8()
    test_depthwise_conv2d_tflite_u8_i8()
    test_axi_driver_reads_tdc_timing_registers()
    test_batch_stream_pads_each_vector_independently()
    test_numpy_batch_reference()
    test_axi_driver_batch_uses_custom_ddr_contract()
    test_dual_axi_driver_starts_both_pes_before_waiting()
    print("PASS")
