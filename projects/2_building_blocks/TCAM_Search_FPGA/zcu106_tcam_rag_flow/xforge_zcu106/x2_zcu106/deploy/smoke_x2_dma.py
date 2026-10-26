from pathlib import Path
import time

import numpy as np
from pynq import Overlay, allocate


HEADER_TAG = 0x0DED0000
HEADER_MASK = 0xFFFF0000
EMPTY_TOKEN = 0xDEADC0DE

OP_READ = 0b01 << 30
OP_COMPUTE = 0b10 << 30
OP_WRITE = 0b11 << 30


def fmt_wr_cmd(row: int, col: int, value: int) -> np.uint32:
    return np.uint32(OP_WRITE | ((row & 0x1F) << 25) | ((col & 0x1F) << 20) | (value & 0xFF))


def fmt_comp_cmd(row: int, value: int) -> np.uint32:
    return np.uint32(OP_COMPUTE | ((row & 0x1F) << 25) | (value & 0xFF))


def is_header(word: int) -> bool:
    return (word & HEADER_MASK) == HEADER_TAG


def make_wr_matrix_packet(matrix: np.ndarray, inst_id: int = 0) -> np.ndarray:
    header = np.uint32(HEADER_TAG | (1 << inst_id))
    packet = np.empty((32, 33), dtype=np.uint32)
    packet[:, 0] = header
    for row in range(32):
        for col in range(32):
            packet[row, col + 1] = fmt_wr_cmd(row, col, int(matrix[row, col]))
    return packet.ravel()


def make_vmm_packet(vector: np.ndarray, inst_id: int = 0) -> np.ndarray:
    header = np.uint32(HEADER_TAG | (1 << inst_id))
    packet = np.empty(33, dtype=np.uint32)
    packet[0] = header
    for row in range(32):
        packet[row + 1] = fmt_comp_cmd(row, int(vector[row]))
    return packet


def dma_transfer(dma, tx_words: np.ndarray, rx_words: int = 1056) -> np.ndarray:
    tx = allocate(shape=(len(tx_words),), dtype=np.uint32)
    rx = allocate(shape=(rx_words,), dtype=np.uint32)
    try:
        tx[:] = tx_words
        rx[:] = np.uint32(EMPTY_TOKEN)
        rx.flush()
        tx.flush()

        dma.recvchannel.transfer(rx)
        dma.sendchannel.transfer(tx)
        dma.sendchannel.wait()
        dma.recvchannel.wait()
        rx.invalidate()
        return np.array(rx, dtype=np.uint32)
    finally:
        tx.freebuffer()
        rx.freebuffer()


def extract_payload(words: np.ndarray) -> np.ndarray:
    payload = []
    for word in words.astype(np.uint32):
        value = int(word)
        if value == EMPTY_TOKEN or is_header(value):
            continue
        payload.append(value & 0xFFFF)
        if len(payload) >= 32:
            break
    return np.array(payload, dtype=np.uint16)


def main() -> None:
    here = Path(__file__).resolve().parent
    bit_path = here / "zcu106_x2_dma.bit"

    print(f"Loading overlay: {bit_path}")
    overlay = Overlay(str(bit_path))
    dma = overlay.axi_dma_x2_simd

    matrix = np.zeros((32, 32), dtype=np.uint8)
    np.fill_diagonal(matrix, 0xFF)
    vector = np.arange(32, dtype=np.uint8)

    print("Programming X2 matrix over DMA...")
    t0 = time.perf_counter()
    dma_transfer(dma, make_wr_matrix_packet(matrix))

    print("Running one vector-matrix multiply over DMA...")
    rx = dma_transfer(dma, make_vmm_packet(vector))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    result = extract_payload(rx)
    expected = vector.astype(np.uint16)

    print("Result:", result.tolist())
    print("Expected:", expected.tolist())
    print(f"Elapsed including matrix load: {elapsed_ms:.3f} ms")

    if result.shape != expected.shape or not np.array_equal(result, expected):
        raise SystemExit("FAIL: DMA/X2 result mismatch")

    print("PASS: AXI DMA -> X2 -> AXI DMA smoke test matched.")


if __name__ == "__main__":
    main()
