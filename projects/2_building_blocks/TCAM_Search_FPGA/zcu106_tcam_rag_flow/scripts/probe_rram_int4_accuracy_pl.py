#!/usr/bin/env python3
"""Probe the PL-side INT4 RRAM accuracy LUT through AXI-Lite MMIO."""

from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import time


CTRL = 0x00
STATUS = 0x04
TARGET = 0x08
INIT_G = 0x0C
FINAL_G = 0x10
RESULT = 0x14
TARGET_G = 0x18
LAST_DG = 0x1C
LAST_V = 0x20
MAGIC = 0x24
VERSION = 0x28


def _u32(mm: mmap.mmap, offset: int) -> int:
    mm.seek(offset)
    return struct.unpack("<I", mm.read(4))[0]


def _w32(mm: mmap.mmap, offset: int, value: int) -> None:
    mm.seek(offset)
    mm.write(struct.pack("<I", value & 0xFFFFFFFF))
    mm.flush()


def _s16(value: int) -> int:
    value &= 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def run_probe(base: int, target_code: int, initial_g_tenth_us: int, timeout_s: float) -> dict[str, object]:
    page_size = mmap.PAGESIZE
    page_base = base & ~(page_size - 1)
    page_offset = base - page_base
    span = page_offset + 0x1000
    with open("/dev/mem", "r+b", buffering=0) as devmem:
        mm = mmap.mmap(devmem.fileno(), span, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=page_base)
        try:
            def read(offset: int) -> int:
                return _u32(mm, page_offset + offset)

            def write(offset: int, value: int) -> None:
                _w32(mm, page_offset + offset, value)

            magic = read(MAGIC)
            version = read(VERSION)
            if magic != 0x52344143:
                raise RuntimeError(f"unexpected PL RRAM magic 0x{magic:08X}, expected 0x52344143")

            write(CTRL, 0x2)
            write(TARGET, target_code)
            write(INIT_G, initial_g_tenth_us)
            write(CTRL, 0x1)

            deadline = time.monotonic() + timeout_s
            status = 0
            while time.monotonic() < deadline:
                status = read(STATUS)
                if status & 0x4:
                    break
                time.sleep(0.001)
            else:
                raise TimeoutError(f"PL RRAM model did not finish; status=0x{status:08X}")

            result = read(RESULT)
            last_v_word = read(LAST_V)
            return {
                "runner": "zynq_ps_to_pl_rram_int4_accuracy",
                "base": f"0x{base:08X}",
                "magic": f"0x{magic:08X}",
                "version": f"0x{version:08X}",
                "target_code": target_code,
                "initial_g_tenth_us": initial_g_tenth_us,
                "target_g_tenth_us": read(TARGET_G) & 0x3FF,
                "final_g_tenth_us": read(FINAL_G) & 0x3FF,
                "decoded_code": result & 0xF,
                "pulse_count": (result >> 8) & 0x3F,
                "correct": bool(status & 0x8),
                "last_delta_tenth_us": _s16(read(LAST_DG)),
                "last_mode": "RESET" if ((last_v_word >> 16) & 0x1) else "SET",
                "last_voltage_mv": last_v_word & 0xFFFF,
            }
        finally:
            mm.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=lambda s: int(s, 0), default=0xA0060000)
    parser.add_argument("--target-code", type=int, default=10)
    parser.add_argument("--initial-g-tenth-us", type=int, default=500)
    parser.add_argument("--timeout-s", type=float, default=1.0)
    args = parser.parse_args()

    if os.geteuid() != 0:
        raise SystemExit("Run with sudo so /dev/mem is accessible.")
    print(json.dumps(run_probe(args.base, args.target_code, args.initial_g_tenth_us, args.timeout_s), indent=2))


if __name__ == "__main__":
    main()
