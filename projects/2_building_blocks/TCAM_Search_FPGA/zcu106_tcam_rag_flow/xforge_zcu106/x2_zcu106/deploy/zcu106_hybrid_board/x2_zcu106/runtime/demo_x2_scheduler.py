from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from x2_runtime import Int8BitPlaneScheduler, NumpyBinaryX2Backend, PynqDmaBinaryX2Backend


def build_backend(args: argparse.Namespace):
    if args.backend == "numpy":
        return NumpyBinaryX2Backend(num_devices=args.devices)
    return PynqDmaBinaryX2Backend(args.bit, num_devices=args.devices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise the ZCU106 X2 INT8 bit-plane scheduler.")
    parser.add_argument("--backend", choices=("numpy", "pynq"), default="numpy")
    parser.add_argument("--bit", default=str(Path(__file__).resolve().parents[1] / "deploy" / "zcu106_x2_dma.bit"))
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--k", type=int, default=77)
    parser.add_argument("--n", type=int, default=71)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    x = rng.integers(0, 256, size=(args.batch, args.k), dtype=np.uint8)
    w = rng.integers(-128, 128, size=(args.k, args.n), dtype=np.int16).astype(np.int8)

    backend = build_backend(args)
    scheduler = Int8BitPlaneScheduler(backend)
    estimate = scheduler.estimate_matmul(x, w, signed_inputs=False)
    y = scheduler.matmul_u8_i8(x, w)
    expected = x.astype(np.int32) @ w.astype(np.int32)

    print(f"Backend: {args.backend}")
    print(f"Shape: {x.shape} @ {w.shape} -> {y.shape}")
    print(f"Weight programs: {estimate.weight_programs}")
    print(f"Binary VMM calls: {estimate.vmm_calls}")
    print(f"Max abs error: {int(np.max(np.abs(y - expected)))}")

    if not np.array_equal(y, expected):
        raise SystemExit("FAIL: scheduler result mismatch")

    print("PASS: INT8 matmul matches NumPy exactly.")


if __name__ == "__main__":
    main()
