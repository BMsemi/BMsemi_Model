from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np


LAYER0_TILE0_VECTOR = np.array(
    [
        10,
        15,
        12,
        -12,
        -14,
        -9,
        -14,
        16,
        -7,
        -3,
        -11,
        11,
        -16,
        16,
        -13,
        -1,
        -2,
        -6,
        10,
        -3,
        5,
        1,
        5,
        -5,
        -6,
        10,
        0,
    ],
    dtype=np.int8,
)

KNOWN_EXPECTED = np.array(
    [
        -639,
        6036,
        930,
        -1578,
        -1169,
        -5337,
        2450,
        -2970,
        1529,
        -5588,
        3381,
        907,
        -2073,
        768,
        -658,
        1119,
    ],
    dtype=np.int32,
)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO26n Conv2D op 2 tile 0 on both PL PEs and report live TDC data."
        )
    )
    parser.add_argument(
        "--bitstream",
        type=Path,
        default=root / "overlay" / "zcu106_axi_x2_mvm.bit",
    )
    parser.add_argument(
        "--constants",
        type=Path,
        default=root / "artifacts" / "op_0002_conv2d" / "constants.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "board_output" / "first_conv_tile0_tdc.json",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(root))
    from x2_zcu106.runtime.axi_x2_mvm_driver import (
        DualAxiX2MvmPynq,
        numpy_mvm_batch_reference,
    )

    with np.load(args.constants) as constants:
        weights = np.asarray(constants["weight_k0_n0"], dtype=np.int8)
    vectors = np.stack((LAYER0_TILE0_VECTOR, LAYER0_TILE0_VECTOR))
    expected = numpy_mvm_batch_reference(vectors, weights)
    if not np.array_equal(expected[0], KNOWN_EXPECTED):
        raise RuntimeError(
            "The packaged first-layer weights do not match the validated tile-0 artifact."
        )

    x2 = DualAxiX2MvmPynq(args.bitstream)
    result = x2.run_mvm_batch(vectors, weights)
    stats = [engine.stats() for engine in x2.engines]
    output_matches = np.array_equal(result, expected)
    tdc_valid = all(item.tdc_timing_valid for item in stats)

    report = {
        "status": "pass" if output_matches and tdc_valid else "fail",
        "case": "tflite_conv2d_op_2_tile_0",
        "scope": {
            "rows": 27,
            "cols": 16,
            "vectors_per_pe": 1,
            "full_weight_tile": True,
            "full_spatial_layer": False,
            "full_spatial_layer_vectors": 102400,
        },
        "bitstream": str(args.bitstream.resolve()),
        "pe_versions": [f"0x{engine.version:08X}" for engine in x2.engines],
        "expected_i32": expected.tolist(),
        "result_i32": result.tolist(),
        "output_matches": output_matches,
        "pe_stats": [asdict(item) for item in stats],
        "x2.engines[0].stats().tdc_pe_completion_time_ps": (
            stats[0].tdc_pe_completion_time_ps
        ),
        "x2.engines[1].stats().tdc_pe_completion_time_ps": (
            stats[1].tdc_pe_completion_time_ps
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if not output_matches:
        raise SystemExit("FAIL: one or both FPGA PE outputs differ from the reference")
    if not tdc_valid:
        raise SystemExit("FAIL: FPGA outputs matched, but one or both TDC reports are invalid")


if __name__ == "__main__":
    main()
