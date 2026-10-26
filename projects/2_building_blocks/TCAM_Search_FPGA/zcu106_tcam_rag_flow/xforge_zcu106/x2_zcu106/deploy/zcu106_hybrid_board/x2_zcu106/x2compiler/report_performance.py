from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .ir import X2Program


def _fit_stall_costs() -> tuple[float, float]:
    """
    Fit input/output memory-stall cycles from both physical JTAG measurements.

    Built-in 3x2,batch1: 219 measured cycles, 79 no-stall cycles.
    Compiled 27x16,batch32: 24,436 measured, 5,356 no-stall cycles.
    """

    # 3*a + 2*b = 140; 332*a + 512*b = 19080.
    determinant = 3.0 * 512.0 - 2.0 * 332.0
    input_cost = (140.0 * 512.0 - 2.0 * 19080.0) / determinant
    output_cost = (3.0 * 19080.0 - 332.0 * 140.0) / determinant
    return input_cost, output_cost


def performance_report(
    artifact_directory: Path,
    *,
    clock_hz: float = 100_000_000.0,
) -> dict[str, Any]:
    manifest = json.loads(
        (artifact_directory / "model_manifest.json").read_text(encoding="utf-8")
    )
    calls = 0
    logical_macs = 0
    input_words = 0
    output_words = 0
    no_stall_cycles = 0
    unique_weight_bytes = 0
    tile_count = 0
    for item in manifest["compiled_programs"]:
        program = X2Program.load(artifact_directory / item["program"])
        logical_macs += program.m_dim * program.k_dim * program.n_dim
        tile_count += len(program.tiles)
        for tile in program.tiles:
            unique_weight_bytes += tile.rows * tile.cols
            for batch_start in range(0, program.m_dim, program.target.max_batch):
                batch = min(
                    program.target.max_batch,
                    program.m_dim - batch_start,
                )
                weight_words = (tile.rows * tile.cols + 3) // 4
                vector_words = (tile.rows + 3) // 4
                call_input_words = weight_words + batch * vector_words
                call_output_words = batch * tile.cols
                column_groups = math.ceil(tile.cols / 8)
                per_batch_core = (
                    column_groups * (tile.rows + 2)
                    + 2 * tile.cols
                    + 67
                )
                no_stall_cycles += call_input_words + batch * per_batch_core
                input_words += call_input_words
                output_words += call_output_words
                calls += 1

    input_stall, output_stall = _fit_stall_costs()
    estimated_cycles = (
        no_stall_cycles
        + input_stall * input_words
        + output_stall * output_words
    )
    return {
        "schema_version": 1,
        "artifact_directory": str(artifact_directory.resolve()),
        "clock_hz": clock_hz,
        "compiled_conv2d_count": len(manifest["compiled_programs"]),
        "tile_count": tile_count,
        "hardware_launches": calls,
        "logical_macs": logical_macs,
        "unique_weight_bytes": unique_weight_bytes,
        "input_words": input_words,
        "output_words": output_words,
        "physical_input_bytes": input_words * 16,
        "physical_output_bytes": math.ceil(output_words / 4) * 16,
        "no_stall_cycles": no_stall_cycles,
        "empirical_stall_model": {
            "input_cycles_per_word": input_stall,
            "output_cycles_per_word": output_stall,
            "calibration_points": [
                {
                    "shape": [3, 2, 1],
                    "measured_cycles": 219,
                },
                {
                    "shape": [27, 16, 32],
                    "measured_cycles": 24436,
                },
            ],
        },
        "estimated_pl_cycles": estimated_cycles,
        "estimated_pl_seconds": estimated_cycles / clock_hz,
        "estimated_pl_fps_upper_bound": clock_hz / estimated_cycles,
        "scope": (
            "Analytical full-Conv2D PL estimate fitted to two physical JTAG "
            "measurements. It excludes Python/PYNQ allocation, MMIO launch, "
            "CPU fallback operators, preprocessing, and postprocessing."
        ),
        "bottleneck": (
            "The prototype retransmits every weight tile on every spatial "
            "batch and performs 99k host launches. Persistent on-chip weights "
            "and a graph/command queue are required for realtime operation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate full-model X2 traffic and PL latency."
    )
    parser.add_argument("artifacts", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--clock-hz", type=float, default=100_000_000.0)
    args = parser.parse_args()
    report = performance_report(args.artifacts, clock_hz=args.clock_hz)
    output = args.output or args.artifacts / "performance_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
