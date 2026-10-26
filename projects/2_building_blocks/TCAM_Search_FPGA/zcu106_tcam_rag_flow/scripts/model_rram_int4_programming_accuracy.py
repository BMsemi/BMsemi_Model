#!/usr/bin/env python3
"""
Diagnostic 4-bit RRAM programming-accuracy model from preparation-pulse data.

The source heatmap is labeled diagnostic-only, so this script is a compact
what-if model for architecture exploration, not a qualified device model.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


NAN = float("nan")


@dataclass(frozen=True)
class PulseHeatmap:
    mode: str
    conductance_edges_us: tuple[float, ...]
    voltage_centers_v: tuple[float, ...]
    delta_g_us: tuple[tuple[float, ...], ...]

    def rows_for_conductance(self, conductance_us: float) -> list[tuple[float, float]]:
        values = np.asarray(self.delta_g_us, dtype=float)
        edges = np.asarray(self.conductance_edges_us, dtype=float)
        voltage = np.asarray(self.voltage_centers_v, dtype=float)
        bin_index = int(np.searchsorted(edges, conductance_us, side="right") - 1)
        bin_index = max(0, min(bin_index, values.shape[1] - 1))

        rows: list[tuple[float, float]] = []
        for v, mean_delta in zip(voltage, values[:, bin_index]):
            if not math.isnan(float(mean_delta)):
                rows.append((float(v), float(mean_delta)))

        if rows:
            return rows

        # Sparse gray heatmap bins are filled by nearest measured conductance bin.
        for offset in range(1, values.shape[1]):
            for near in (bin_index - offset, bin_index + offset):
                if near < 0 or near >= values.shape[1]:
                    continue
                rows = []
                for v, mean_delta in zip(voltage, values[:, near]):
                    if not math.isnan(float(mean_delta)):
                        rows.append((float(v), float(mean_delta)))
                if rows:
                    return rows
        return []


SET_HEATMAP = PulseHeatmap(
    mode="SET",
    conductance_edges_us=(0.0, 20.0, 40.0, 60.0, 80.0, 100.0),
    voltage_centers_v=(1.35, 1.25, 1.15, 1.05, 0.95, 0.85, 0.75, 0.65, 0.55, 0.45),
    delta_g_us=(
        (NAN, NAN, NAN, 10.1, NAN),
        (NAN, 18.4, 8.6, 3.6, NAN),
        (21.3, 15.1, 0.8, -1.8, NAN),
        (0.2, 3.3, 0.2, -2.0, NAN),
        (7.5, 4.5, -3.2, -0.7, NAN),
        (10.7, 2.6, 1.9, -2.6, NAN),
        (-4.2, -4.2, 2.8, 2.0, NAN),
        (3.0, 1.2, 3.1, 4.2, NAN),
        (5.3, 4.1, 4.9, 4.2, NAN),
        (6.5, 4.0, 6.9, 4.8, -8.5),
    ),
)


RESET_HEATMAP = PulseHeatmap(
    mode="RESET",
    conductance_edges_us=(20.0, 40.0, 60.0, 80.0, 100.0),
    voltage_centers_v=(
        2.75,
        2.65,
        2.55,
        2.45,
        2.35,
        2.25,
        2.15,
        2.05,
        1.95,
        1.85,
        1.75,
        1.65,
        1.55,
        1.45,
        1.35,
        1.25,
        1.15,
        1.05,
        0.95,
    ),
    delta_g_us=(
        (NAN, 29.9, NAN, NAN),
        (10.2, 16.2, NAN, NAN),
        (2.9, -12.3, NAN, NAN),
        (-0.5, 5.7, NAN, NAN),
        (2.3, 6.7, NAN, NAN),
        (-1.4, 7.7, 20.8, NAN),
        (-2.4, -2.6, 14.0, 13.9),
        (-0.8, -4.6, 0.1, 12.4),
        (-1.7, 0.5, 1.2, 2.7),
        (4.9, -3.8, 0.0, NAN),
        (2.2, -3.4, 1.5, 4.9),
        (-0.2, 1.0, -0.5, 7.4),
        (-3.0, -2.8, 0.8, -2.3),
        (-0.3, 2.4, 0.6, 6.9),
        (-1.1, 2.5, 2.8, 6.1),
        (2.2, 0.8, 0.7, 9.0),
        (-1.5, 2.7, 0.7, 6.1),
        (1.7, 0.1, 1.2, 7.0),
        (NAN, NAN, NAN, NAN),
    ),
)


def int4_targets(g_min_us: float, g_max_us: float) -> np.ndarray:
    return np.linspace(g_min_us, g_max_us, 16, dtype=float)


def decode_int4(conductance_us: float, targets: np.ndarray) -> int:
    return int(np.argmin(np.abs(targets - conductance_us)))


def choose_pulse(
    heatmap: PulseHeatmap,
    conductance_us: float,
    required_delta_us: float,
) -> tuple[float, float]:
    candidates = [(v, d) for v, d in heatmap.rows_for_conductance(conductance_us) if d > 0.0]
    if not candidates:
        return (0.0, 0.0)

    under = [(v, d) for v, d in candidates if d <= required_delta_us]
    if under:
        return min(under, key=lambda vd: abs(required_delta_us - vd[1]))
    return min(candidates, key=lambda vd: vd[1])


def program_one_cell(
    target_code: int,
    rng: np.random.Generator,
    targets: np.ndarray,
    g_min_us: float,
    g_max_us: float,
    read_noise_sigma_us: float,
    pulse_cv: float,
    max_pulses: int,
) -> dict[str, object]:
    target = float(targets[target_code])
    g = float(rng.uniform(g_min_us, g_max_us))
    initial_g = g
    pulses: list[dict[str, float]] = []

    for _ in range(max_pulses):
        measured = float(g + rng.normal(0.0, read_noise_sigma_us))
        decoded = decode_int4(measured, targets)
        if decoded == target_code:
            break

        if measured < target:
            mode = "SET"
            voltage, mean_delta = choose_pulse(SET_HEATMAP, g, target - measured)
            signed_delta = 1.0
        else:
            mode = "RESET"
            voltage, mean_delta = choose_pulse(RESET_HEATMAP, g, measured - target)
            signed_delta = -1.0

        if mean_delta <= 0.0:
            break

        sigma = max(read_noise_sigma_us, abs(mean_delta) * pulse_cv)
        actual_delta = max(0.0, float(rng.normal(mean_delta, sigma)))
        g = float(np.clip(g + signed_delta * actual_delta, g_min_us, g_max_us))
        pulses.append(
            {
                "mode": mode,
                "voltage_v": voltage,
                "mean_delta_g_us": mean_delta,
                "actual_delta_g_us": actual_delta,
                "after_g_us": g,
            }
        )

    final_measured = float(g + rng.normal(0.0, read_noise_sigma_us))
    final_code = decode_int4(final_measured, targets)
    return {
        "target_code": target_code,
        "target_g_us": target,
        "initial_g_us": initial_g,
        "final_g_us": g,
        "final_measured_g_us": final_measured,
        "decoded_code": final_code,
        "correct": final_code == target_code,
        "pulse_count": len(pulses),
        "pulses": pulses,
    }


def summarize_trials(records: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(records)
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    by_code = []
    for code in range(16):
        subset = [row for row in rows if row["target_code"] == code]
        if not subset:
            continue
        by_code.append(
            {
                "code": code,
                "trials": len(subset),
                "accuracy": sum(1 for row in subset if row["correct"]) / len(subset),
                "mean_pulses": float(np.mean([row["pulse_count"] for row in subset])),
                "p95_abs_error_us": float(
                    np.percentile(
                        [abs(float(row["final_measured_g_us"]) - float(row["target_g_us"])) for row in subset],
                        95,
                    )
                ),
            }
        )
    return {
        "total_trials": total,
        "correct_trials": correct,
        "accuracy": correct / total if total else 0.0,
        "mean_pulses": float(np.mean([row["pulse_count"] for row in rows])) if rows else 0.0,
        "max_pulses_used": max((int(row["pulse_count"]) for row in rows), default=0),
        "by_code": by_code,
    }


def run_model(args: argparse.Namespace) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    targets = int4_targets(args.g_min_us, args.g_max_us)
    records = []
    for code in range(16):
        for _ in range(args.trials_per_code):
            records.append(
                program_one_cell(
                    code,
                    rng,
                    targets,
                    args.g_min_us,
                    args.g_max_us,
                    args.read_noise_sigma_us,
                    args.pulse_cv,
                    args.max_pulses,
                )
            )
    summary = summarize_trials(records)
    return {
        "schema_version": 1,
        "source": {
            "image": "Accuracy_INT4.jpeg",
            "warning": "Diagnostic preparation-pulse data only; not a qualified characterization grid.",
            "set_pulse_pairs": 405,
            "reset_pulse_pairs": 933,
            "read_vcc_set_v": 0.50,
            "set_reset_vcc_set_v": 2.30,
            "delta_definition": "SET: G_after-G_before; RESET: G_before-G_after",
        },
        "model": {
            "application": "INT4 / 16-level RRAM weight programming",
            "target_conductance_us": [float(v) for v in targets],
            "conductance_range_us": [args.g_min_us, args.g_max_us],
            "read_noise_sigma_us": args.read_noise_sigma_us,
            "pulse_coefficient_of_variation": args.pulse_cv,
            "max_verify_pulses": args.max_pulses,
            "pulse_policy": "verify after each pulse; choose measured positive SET/RESET delta closest to remaining target distance",
        },
        "summary": summary,
        "sample_records": records[: args.sample_records],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-per-code", type=int, default=250)
    parser.add_argument("--g-min-us", type=float, default=5.0)
    parser.add_argument("--g-max-us", type=float, default=95.0)
    parser.add_argument("--read-noise-sigma-us", type=float, default=0.75)
    parser.add_argument("--pulse-cv", type=float, default=0.20)
    parser.add_argument("--max-pulses", type=int, default=24)
    parser.add_argument("--sample-records", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    result = run_model(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
