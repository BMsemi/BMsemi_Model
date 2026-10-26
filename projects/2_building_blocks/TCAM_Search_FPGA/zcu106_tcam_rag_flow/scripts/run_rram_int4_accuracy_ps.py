#!/usr/bin/env python3
"""Run the INT4 RRAM programming-accuracy model on the Zynq PS."""

from __future__ import annotations

import argparse
import json

from model_rram_int4_programming_accuracy import run_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-per-code", type=int, default=1000)
    parser.add_argument("--sample-records", type=int, default=0)
    parser.add_argument("--g-min-us", type=float, default=5.0)
    parser.add_argument("--g-max-us", type=float, default=95.0)
    parser.add_argument("--read-noise-sigma-us", type=float, default=0.75)
    parser.add_argument("--pulse-cv", type=float, default=0.20)
    parser.add_argument("--max-pulses", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    result = run_model(args)
    summary = result["summary"]
    print(
        json.dumps(
            {
                "runner": "zynq_ps_rram_int4_accuracy",
                "accuracy": summary["accuracy"],
                "total_trials": summary["total_trials"],
                "mean_pulses": summary["mean_pulses"],
                "max_pulses_used": summary["max_pulses_used"],
                "source_warning": result["source"]["warning"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
