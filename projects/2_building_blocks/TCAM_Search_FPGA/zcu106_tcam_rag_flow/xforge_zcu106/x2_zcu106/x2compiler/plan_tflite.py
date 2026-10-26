from __future__ import annotations

import argparse
import json
from pathlib import Path

from .planner import plan_tflite_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partition a TFLite graph and build its X2 activation arena."
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output (default: <artifacts>/execution_plan.json).",
    )
    parser.add_argument("--alignment", type=int, default=64)
    args = parser.parse_args()

    plan = plan_tflite_model(
        args.model,
        args.artifacts,
        alignment=args.alignment,
    )
    output = args.output or args.artifacts / "execution_plan.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output.resolve()),
        "backend_counts": plan["partition"]["backend_counts"],
        "partition_count": plan["partition"]["partition_count"],
        "arena_bytes": plan["memory"]["arena_bytes"],
        "naive_activation_bytes": plan["memory"]["naive_activation_bytes"],
        "reduction_percent": plan["memory"]["reduction_percent"],
    }, indent=2))


if __name__ == "__main__":
    main()
