#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
IMAGE="${IMAGE:-deploy/tcam_yolo_shared_pe/bus.jpg}"
INSTALL_REQS="${INSTALL_REQS:-1}"
NO_SUDO="${NO_SUDO:-0}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-runs/remote_hw_${TIMESTAMP}}"
CACHE_DIR="${CACHE_DIR:-${RUN_DIR}/cache}"
FIRST_OUTPUT="${FIRST_OUTPUT:-${RUN_DIR}/output_first}"
SECOND_OUTPUT="${SECOND_OUTPUT:-${RUN_DIR}/output_second}"

mkdir -p "$RUN_DIR"

if [[ "$INSTALL_REQS" != "0" ]]; then
  "$PYTHON_BIN" -m pip install -r deploy/tcam_yolo_shared_pe/requirements-board.txt \
    2>&1 | tee "${RUN_DIR}/pip_install.log"
fi

if [[ "$NO_SUDO" == "1" || "$(id -u)" == "0" ]]; then
  RUN_PREFIX=()
else
  RUN_PREFIX=(sudo -E)
fi

run_pass() {
  local name="$1"
  local output_dir="$2"
  local log_file="${RUN_DIR}/${name}_run.log"
  "${RUN_PREFIX[@]}" "$PYTHON_BIN" run_tcam_yolo.py "$IMAGE" \
    --mode hardware \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$output_dir" \
    2>&1 | tee "$log_file"
  return "${PIPESTATUS[0]}"
}

run_pass first "$FIRST_OUTPUT"
FIRST_STATUS=$?
run_pass second "$SECOND_OUTPUT"
SECOND_STATUS=$?

"$PYTHON_BIN" - "$RUN_DIR" "$FIRST_STATUS" "$SECOND_STATUS" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
first_status = int(sys.argv[2])
second_status = int(sys.argv[3])


def parse_json_log(path: Path):
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def summarize(payload):
    if not isinstance(payload, dict):
        return None
    result = payload.get("result", {})
    tcam = payload.get("tcam", {})
    execution = result.get("execution", {})
    detections = result.get("detections", [])
    dispatch = result.get("dispatch", {})
    return {
        "path": payload.get("path"),
        "yolo_skipped": payload.get("yolo_skipped"),
        "visual_key": tcam.get("visual_key"),
        "match": tcam.get("match"),
        "inserted": payload.get("inserted"),
        "backend_counts": execution.get("backend_counts"),
        "dispatch_counts": dispatch.get("dispatch_counts"),
        "fpga_layer_embeddings": result.get("fpga_layer_embeddings"),
        "detection_count": result.get("detection_count", len(detections)),
        "detections": [
            {
                "label": det.get("label") or det.get("class_name"),
                "confidence": det.get("confidence"),
                "box_xyxy": det.get("box_xyxy"),
            }
            for det in detections
        ],
        "artifacts": result.get("artifacts"),
    }


first = parse_json_log(run_dir / "first_run.log")
second = parse_json_log(run_dir / "second_run.log")
summary = {
    "run_dir": str(run_dir),
    "first_status": first_status,
    "second_status": second_status,
    "first": summarize(first),
    "second": summarize(second),
}
(run_dir / "summary_remote_hw.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2))
PY

if [[ "$FIRST_STATUS" -ne 0 || "$SECOND_STATUS" -ne 0 ]]; then
  exit 1
fi
