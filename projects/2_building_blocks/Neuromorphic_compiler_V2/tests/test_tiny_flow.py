from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts" / "test_run_summary.json"
ONNX_STATUS = ROOT / "artifacts" / "onnx_export_status.json"
TFLITE_STATUS = ROOT / "artifacts" / "tflite_export_status.json"
MAESTRO_JSON = ROOT / "artifacts" / "maestro_layer_dataflow.json"
MAESTRO_STATUS = ROOT / "artifacts" / "maestro_integration_status.json"
MAESTRO_SCHEDULE = ROOT / "artifacts" / "maestro_schedule.json"


def test_flow_runs_end_to_end() -> None:
    subprocess.run(["python", str(ROOT / "optimize_onnx_and_map.py")], check=True, cwd=str(ROOT))
    summary = json.loads(REPORT.read_text())
    assert summary["test_accuracy"] >= summary["accuracy_threshold"]
    assert summary["schedule_entries"] > 0
    assert summary["dispatch_report"]["executed_entries"] == summary["schedule_entries"]
    assert summary["maestro_reference_matches_generated"] is True
    assert summary["maestro_json_reference_matches_generated"] is True
    assert summary["maestro_layer_count"] > 0
    assert ONNX_STATUS.exists()
    assert TFLITE_STATUS.exists()
    assert MAESTRO_JSON.exists()
    assert MAESTRO_STATUS.exists()
    assert MAESTRO_SCHEDULE.exists()
