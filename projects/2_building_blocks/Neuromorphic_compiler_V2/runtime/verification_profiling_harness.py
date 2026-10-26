from __future__ import annotations

from typing import Any, Dict


def verify_run(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "accuracy_ok": summary.get("test_accuracy", 0.0) >= summary.get("accuracy_threshold", 0.0),
        "schedule_ok": summary.get("schedule_entries", 0) > 0,
        "dispatch_ok": summary.get("dispatch_report", {}).get("executed_entries", 0) == summary.get("schedule_entries", 0),
        "maestro_ok": bool(summary.get("maestro_reference_matches_generated", False)) and bool(summary.get("maestro_json_reference_matches_generated", False)) and int(summary.get("maestro_layer_count", 0)) > 0,
    }
