from __future__ import annotations

from typing import Any, Dict, List


class RuntimeDispatcher:
    """Simulate dispatch of the scheduled work items."""

    def __init__(self, plan: Dict[str, Any], schedule: List[Dict[str, Any]]) -> None:
        self.plan = plan
        self.schedule = schedule
        print(
            "CHECKPOINT [INFO] runtime_dispatcher_init: "
            f"schedule_entries={len(schedule)} total_bytes={plan.get('total_bytes', 0)}"
        )

    def dispatch(self) -> Dict[str, Any]:
        unique_addresses = {entry["address"] for entry in self.plan["layout"].values()}
        if len(unique_addresses) != len(self.plan["layout"]):
            raise AssertionError("Duplicate memory addresses detected in runtime plan.")
        dispatch_report = {
            "executed_entries": len(self.schedule),
            "total_cycles": self.schedule[-1]["end_cycle"] if self.schedule else 0,
            "unique_addresses": len(unique_addresses),
        }
        print(
            "CHECKPOINT [PASS] runtime_dispatch_completed: "
            f"executed_entries={dispatch_report['executed_entries']} total_cycles={dispatch_report['total_cycles']}"
        )
        return dispatch_report
