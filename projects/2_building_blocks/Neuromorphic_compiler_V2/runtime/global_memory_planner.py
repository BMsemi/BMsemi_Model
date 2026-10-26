from __future__ import annotations

from typing import Any, Dict, List


class GlobalMemoryPlanner:
    """Assign a simple contiguous memory layout to scheduled tiles."""

    def __init__(self, schedule: List[Dict[str, Any]]) -> None:
        self.schedule = schedule
        print(
            "CHECKPOINT [INFO] runtime_memory_planner_init: "
            f"schedule_entries={len(schedule)}"
        )

    def plan(self) -> Dict[str, Any]:
        layout: Dict[str, Dict[str, int | str]] = {}
        cursor = 0
        for entry in self.schedule:
            tile_id = f"{entry['module_name']}::tile{entry['tile_index']}"
            size_bytes = int(entry["tile_channels"]) * 4
            layout[tile_id] = {
                "address": cursor,
                "size_bytes": size_bytes,
                "bank": entry["bank"],
            }
            cursor += size_bytes
        plan = {
            "layout": layout,
            "total_bytes": cursor,
            "num_entries": len(self.schedule),
        }
        print(
            "CHECKPOINT [PASS] runtime_memory_plan_created: "
            f"allocated {plan['total_bytes']} bytes across {plan['num_entries']} entries"
        )
        return plan
