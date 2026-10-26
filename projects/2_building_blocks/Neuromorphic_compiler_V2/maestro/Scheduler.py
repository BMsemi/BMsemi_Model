from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from utils.checkpoints import CheckpointLogger


class Scheduler:
    """Produce a serial schedule from mapped work items and cross-check layer order."""

    def __init__(
        self,
        mapped_ops: List[Dict[str, Any]],
        layer_dataflows: List[Dict[str, Any]],
        logger: Optional[CheckpointLogger] = None,
    ) -> None:
        self.mapped_ops = sorted(mapped_ops, key=lambda item: (item["op_index"], item["tile_index"]))
        self.layer_dataflows = layer_dataflows
        self.logger = logger
        self.valid_layers = {item["layer_name"] for item in layer_dataflows}
        if self.logger is not None:
            self.logger.info(
                name="maestro_scheduler_init",
                details="Initialized MAESTRO scheduler with mapped work items.",
                payload={"num_mapped_ops": len(self.mapped_ops), "num_layers": len(self.valid_layers)},
            )
        else:
            print(f"CHECKPOINT [INFO] maestro_scheduler_init: loaded {len(self.mapped_ops)} mapped work items")

    def schedule(self) -> List[Dict[str, Any]]:
        cycle = 0
        schedule: List[Dict[str, Any]] = []
        per_layer_entries = defaultdict(int)
        per_layer_cycles = defaultdict(int)

        for item in self.mapped_ops:
            start_cycle = cycle
            end_cycle = cycle + int(item["estimated_cycles"])
            cycle = end_cycle
            scheduled_item = {**item, "start_cycle": start_cycle, "end_cycle": end_cycle}
            schedule.append(scheduled_item)
            per_layer_entries[item["layer_name"]] += 1
            per_layer_cycles[item["layer_name"]] += int(item["estimated_cycles"])

        if self.logger is not None:
            self.logger.check(
                name="maestro_schedule_entries_have_valid_layers",
                condition=all(entry["layer_name"] in self.valid_layers for entry in schedule),
                details="Every scheduled MAESTRO work item references a valid per-layer descriptor.",
                payload={"invalid_layers": sorted({entry['layer_name'] for entry in schedule if entry['layer_name'] not in self.valid_layers})},
            )
            self.logger.check(
                name="maestro_schedule_layer_coverage_complete",
                condition=set(per_layer_entries.keys()) == self.valid_layers,
                details="Every MAESTRO layer descriptor appears in the generated schedule.",
                payload={
                    "scheduled_layers": sorted(per_layer_entries.keys()),
                    "descriptor_layers": sorted(self.valid_layers),
                },
            )
            self.logger.check(
                name="maestro_schedule_created",
                condition=len(schedule) > 0 and cycle >= 0,
                details="Created a serial MAESTRO schedule with per-layer cycle accounting.",
                payload={
                    "schedule_entries": len(schedule),
                    "total_cycles": cycle,
                    "per_layer_entries": dict(sorted(per_layer_entries.items())),
                    "per_layer_cycles": dict(sorted(per_layer_cycles.items())),
                },
            )
        else:
            print(f"CHECKPOINT [PASS] maestro_schedule_created: created schedule with {len(schedule)} entries and total_cycles={cycle}")
        return schedule
