from __future__ import annotations

from typing import Dict, List

from .onnx_ir import ONNXIR


NUM_PES = 4
NUM_BANKS = 2


def assign(ir: ONNXIR) -> Dict[str, List[Dict[str, int | str]]]:
    assignments: List[Dict[str, int | str]] = []
    for idx, tile in enumerate(ir.tiles):
        entry = dict(tile)
        entry["pe"] = f"PE_{idx % NUM_PES}"
        entry["bank"] = f"BANK_{idx % NUM_BANKS}"
        entry["estimated_work_units"] = int(tile["tile_channels"]) * 4
        assignments.append(entry)

    ir.mapping = assignments
    if ir.logger is not None:
        ir.logger.check(
            name="mapping_assignments_cover_all_tiles",
            condition=len(assignments) == len(ir.tiles),
            details="Assigned every tile to a PE and memory bank.",
            payload={"num_assignments": len(assignments), "num_tiles": len(ir.tiles)},
        )
    return {"tiles": assignments}
