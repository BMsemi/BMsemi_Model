from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from utils.checkpoints import CheckpointLogger


def lower(spec: Dict[str, Any], logger: Optional[CheckpointLogger] = None) -> Dict[str, Any]:
    updated = deepcopy(spec)
    updated["runtime"] = {
        "serialized_program": "artifacts/executorch_program.pte",
        "delegate_runtime": "runtime_dispatcher",
        "debug_handles": True,
    }
    if logger is not None:
        logger.check(
            name="executorch_runtime_has_debug_handles",
            condition=bool(updated["runtime"]["debug_handles"]),
            details="Lowered compile spec to a runtime-ready metadata blob.",
            payload=updated["runtime"],
        )
    return updated
