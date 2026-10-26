from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CheckpointRecord:
    name: str
    status: str
    details: str
    payload: Optional[Dict[str, Any]] = None
    timestamp_utc: str = ""


class CheckpointLogger:
    """Collects debug checkpoints and writes a machine-readable report."""

    def __init__(self) -> None:
        self.records: List[CheckpointRecord] = []

    def _append(
        self,
        name: str,
        status: str,
        details: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = CheckpointRecord(
            name=name,
            status=status,
            details=details,
            payload=payload,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        self.records.append(record)
        print(f"CHECKPOINT [{status}] {name}: {details}")
        if payload:
            print(f"  payload={json.dumps(payload, sort_keys=True, default=str)}")

    def info(self, name: str, details: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self._append(name=name, status="INFO", details=details, payload=payload)

    def check(
        self,
        name: str,
        condition: bool,
        details: str,
        payload: Optional[Dict[str, Any]] = None,
        raise_on_fail: bool = True,
    ) -> None:
        status = "PASS" if condition else "FAIL"
        self._append(name=name, status=status, details=details, payload=payload)
        if raise_on_fail and not condition:
            raise AssertionError(f"Checkpoint failed: {name} :: {details}")

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for record in self.records:
            counts[record.status] = counts.get(record.status, 0) + 1
        return {
            "total": len(self.records),
            "counts": counts,
            "records": [asdict(record) for record in self.records],
        }

    def dump_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2, sort_keys=True))
