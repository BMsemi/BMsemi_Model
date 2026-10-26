from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Iterator


@dataclass(frozen=True)
class ArbiterSnapshot:
    owner: str | None
    transitions: int


class SharedPEArbiter:
    """Record ownership transitions across the partitioned PE engine group."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._owner: str | None = None
        self._transitions = 0

    @contextmanager
    def reserve(self, owner: str) -> Iterator[None]:
        if not owner:
            raise ValueError("PE owner must be named.")
        with self._lock:
            previous = self._owner
            self._owner = owner
            self._transitions += 1
            try:
                yield
            finally:
                self._owner = previous

    def snapshot(self) -> ArbiterSnapshot:
        with self._lock:
            return ArbiterSnapshot(self._owner, self._transitions)
