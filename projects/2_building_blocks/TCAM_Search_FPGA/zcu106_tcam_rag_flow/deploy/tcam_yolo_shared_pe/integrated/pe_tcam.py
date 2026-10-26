from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .arbiter import SharedPEArbiter


LOGICAL_BITS = 16
PHYSICAL_ROWS = 32
PHYSICAL_COLUMNS = 32
ENTRIES_PER_PE = 16
PE_COUNT = 2
TCAM_ENGINE_COUNT = PE_COUNT
YOLO_LAYER0_ENGINE_COUNT = 4
TOTAL_ENGINE_COUNT = TCAM_ENGINE_COUNT + YOLO_LAYER0_ENGINE_COUNT
TOTAL_ENTRIES = ENTRIES_PER_PE * PE_COUNT
FULL_CARE = (1 << LOGICAL_BITS) - 1


@dataclass(frozen=True)
class TcamEntry:
    slot: int
    key: int
    care: int
    chunk_id: int
    valid: bool = True

    def validate(self) -> None:
        if not 0 <= self.slot < TOTAL_ENTRIES:
            raise ValueError(f"TCAM slot {self.slot} is outside 0..31.")
        if not 0 <= self.key <= FULL_CARE:
            raise ValueError("TCAM key must be a 16-bit value.")
        if not 0 <= self.care <= FULL_CARE:
            raise ValueError("TCAM care must be a 16-bit value.")
        if not 0 <= self.chunk_id <= 0xFFFFFFFF:
            raise ValueError("Chunk ID must fit in 32 bits.")


@dataclass(frozen=True)
class TcamMatch:
    slot: int
    chunk_id: int
    key: int
    care: int
    a_current: int
    b_current: int
    b_minus_a: int
    tdc_code: int
    hamming_distance: int
    compared_bits: int


class NumpyMvmEngine:
    """Bit-exact software stand-in for one signed-INT8 FPGA PE."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def run_mvm(self, vector: np.ndarray, weights: np.ndarray) -> np.ndarray:
        self.calls += 1
        return (
            np.asarray(vector, dtype=np.int32)
            @ np.asarray(weights, dtype=np.int32)
        ).astype(np.int32)


def _check_word(value: int, name: str) -> int:
    resolved = int(value)
    if not 0 <= resolved <= FULL_CARE:
        raise ValueError(f"{name} must be a 16-bit value.")
    return resolved


def encode_query_rows(value: int, care: int = FULL_CARE) -> np.ndarray:
    """Encode logical query bits as q/qbar activation rows."""

    query = _check_word(value, "query")
    query_care = _check_word(care, "query care")
    rows = np.zeros(PHYSICAL_ROWS, dtype=np.int8)
    for bit in range(LOGICAL_BITS):
        if not (query_care >> bit) & 1:
            continue
        rows[2 * bit + (0 if (query >> bit) & 1 else 1)] = 1
    return rows


def encode_tcam_bank(entries: Iterable[TcamEntry], pe_index: int) -> np.ndarray:
    """Map 16 ternary keys to complementary A/B columns of one 32x32 PE."""

    if not 0 <= int(pe_index) < PE_COUNT:
        raise ValueError("pe_index must be 0 or 1.")
    matrix = np.zeros((PHYSICAL_ROWS, PHYSICAL_COLUMNS), dtype=np.int8)
    first_slot = int(pe_index) * ENTRIES_PER_PE
    last_slot = first_slot + ENTRIES_PER_PE
    for entry in entries:
        entry.validate()
        if not entry.valid or not first_slot <= entry.slot < last_slot:
            continue
        local_slot = entry.slot - first_slot
        a_col = 2 * local_slot
        b_col = a_col + 1
        for bit in range(LOGICAL_BITS):
            if not (entry.care >> bit) & 1:
                continue
            q_row = 2 * bit
            qbar_row = q_row + 1
            if (entry.key >> bit) & 1:
                matrix[q_row, b_col] = 1
                matrix[qbar_row, a_col] = 1
            else:
                matrix[q_row, a_col] = 1
                matrix[qbar_row, b_col] = 1
    return matrix


class DualPETcamSearch:
    """Run differential TCAM lookup on dedicated Engine 0/1 TCAM PEs."""

    def __init__(
        self,
        engines: Sequence[Any],
        *,
        arbiter: SharedPEArbiter | None = None,
        column_window: int = 16,
    ) -> None:
        if len(engines) != PE_COUNT:
            raise ValueError("The ZCU106 TCAM mapping requires exactly two PEs.")
        self.engines = tuple(engines)
        self.arbiter = arbiter or SharedPEArbiter()
        if int(column_window) not in (16, 32):
            raise ValueError("column_window must be 16 or 32.")
        self.column_window = int(column_window)
        self._entries: dict[int, TcamEntry] = {}
        self.searches = 0
        self.pe_launches = 0

    def program(self, entries: Iterable[TcamEntry]) -> None:
        programmed: dict[int, TcamEntry] = {}
        for entry in entries:
            entry.validate()
            if entry.slot in programmed:
                raise ValueError(f"Duplicate TCAM slot {entry.slot}.")
            if entry.valid:
                programmed[entry.slot] = entry
        self._entries = programmed

    @property
    def entries(self) -> tuple[TcamEntry, ...]:
        return tuple(self._entries[index] for index in sorted(self._entries))

    @staticmethod
    def _run_parallel(
        engines: Sequence[Any],
        query: np.ndarray,
        matrices: Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        if all(
            callable(getattr(engine, "prepare_mvm_batch", None))
            and callable(getattr(engine, "start_job", None))
            and callable(getattr(engine, "wait_job", None))
            for engine in engines
        ):
            outputs: list[np.ndarray] = []
            for engine, matrix in zip(engines, matrices):
                job = engine.prepare_mvm_batch(query.reshape(1, -1), matrix)
                try:
                    engine.start_job(job)
                    outputs.append(np.asarray(engine.wait_job(job)[0], dtype=np.int32))
                except Exception:
                    release = getattr(engine, "release_job", None)
                    if callable(release):
                        release(job, abort=bool(getattr(job, "started", False)))
                    raise
            return outputs

        outputs = []
        for engine, matrix in zip(engines, matrices):
            outputs.append(
                np.asarray(engine.run_mvm(query, matrix), dtype=np.int32)
            )
        return outputs

    def search(
        self,
        query: int,
        *,
        query_care: int = FULL_CARE,
        top_k: int = TOTAL_ENTRIES,
    ) -> list[TcamMatch]:
        if not 1 <= int(top_k) <= TOTAL_ENTRIES:
            raise ValueError("top_k must be in 1..32.")
        query_value = _check_word(query, "query")
        care_value = _check_word(query_care, "query care")
        row_vector = encode_query_rows(query_value, care_value)
        entries = self.entries
        full_matrices = [
            encode_tcam_bank(entries, index) for index in range(PE_COUNT)
        ]
        currents = [np.zeros(PHYSICAL_COLUMNS, dtype=np.int32) for _ in range(PE_COUNT)]
        with self.arbiter.reserve("tcam_search"):
            for first_column in range(0, PHYSICAL_COLUMNS, self.column_window):
                last_column = first_column + self.column_window
                window_outputs = self._run_parallel(
                    self.engines,
                    row_vector,
                    [
                        matrix[:, first_column:last_column]
                        for matrix in full_matrices
                    ],
                )
                for pe_index, output in enumerate(window_outputs):
                    currents[pe_index][first_column:last_column] = output
                self.pe_launches += PE_COUNT
        self.searches += 1

        matches: list[TcamMatch] = []
        for entry in entries:
            pe_index, local_slot = divmod(entry.slot, ENTRIES_PER_PE)
            a_current = int(currents[pe_index][2 * local_slot])
            b_current = int(currents[pe_index][2 * local_slot + 1])
            compared_bits = int((entry.care & care_value).bit_count())
            difference = b_current - a_current
            tdc_code = difference + compared_bits
            numerator = compared_bits - difference
            if numerator < 0 or numerator & 1:
                raise RuntimeError(
                    "Invalid differential TCAM current relation: "
                    f"A={a_current}, B={b_current}, active={compared_bits}."
                )
            matches.append(
                TcamMatch(
                    slot=entry.slot,
                    chunk_id=entry.chunk_id,
                    key=entry.key,
                    care=entry.care,
                    a_current=a_current,
                    b_current=b_current,
                    b_minus_a=difference,
                    tdc_code=tdc_code,
                    hamming_distance=numerator // 2,
                    compared_bits=compared_bits,
                )
            )
        matches.sort(
            key=lambda match: (
                match.hamming_distance,
                -match.compared_bits,
                match.slot,
            )
        )
        return matches[: int(top_k)]
