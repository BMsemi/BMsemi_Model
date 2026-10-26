from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0
DEFAULT_KV_CACHE_CONTEXT_TOKENS = 45_056
DEFAULT_KV_CACHE_TOKEN_RATE_TPS = 10.0
DEFAULT_KV_CACHE_TOKEN_TIME_S = 1.0 / DEFAULT_KV_CACHE_TOKEN_RATE_TPS
DEFAULT_KV_CACHE_SERVICE_LIFE_YEARS = 5.0


@dataclass(frozen=True)
class StatefulKVCacheConfig:
    layers: int
    heads: int
    head_dim: int
    max_tokens: int
    token_time_s: float = DEFAULT_KV_CACHE_TOKEN_TIME_S
    spare_slots: int = 0
    endurance_cycles: int = 10_000_000
    remap_life_fraction: float = 0.80
    bad_cell_probability: float = 0.0
    seed: int | None = None
    bad_cell_value_policy: str = "zero"
    service_life_years: float = DEFAULT_KV_CACHE_SERVICE_LIFE_YEARS

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KVCacheRead:
    keys: np.ndarray
    values: np.ndarray
    token_ids: np.ndarray
    logical_slots: np.ndarray
    key_physical_slots: np.ndarray
    value_physical_slots: np.ndarray
    key_ages_s: np.ndarray
    value_ages_s: np.ndarray


class StatefulKVCache:
    """
    Append-only token-addressed K/V cache state for decode experiments.

    The model tracks logical cache slots separately from physical ReRAM slots.
    Decode writes program only the newly generated token's K/V vectors; older
    context slots are read in place until eviction. Slot wear therefore spreads
    round-robin over the configured context window. When a mapped physical slot
    approaches its endurance target, the logical slot is remapped to a spare
    physical slot if one is available.
    """

    def __init__(self, config: StatefulKVCacheConfig, dtype: np.dtype | type = np.float32):
        self.config = config
        self.dtype = np.dtype(dtype)
        self.layers = int(config.layers)
        self.heads = int(config.heads)
        self.head_dim = int(config.head_dim)
        self.max_tokens = int(config.max_tokens)
        self.physical_slots = self.max_tokens + int(config.spare_slots)
        if self.layers <= 0:
            raise ValueError("layers must be > 0")
        if self.heads <= 0:
            raise ValueError("heads must be > 0")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be > 0")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.physical_slots < self.max_tokens:
            raise ValueError("spare_slots cannot make physical slots smaller than max_tokens")

        shape = (self.layers, self.physical_slots, self.heads, self.head_dim)
        self.keys = np.zeros(shape, dtype=self.dtype)
        self.values = np.zeros(shape, dtype=self.dtype)

        logical_shape = (self.layers, self.max_tokens)
        initial_map = np.tile(np.arange(self.max_tokens, dtype=np.int32), (self.layers, 1))
        self._key_logical_to_physical = initial_map.copy()
        self._value_logical_to_physical = initial_map.copy()
        self._occupied = np.zeros(logical_shape, dtype=bool)
        self._token_ids = np.full(logical_shape, -1, dtype=np.int64)

        physical_shape = (self.layers, self.physical_slots)
        self._key_write_counts = np.zeros(physical_shape, dtype=np.int64)
        self._value_write_counts = np.zeros(physical_shape, dtype=np.int64)
        self._key_program_time_s = np.full(physical_shape, np.nan, dtype=np.float64)
        self._value_program_time_s = np.full(physical_shape, np.nan, dtype=np.float64)

        rng = np.random.default_rng(config.seed)
        p_bad = np.clip(float(config.bad_cell_probability), 0.0, 1.0)
        self._bad_keys = rng.random(physical_shape) < p_bad if p_bad > 0.0 else np.zeros(physical_shape, dtype=bool)
        self._bad_values = rng.random(physical_shape) < p_bad if p_bad > 0.0 else np.zeros(physical_shape, dtype=bool)
        self._initial_bad_keys = int(np.count_nonzero(self._bad_keys))
        self._initial_bad_values = int(np.count_nonzero(self._bad_values))

        self.total_key_writes = 0
        self.total_value_writes = 0
        self.total_key_reads = 0
        self.total_value_reads = 0
        self.evictions = 0
        self.key_remaps = 0
        self.value_remaps = 0
        self.failed_key_remaps = 0
        self.failed_value_remaps = 0
        self.key_bad_cell_write_events = 0
        self.value_bad_cell_write_events = 0
        self.key_bad_cell_read_events = 0
        self.value_bad_cell_read_events = 0
        self.key_cells_retired = 0
        self.value_cells_retired = 0
        self.duplicate_token_appends = 0

    @property
    def remap_threshold(self) -> int | None:
        endurance = int(self.config.endurance_cycles)
        if endurance <= 0:
            return None
        fraction = min(max(float(self.config.remap_life_fraction), 0.0), 1.0)
        return max(1, int(np.floor(float(endurance) * fraction)))

    def append(
        self,
        layer_idx: int,
        token_idx: int,
        key: np.ndarray,
        value: np.ndarray,
        *,
        now_s: float | None = None,
    ) -> None:
        layer = self._check_layer(layer_idx)
        logical_slot = int(token_idx) % self.max_tokens
        time_s = self._time_for_token(token_idx, now_s)
        key_arr = self._coerce_vector(key, "key")
        value_arr = self._coerce_vector(value, "value")

        if self._occupied[layer, logical_slot]:
            if self._token_ids[layer, logical_slot] == int(token_idx):
                self.duplicate_token_appends += 1
                return
            self.evictions += 1

        key_phys = self._prepare_write(layer, logical_slot, kind="key")
        value_phys = self._prepare_write(layer, logical_slot, kind="value")
        self.keys[layer, key_phys] = key_arr
        self.values[layer, value_phys] = value_arr
        self._occupied[layer, logical_slot] = True
        self._token_ids[layer, logical_slot] = int(token_idx)

        self._key_write_counts[layer, key_phys] += 1
        self._value_write_counts[layer, value_phys] += 1
        self._key_program_time_s[layer, key_phys] = time_s
        self._value_program_time_s[layer, value_phys] = time_s
        self.total_key_writes += 1
        self.total_value_writes += 1

        self._retire_if_past_endurance(layer, key_phys, kind="key")
        self._retire_if_past_endurance(layer, value_phys, kind="value")

    def read_layer(self, layer_idx: int, *, upto_token_idx: int | None = None, now_s: float | None = None) -> KVCacheRead:
        layer = self._check_layer(layer_idx)
        logical_slots = self._active_logical_slots(layer, upto_token_idx)
        if logical_slots.size == 0:
            empty = np.zeros((0, self.heads, self.head_dim), dtype=self.dtype)
            empty_int = np.zeros((0,), dtype=np.int64)
            empty_float = np.zeros((0,), dtype=np.float64)
            return KVCacheRead(
                empty,
                empty.copy(),
                empty_int,
                empty_int.copy(),
                empty_int.copy(),
                empty_int.copy(),
                empty_float,
                empty_float.copy(),
            )

        token_ids = self._token_ids[layer, logical_slots].astype(np.int64, copy=True)
        time_s = self._time_for_token(int(np.max(token_ids)), now_s)
        key_phys = self._key_logical_to_physical[layer, logical_slots]
        value_phys = self._value_logical_to_physical[layer, logical_slots]

        keys = self.keys[layer, key_phys].copy()
        values = self.values[layer, value_phys].copy()
        bad_key_mask = self._bad_keys[layer, key_phys]
        bad_value_mask = self._bad_values[layer, value_phys]
        self.key_bad_cell_read_events += int(np.count_nonzero(bad_key_mask))
        self.value_bad_cell_read_events += int(np.count_nonzero(bad_value_mask))
        if str(self.config.bad_cell_value_policy).lower() == "zero":
            keys[bad_key_mask] = 0
            values[bad_value_mask] = 0

        self.total_key_reads += int(logical_slots.size)
        self.total_value_reads += int(logical_slots.size)
        key_ages = np.maximum(0.0, time_s - self._key_program_time_s[layer, key_phys])
        value_ages = np.maximum(0.0, time_s - self._value_program_time_s[layer, value_phys])
        return KVCacheRead(
            keys,
            values,
            token_ids,
            logical_slots.astype(np.int64, copy=True),
            key_phys.astype(np.int64, copy=True),
            value_phys.astype(np.int64, copy=True),
            key_ages,
            value_ages,
        )

    def cached_token_count(self, layer_idx: int | None = None) -> int:
        if layer_idx is None:
            return int(np.count_nonzero(self._occupied))
        return int(np.count_nonzero(self._occupied[self._check_layer(layer_idx)]))

    def step_summary(self, *, now_s: float | None = None) -> dict[str, Any]:
        return {
            "cached_entries": self.cached_token_count(),
            "total_key_writes": int(self.total_key_writes),
            "total_value_writes": int(self.total_value_writes),
            "key_remaps": int(self.key_remaps),
            "value_remaps": int(self.value_remaps),
            "failed_key_remaps": int(self.failed_key_remaps),
            "failed_value_remaps": int(self.failed_value_remaps),
            "max_key_write_count": int(np.max(self._key_write_counts)) if self._key_write_counts.size else 0,
            "max_value_write_count": int(np.max(self._value_write_counts)) if self._value_write_counts.size else 0,
            "max_age_s": self._age_stats(now_s)["max_age_s"],
            "bad_key_cells": int(np.count_nonzero(self._bad_keys)),
            "bad_value_cells": int(np.count_nonzero(self._bad_values)),
            "duplicate_token_appends": int(self.duplicate_token_appends),
        }

    def summary(self, *, now_s: float | None = None) -> dict[str, Any]:
        key_counts = self._active_write_counts(kind="key")
        value_counts = self._active_write_counts(kind="value")
        key_physical_counts = self._physical_write_counts(kind="key")
        value_physical_counts = self._physical_write_counts(kind="value")
        ages = self._age_stats(now_s)
        lifetime = self._append_only_lifetime_projection()
        return {
            "config": self.config.as_dict(),
            "write_policy": "append_only_new_token",
            "slot_wear_policy": "round_robin_context_slots",
            "physical_slots_per_layer": int(self.physical_slots),
            "logical_slots_per_layer": int(self.max_tokens),
            "cached_entries": self.cached_token_count(),
            "evictions": int(self.evictions),
            "duplicate_token_appends": int(self.duplicate_token_appends),
            "total_key_writes": int(self.total_key_writes),
            "total_value_writes": int(self.total_value_writes),
            "total_key_reads": int(self.total_key_reads),
            "total_value_reads": int(self.total_value_reads),
            "key_remaps": int(self.key_remaps),
            "value_remaps": int(self.value_remaps),
            "failed_key_remaps": int(self.failed_key_remaps),
            "failed_value_remaps": int(self.failed_value_remaps),
            "initial_bad_key_cells": int(self._initial_bad_keys),
            "initial_bad_value_cells": int(self._initial_bad_values),
            "bad_key_cells": int(np.count_nonzero(self._bad_keys)),
            "bad_value_cells": int(np.count_nonzero(self._bad_values)),
            "key_cells_retired": int(self.key_cells_retired),
            "value_cells_retired": int(self.value_cells_retired),
            "key_bad_cell_write_events": int(self.key_bad_cell_write_events),
            "value_bad_cell_write_events": int(self.value_bad_cell_write_events),
            "key_bad_cell_read_events": int(self.key_bad_cell_read_events),
            "value_bad_cell_read_events": int(self.value_bad_cell_read_events),
            "remap_threshold": self.remap_threshold,
            "key_write_count_min": int(np.min(key_counts)) if key_counts.size else 0,
            "key_write_count_mean": float(np.mean(key_counts)) if key_counts.size else 0.0,
            "key_write_count_max": int(np.max(key_counts)) if key_counts.size else 0,
            "value_write_count_min": int(np.min(value_counts)) if value_counts.size else 0,
            "value_write_count_mean": float(np.mean(value_counts)) if value_counts.size else 0.0,
            "value_write_count_max": int(np.max(value_counts)) if value_counts.size else 0,
            "key_physical_write_count_min": int(np.min(key_physical_counts)) if key_physical_counts.size else 0,
            "key_physical_write_count_mean": float(np.mean(key_physical_counts)) if key_physical_counts.size else 0.0,
            "key_physical_write_count_max": int(np.max(key_physical_counts)) if key_physical_counts.size else 0,
            "value_physical_write_count_min": int(np.min(value_physical_counts)) if value_physical_counts.size else 0,
            "value_physical_write_count_mean": float(np.mean(value_physical_counts)) if value_physical_counts.size else 0.0,
            "value_physical_write_count_max": int(np.max(value_physical_counts)) if value_physical_counts.size else 0,
            "append_only_lifetime_projection": lifetime,
            **ages,
        }

    def _check_layer(self, layer_idx: int) -> int:
        layer = int(layer_idx)
        if layer < 0 or layer >= self.layers:
            raise IndexError(f"layer_idx {layer_idx} is outside [0, {self.layers})")
        return layer

    def _coerce_vector(self, arr: np.ndarray, name: str) -> np.ndarray:
        value = np.asarray(arr, dtype=self.dtype)
        if value.shape == (1, self.heads, self.head_dim):
            value = value[0]
        if value.shape != (self.heads, self.head_dim):
            raise ValueError(f"{name} must have shape ({self.heads}, {self.head_dim}), got {value.shape}")
        return value

    def _time_for_token(self, token_idx: int, now_s: float | None) -> float:
        if now_s is not None:
            return float(now_s)
        return float(token_idx) * max(float(self.config.token_time_s), 0.0)

    def _kind_arrays(self, kind: str):
        if kind == "key":
            return (
                self._key_logical_to_physical,
                self._key_write_counts,
                self._bad_keys,
            )
        if kind == "value":
            return (
                self._value_logical_to_physical,
                self._value_write_counts,
                self._bad_values,
            )
        raise ValueError(f"unknown cache kind {kind!r}")

    def _prepare_write(self, layer: int, logical_slot: int, *, kind: str) -> int:
        logical_to_physical, write_counts, bad = self._kind_arrays(kind)
        phys = int(logical_to_physical[layer, logical_slot])
        threshold = self.remap_threshold
        needs_remap = bool(bad[layer, phys])
        if threshold is not None and int(write_counts[layer, phys]) >= threshold:
            needs_remap = True
        if needs_remap:
            next_phys = self._find_spare_physical(layer, kind=kind)
            if next_phys is not None:
                logical_to_physical[layer, logical_slot] = next_phys
                phys = int(next_phys)
                if kind == "key":
                    self.key_remaps += 1
                else:
                    self.value_remaps += 1
            else:
                if kind == "key":
                    self.failed_key_remaps += 1
                else:
                    self.failed_value_remaps += 1
        if bool(bad[layer, phys]):
            if kind == "key":
                self.key_bad_cell_write_events += 1
            else:
                self.value_bad_cell_write_events += 1
        return phys

    def _find_spare_physical(self, layer: int, *, kind: str) -> int | None:
        logical_to_physical, write_counts, bad = self._kind_arrays(kind)
        used = np.zeros((self.physical_slots,), dtype=bool)
        used[logical_to_physical[layer]] = True
        candidates = np.flatnonzero(~used & ~bad[layer])
        if candidates.size == 0:
            return None
        candidate_counts = write_counts[layer, candidates]
        return int(candidates[int(np.argmin(candidate_counts))])

    def _retire_if_past_endurance(self, layer: int, phys: int, *, kind: str) -> None:
        endurance = int(self.config.endurance_cycles)
        if endurance <= 0:
            return
        _, write_counts, bad = self._kind_arrays(kind)
        if int(write_counts[layer, phys]) <= endurance or bool(bad[layer, phys]):
            return
        bad[layer, phys] = True
        if kind == "key":
            self.key_cells_retired += 1
        else:
            self.value_cells_retired += 1

    def _active_logical_slots(self, layer: int, upto_token_idx: int | None) -> np.ndarray:
        mask = self._occupied[layer].copy()
        if upto_token_idx is not None:
            mask &= self._token_ids[layer] <= int(upto_token_idx)
        slots = np.flatnonzero(mask)
        if slots.size == 0:
            return slots
        order = np.argsort(self._token_ids[layer, slots])
        return slots[order]

    def _active_write_counts(self, *, kind: str) -> np.ndarray:
        logical_to_physical, write_counts, _bad = self._kind_arrays(kind)
        counts: list[np.ndarray] = []
        for layer in range(self.layers):
            slots = np.flatnonzero(self._occupied[layer])
            if slots.size:
                counts.append(write_counts[layer, logical_to_physical[layer, slots]])
        if not counts:
            return np.zeros((0,), dtype=np.int64)
        return np.concatenate(counts).astype(np.int64, copy=False)

    def _physical_write_counts(self, *, kind: str) -> np.ndarray:
        _, write_counts, _bad = self._kind_arrays(kind)
        return write_counts.reshape(-1).astype(np.int64, copy=False)

    def _append_only_lifetime_projection(self) -> dict[str, Any]:
        token_time_s = max(float(self.config.token_time_s), 0.0)
        service_life_years = max(float(self.config.service_life_years), 0.0)
        service_life_s = service_life_years * SECONDS_PER_YEAR
        token_rate_tps = 1.0 / token_time_s if token_time_s > 0.0 else 0.0
        projected_tokens = service_life_s * token_rate_tps
        cycles_per_slot = projected_tokens / float(max(1, self.max_tokens))
        endurance = int(self.config.endurance_cycles)
        return {
            "service_life_years": service_life_years,
            "service_life_s": service_life_s,
            "decode_token_rate_tps": token_rate_tps,
            "context_slots": int(self.max_tokens),
            "projected_decode_tokens": projected_tokens,
            "projected_key_write_cycles_per_slot": cycles_per_slot,
            "projected_value_write_cycles_per_slot": cycles_per_slot,
            "endurance_cycles": endurance,
            "endurance_margin": float(endurance) / cycles_per_slot if endurance > 0 and cycles_per_slot > 0.0 else None,
            "assumption": "Append-only K/V cache writes one new token slot per decode step and reuses older slots until context eviction.",
        }

    def _age_stats(self, now_s: float | None) -> dict[str, Any]:
        key_ages: list[np.ndarray] = []
        value_ages: list[np.ndarray] = []
        for layer in range(self.layers):
            slots = np.flatnonzero(self._occupied[layer])
            if slots.size == 0:
                continue
            token_now = int(np.max(self._token_ids[layer, slots]))
            time_s = self._time_for_token(token_now, now_s)
            key_phys = self._key_logical_to_physical[layer, slots]
            value_phys = self._value_logical_to_physical[layer, slots]
            key_ages.append(np.maximum(0.0, time_s - self._key_program_time_s[layer, key_phys]))
            value_ages.append(np.maximum(0.0, time_s - self._value_program_time_s[layer, value_phys]))
        if not key_ages:
            return {
                "mean_key_age_s": 0.0,
                "max_key_age_s": 0.0,
                "mean_value_age_s": 0.0,
                "max_value_age_s": 0.0,
                "max_age_s": 0.0,
            }
        key = np.concatenate(key_ages)
        value = np.concatenate(value_ages)
        return {
            "mean_key_age_s": float(np.mean(key)),
            "max_key_age_s": float(np.max(key)),
            "mean_value_age_s": float(np.mean(value)),
            "max_value_age_s": float(np.max(value)),
            "max_age_s": float(max(np.max(key), np.max(value))),
        }
