from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .mapping import ceil_div
from .planner import ProgramManifest


@dataclass(frozen=True)
class X1SystemConfig:
    max_x1_macros: int = 20000
    x1_macros_per_endpoint: int = 4
    x1_rows: int = 32
    x1_cols: int = 32
    fp4llm_tile_rows: int = 64
    fp4llm_tile_cols: int = 32
    stored_cell_banks_per_weight: int = 4
    flit_bits: int = 128
    peripheral_memory_bytes: int = 8 * 1024**3
    die_count: int = 1
    dma_available: bool = False

    @property
    def endpoints(self) -> int:
        return max(1, int(self.max_x1_macros) // max(1, int(self.x1_macros_per_endpoint)))

    @property
    def x1_cell_capacity(self) -> int:
        return int(self.max_x1_macros) * int(self.x1_rows) * int(self.x1_cols)

    @property
    def fp4llm_row_shards_per_tile(self) -> int:
        return ceil_div(self.fp4llm_tile_rows, self.x1_rows)

    @property
    def resident_fp4llm_tiles(self) -> int:
        return max(1, int(self.max_x1_macros) // self.fp4llm_row_shards_per_tile)

    def as_dict(self) -> dict[str, object]:
        out = asdict(self)
        out.update(
            {
                "endpoints": self.endpoints,
                "x1_cell_capacity": self.x1_cell_capacity,
                "fp4llm_row_shards_per_tile": self.fp4llm_row_shards_per_tile,
                "resident_fp4llm_tiles": self.resident_fp4llm_tiles,
            }
        )
        return out


def _layer_capacity(layer_plan: object, system: X1SystemConfig) -> dict[str, object]:
    layer = dict(layer_plan.layer)
    in_dim = int(layer["in_dim"])
    out_dim = int(layer["out_dim"])
    static_weights = bool(layer["static_weights"])
    dense_x1_tiles = (
        ceil_div(in_dim, system.x1_rows)
        * ceil_div(out_dim, system.x1_cols)
        * (system.stored_cell_banks_per_weight if static_weights else 0)
    )
    stored_cells = int(layer_plan.weight_program_cells)
    min_x1_by_cells = ceil_div(stored_cells, system.x1_rows * system.x1_cols) if stored_cells else 0
    return {
        "phase": layer["phase"],
        "name": layer["name"],
        "role": layer["role"],
        "tokens": int(layer["tokens"]),
        "in_dim": in_dim,
        "out_dim": out_dim,
        "static_weights": static_weights,
        "weight_elements": int(layer["weight_elements"]) if static_weights else 0,
        "stored_cells": stored_cells,
        "dense_x1_tiles": dense_x1_tiles,
        "min_x1_by_cells": min_x1_by_cells,
        "resident_fits_system": min_x1_by_cells <= system.max_x1_macros,
        "compute_commands": int(layer_plan.x1_compute_commands),
        "noc_bits": int(layer_plan.x1_noc_flits) * system.flit_bits,
    }


def build_deployment_report(manifest: ProgramManifest, system: X1SystemConfig | None = None) -> dict[str, object]:
    cfg = system or X1SystemConfig(
        x1_macros_per_endpoint=int(manifest.compiler_config.get("x1_macros_per_endpoint", 4)),
        x1_rows=int(manifest.compiler_config.get("x1_macro_rows", 32)),
        x1_cols=int(manifest.compiler_config.get("x1_macro_cols", 32)),
        fp4llm_tile_rows=int(manifest.compiler_config.get("fp4llm_tile_rows", 64)),
        fp4llm_tile_cols=int(manifest.compiler_config.get("fp4llm_tile_cols", 32)),
        stored_cell_banks_per_weight=int(manifest.compiler_config.get("stored_cell_banks_per_weight", 4)),
    )
    layers = [_layer_capacity(layer, cfg) for layer in manifest.layers]
    static_layers = [layer for layer in layers if layer["static_weights"]]
    unique_static_layers: list[dict[str, object]] = []
    seen_static: set[tuple[str, str, int, int]] = set()
    for layer in static_layers:
        key = (
            str(layer["name"]),
            str(layer["role"]),
            int(layer["in_dim"]),
            int(layer["out_dim"]),
        )
        if key in seen_static:
            continue
        seen_static.add(key)
        unique_static_layers.append(layer)
    total_stored_cells = sum(int(layer["stored_cells"]) for layer in unique_static_layers)
    total_weight_elements = sum(int(layer["weight_elements"]) for layer in unique_static_layers)
    min_x1_for_resident = ceil_div(total_stored_cells, cfg.x1_rows * cfg.x1_cols)
    largest_layers = sorted(unique_static_layers, key=lambda item: int(item["min_x1_by_cells"]), reverse=True)[:12]
    resident_fit = min_x1_for_resident <= cfg.max_x1_macros
    execution_mode = "resident" if resident_fit else "streamed_weight_windows"
    dma_needed = (not resident_fit) and (sum(int(layer.weight_program_commands) for layer in manifest.layers) > 10000)
    required_20k_dies = ceil_div(min_x1_for_resident, 20000)
    return {
        "model_name": manifest.model_name,
        "system": cfg.as_dict(),
        "summary": {
            "execution_mode": execution_mode,
            "resident_fit": resident_fit,
            "total_static_weight_elements": total_weight_elements,
            "total_programmed_cells": total_stored_cells,
            "unique_static_layer_weights": len(unique_static_layers),
            "scheduled_static_layer_instances": len(static_layers),
            "x1_cell_capacity": cfg.x1_cell_capacity,
            "min_x1_for_full_resident_weights": min_x1_for_resident,
            "required_20k_x1_dies_for_full_resident_weights": required_20k_dies,
            "csr_only_ok_for_bringup": True,
            "dma_recommended_for_practical_streaming": dma_needed or not cfg.dma_available,
            "materialized_tasks": manifest.summary["materialized_tasks"],
            "total_compute_commands": manifest.summary["x1_compute_commands"],
            "total_noc_bits": sum(int(layer["noc_bits"]) for layer in layers),
        },
        "largest_static_layers": largest_layers,
        "notes": [
            "The current Caravel RTL endpoint exposes CSR control and one 128-bit SAR-FP NoC flit from four X1 macros.",
            "A full LLM deployment should stream layer or tile weight windows from peripheral memory unless resident_fit is true.",
            "CSR access is acceptable for early bring-up; DMA/descriptor fetch becomes important once weight windows exceed on-chip capacity.",
        ],
    }


def write_deployment_report(
    manifest: ProgramManifest,
    out_json: str | Path,
    out_md: str | Path | None = None,
    system: X1SystemConfig | None = None,
) -> dict[str, object]:
    report = build_deployment_report(manifest, system)
    json_path = Path(out_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if out_md is not None:
        md_path = Path(out_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        summary = report["summary"]
        lines = [
            f"# X1 LLM Deployment Report: {report['model_name']}",
            "",
            f"- Execution mode: `{summary['execution_mode']}`",
            f"- Resident fit: `{summary['resident_fit']}`",
            f"- X1 macros available: `{report['system']['max_x1_macros']}`",
            f"- Minimum X1 macros for full resident weights: `{summary['min_x1_for_full_resident_weights']}`",
            f"- Required 20k-X1 dies for full resident weights: `{summary['required_20k_x1_dies_for_full_resident_weights']}`",
            f"- DMA recommended for practical streaming: `{summary['dma_recommended_for_practical_streaming']}`",
            "",
            "## Largest Static Layers",
            "",
            "| Layer | Phase | Role | Min X1 by cells | Stored cells |",
            "|---|---|---:|---:|---:|",
        ]
        for layer in report["largest_static_layers"]:
            lines.append(
                f"| {layer['name']} | {layer['phase']} | {layer['role']} | "
                f"{layer['min_x1_by_cells']} | {layer['stored_cells']} |"
            )
        lines.append("")
        lines.extend(f"- {note}" for note in report["notes"])
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
