"""End-to-end mapping, software-crossbar, and hardware-reporting entry point.

This script intentionally keeps the compiler and runtime boundaries visible:
the X1 compiler produces the optimized mappings and analytical hardware plan;
the reusable Qwen ReRAM runtime consumes those mappings
and produces numerical/hardware-effect telemetry for the full decoder stack.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
COMPILER_ROOT = ROOT / "x1_llm_compiler"

if not COMPILER_ROOT.is_dir():
    raise RuntimeError(f"X1 compiler source directory not found: {COMPILER_ROOT}")

try:
    from x1_llm_compiler.x1_llm_compiler.fabric import FabricConfig
    from x1_llm_compiler.x1_llm_compiler.mapping import parse_mapping_file
    from x1_llm_compiler.x1_llm_compiler.mapping_optimizer import (
        MappingOptimizer,
        OptimizationConfig,
        apply_mapping_candidate,
        write_mapping_file,
    )
    from x1_llm_compiler.x1_llm_compiler.planner import CompilerConfig
except ImportError as exc:
    raise ImportError(
        "Could not import the local X1 compiler package. "
        f"Expected modules under {COMPILER_ROOT}."
    ) from exc


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _default_local_root() -> str:
    """Return a portable default for the repository's local model store."""

    return os.environ.get("HF_MODEL_ROOT", str(Path.home() / "hf_models"))


def _model_directory(local_root: str | Path, model: str) -> Path:
    """Resolve the model directory convention used by the runtime runner."""

    model_path = Path(str(model)).expanduser()
    if model_path.exists():
        return model_path.resolve()
    return _path(local_root) / str(model).replace("/", "--")


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class ExecutionArgumentParser(argparse.ArgumentParser):
    def parse_known_args(self, args=None, namespace=None):
        parsed, remaining = super().parse_known_args(args, namespace)
        default_bits = 8 if parsed.execution_target == "fpga6" else 7
        for name in ("weight_bits", "act_bits"):
            if getattr(parsed, name) is None:
                setattr(parsed, name, default_bits)
        return parsed, remaining


def build_parser() -> argparse.ArgumentParser:
    parser = ExecutionArgumentParser(
        description=(
            "Optimize X1 mappings, compile the hardware plan, run the prompt-aware "
            "software crossbar model, or run block-1 prefill on the Qwen6 FPGA overlay."
        )
    )
    parser.add_argument("--prompt", required=True, help="prefill prompt; must match the exported bundle in fpga6 mode")
    parser.add_argument("--execution-target", choices=("software", "fpga6"), default="software",
                        help="fpga6 runs only block 1 prefill on the four LPC + two NLPC overlay")
    parser.add_argument("--fpga-bundle", type=Path, help="block-1 input/weight bundle from export_fpga_bundle.py")
    parser.add_argument("--fpga-bitstream", type=Path, help="new Qwen6 .bit with matching .hwh")
    parser.add_argument("--fpga-driver", type=Path, help="existing tcam_rag_zcu106 axi_x2_mvm_driver.py")
    parser.add_argument("--fpga-timeout", type=float, default=30.0, help="timeout per FPGA command, seconds")
    parser.add_argument("--decode-text", default=" next", help="one decode token/string used after prefill")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--local-root",
        default=_default_local_root(),
        help="directory containing the local Hugging Face model snapshot",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-input-tokens", type=int, default=128)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--all-blocks", action="store_true", help="run all transformer blocks (the default)")
    scope.add_argument("--block-index", type=int, default=None, help="run only this block instead of the full model")

    parser.add_argument("--prefill", type=Path, default=ROOT / "mappings" / "Qwen3_1_7B_prefill.m")
    parser.add_argument("--decode", type=Path, default=ROOT / "mappings" / "Qwen3_1_7B_decode.m")
    parser.add_argument("--max-candidates", type=int, default=64, help="use -1 for exhaustive mapping search")
    parser.add_argument("--latency-weight", type=float, default=1.0)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--die-count", type=int, default=1)
    parser.add_argument("--x1-per-die", type=int, default=20000)
    parser.add_argument("--max-descriptors", type=int, default=4096)

    parser.add_argument("--weight-bits", type=int, default=None, help="default: software 7, fpga6 8")
    parser.add_argument("--act-bits", type=int, default=None, help="default: software 7, fpga6 8")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--crossbar-backend", choices=("ideal_int", "wrapper"), default="wrapper",
                        help="wrapper uses reram_crossbar.py for projections and attention; ideal_int is a reference backend")
    parser.add_argument(
        "--crossbar-execution", choices=("block_streamed", "eager"), default="block_streamed",
        help="Program and release wrapper crossbars one block at a time (default); eager retains all blocks.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--disable-saturation-audit",
        action="store_true",
        help="disable IV/ADC nonlinearity measurements",
    )
    parser.add_argument(
        "--disable-hardware-effects-audit",
        action="store_true",
        help="disable reliability/noise/IV/TDC measurements",
    )

    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "x1_reram_run")
    parser.add_argument("--run-name", default="x1_reram_run")
    return parser


def optimize_mapping(args: argparse.Namespace, out_dir: Path) -> tuple[Any, dict[str, Any]]:
    prefill = parse_mapping_file(_path(args.prefill), "prefill")
    decode = parse_mapping_file(_path(args.decode), "decode")
    phase_layers = {"prefill": prefill, "decode": decode}

    max_candidates = None if int(args.max_candidates) < 0 else int(args.max_candidates)
    optimizer = MappingOptimizer(
        optimization=OptimizationConfig(
            latency_weight=float(args.latency_weight),
            energy_weight=float(args.energy_weight),
            max_candidates=max_candidates,
        )
    )
    result = optimizer.optimize(
        phase_layers,
        compiler_config=CompilerConfig(model_name=args.model),
        fabric_config=FabricConfig(
            die_count=int(args.die_count),
            x1_macros_per_die=int(args.x1_per_die),
            max_descriptors=int(args.max_descriptors),
        ),
    )

    mapped = apply_mapping_candidate(phase_layers, result.best_candidate)
    prefill_mapping = out_dir / "best_prefill_mapping.m"
    decode_mapping = out_dir / "best_decode_mapping.m"
    write_mapping_file(mapped["prefill"], prefill_mapping, candidate=result.best_candidate)
    write_mapping_file(mapped["decode"], decode_mapping, candidate=result.best_candidate)

    result.best_manifest.write_json(out_dir / "best_manifest.json")
    result.best_stream_plan.write_json(out_dir / "best_stream_plan.json")
    result.best_stream_plan.write_csv(out_dir / "best_stream_plan.csv")
    result.write_json(out_dir / "mapping_optimization.json")

    return result, {
        "prefill": prefill_mapping,
        "decode": decode_mapping,
        "manifest": out_dir / "best_manifest.json",
        "stream_plan": out_dir / "best_stream_plan.json",
        "optimization": out_dir / "mapping_optimization.json",
    }


def validate_inputs(args: argparse.Namespace) -> None:
    """Fail before optimization when the runtime cannot execute the request."""

    if not _path(args.prefill).is_file():
        raise FileNotFoundError(f"Prefill mapping file not found: {_path(args.prefill)}")
    if not _path(args.decode).is_file():
        raise FileNotFoundError(f"Decode mapping file not found: {_path(args.decode)}")
    model_dir = _model_directory(args.local_root, args.model)
    if not model_dir.is_dir():
        raise FileNotFoundError(
            "Local model directory not found. Expected either a direct model path or "
            f"{model_dir}. Set HF_MODEL_ROOT or pass --local-root."
        )
    if int(args.max_input_tokens) < 1:
        raise ValueError("--max-input-tokens must be at least 1.")
    if args.block_index is not None and int(args.block_index) < 0:
        raise ValueError("--block-index must be non-negative.")
    if int(args.max_candidates) == 0 or int(args.max_candidates) < -1:
        raise ValueError("--max-candidates must be positive or -1 for exhaustive search.")
    if int(args.die_count) < 1 or int(args.x1_per_die) < 1:
        raise ValueError("--die-count and --x1-per-die must be positive.")
    if int(args.max_descriptors) < 1:
        raise ValueError("--max-descriptors must be positive.")
    if int(args.weight_bits) < 1 or int(args.act_bits) < 1 or int(args.block_size) < 1:
        raise ValueError("--weight-bits, --act-bits, and --block-size must be positive.")


def run_prompt_crossbar(
    args: argparse.Namespace,
    mapping_paths: dict[str, Path],
    out_dir: Path,
) -> Path:
    from src import qwen_crossbar_runtime

    runtime_dir = out_dir / "software_crossbar"
    runtime_args = [
        "--run-name",
        str(args.run_name),
        "--out-dir",
        str(runtime_dir),
        "--model",
        str(args.model),
        "--local-root",
        str(args.local_root),
        "--dtype",
        str(args.dtype),
        "--device",
        str(args.device),
        "--max-input-tokens",
        str(args.max_input_tokens),
        "--prefill-mapping",
        str(mapping_paths["prefill"]),
        "--decode-mapping",
        str(mapping_paths["decode"]),
        "--prompt",
        str(args.prompt),
        "--decode-text",
        str(args.decode_text),
        "--weight-bits",
        str(args.weight_bits),
        "--act-bits",
        str(args.act_bits),
        "--block-size",
        str(args.block_size),
        "--crossbar-backend",
        str(args.crossbar_backend),
        "--crossbar-execution",
        str(args.crossbar_execution),
        "--seed",
        str(args.seed),
    ]
    if args.block_index is None:
        runtime_args.append("--all-blocks")
    else:
        runtime_args.extend(["--block-index", str(args.block_index)])
    if args.local_files_only:
        runtime_args.append("--local-files-only")
    runtime_args.append(
        "--disable-saturation-audit" if args.disable_saturation_audit else "--enable-saturation-audit"
    )
    runtime_args.append(
        "--disable-hardware-effects-audit"
        if args.disable_hardware_effects_audit
        else "--enable-hardware-effects-audit"
    )

    print("\n=== Running prompt-aware software crossbar ===")
    qwen_crossbar_runtime.run(qwen_crossbar_runtime.parse_args(runtime_args))

    metadata_path = runtime_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Software-crossbar run did not produce {metadata_path}")
    return metadata_path


def build_report(
    args: argparse.Namespace,
    result: Any,
    metadata: dict[str, Any],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    linear_modules = metadata.get("linear_crossbar_modules", [])
    attention_modules = metadata.get("mapped_attention_gemm_records", [])
    if not isinstance(linear_modules, list):
        linear_modules = []
    if not isinstance(attention_modules, list):
        attention_modules = []

    linear_pe_tiles = sum(_int(row.get("mapping_crossbar_tiles")) for row in linear_modules)
    attention_pe_tiles = sum(
        _int(row.get("crossbar_tiles"), _int(row.get("row_tiles")) * _int(row.get("col_tiles")))
        for row in attention_modules
    )
    logical_pe_tiles = int(linear_pe_tiles + attention_pe_tiles)

    prefill_shape = metadata.get("prefill_input_shape", [1, 0])
    decode_shape = metadata.get("decode_input_shape", [1, 0])
    prefill_tokens = _int(prefill_shape[-1] if prefill_shape else 0)
    decode_tokens = _int(decode_shape[-1] if decode_shape else 0)

    runtime_macs = sum(
        (_int(row.get("in_features")) * _int(row.get("out_features")))
        * (prefill_tokens + decode_tokens)
        for row in linear_modules
    )
    runtime_macs += sum(_int(row.get("macs")) for row in attention_modules)

    # This is a logical tile estimate. Physical macro usage depends on the
    # selected X1 bank/endpoint organization and is reported separately by the
    # compiled manifest and stream plan.
    row_shards = max(1, math.ceil(int(result.best_candidate.fp4llm_tile_rows) / 32))
    available_logical_tiles = max(1, int(args.x1_per_die) // row_shards)
    pe_utilization = min(1.0, logical_pe_tiles / float(available_logical_tiles))

    saturation = metadata.get("saturation_summary", {})
    saturation_total = saturation.get("total", {}) if isinstance(saturation, dict) else {}
    effects = metadata.get("hardware_effect_summary", {})
    effects_total = effects.get("total", {}) if isinstance(effects, dict) else {}
    effects_by_phase_role = effects.get("by_phase_role", {}) if isinstance(effects, dict) else {}
    if not isinstance(effects_by_phase_role, dict):
        effects_by_phase_role = {}

    def tdc_timing_view(stats: dict[str, Any]) -> dict[str, Any]:
        histogram = stats.get("tdc_first_ready_column_histogram", {})
        if not isinstance(histogram, dict):
            histogram = {}
        return {
            "enabled": _int(stats.get("tdc_timing_samples")) > 0,
            "samples": _int(stats.get("tdc_timing_samples")),
            "first_ready_column": _int(stats.get("tdc_first_ready_column"), -1),
            "first_ready_column_histogram": histogram,
            "first_ready_arrival_min_ps": _float(stats.get("tdc_first_ready_arrival_min_ps")),
            "first_ready_arrival_max_ps": _float(stats.get("tdc_first_ready_arrival_max_ps")),
            "arrival_spread_max_ps": _float(stats.get("tdc_arrival_spread_max_ps")),
        }

    tdc_timing = tdc_timing_view(effects_total)
    tdc_timing["by_phase_role"] = {
        str(role): tdc_timing_view(stats)
        for role, stats in effects_by_phase_role.items()
        if isinstance(stats, dict)
    }
    saturation_config = metadata.get("saturation_audit_config", {})
    if not isinstance(saturation_config, dict):
        saturation_config = {}

    cost = result.best_cost
    manifest_summary = result.best_manifest.summary
    stream_summary = dict(result.best_stream_plan.summary)
    static_cells = _int(stream_summary.get("total_static_programmed_cells"))
    cell_capacity = _int(stream_summary.get("total_cell_capacity"))
    descriptor_count = _int(stream_summary.get("descriptors"))
    descriptor_capacity = max(1, int(args.max_descriptors))
    report = {
        "prompt": {
            "text": str(metadata.get("prefill_prompt", args.prompt)),
            "prefill_input_ids": metadata.get("prefill_input_ids", []),
            "decode_text": metadata.get("decode_text", args.decode_text),
            "decode_input_ids": metadata.get("decode_input_ids", []),
        },
        "mapping": {
            "best_candidate": result.best_candidate.as_dict(),
            "best_score": result.best_score,
            "evaluated_candidates": len(result.evaluations),
            "prefill_mapping": str(output_paths["prefill"]),
            "decode_mapping": str(output_paths["decode"]),
            "manifest": str(output_paths["manifest"]),
            "stream_plan": str(output_paths["stream_plan"]),
            "compiled_manifest_summary": manifest_summary,
            "stream_plan_summary": stream_summary,
        },
        "compiler_estimate": {
            "scope": "all layers represented by the prefill/decode mapping files",
            "latency_cycles": cost.latency.total_cycles,
            "energy_pj": cost.energy.total_pj,
            "latency_breakdown": cost.latency.as_dict(),
            "energy_breakdown": cost.energy.as_dict(),
            "counters": dict(cost.counters),
        },
        "software_crossbar_run": {
            "scope": metadata.get("execution_scope", "selected transformer block"),
            "executed_block_count": metadata.get("executed_block_count", 1),
            "executed_block_indices": metadata.get("executed_block_indices", [args.block_index]),
            "crossbar_backend": metadata.get("crossbar_backend"),
            "crossbar_implementation": metadata.get("crossbar_implementation"),
            "crossbar_execution": metadata.get("crossbar_execution"),
            "crossbar_tile_lifecycle": metadata.get("crossbar_tile_lifecycle"),
            "forward_hardware_effects": metadata.get("forward_hardware_effects", {}),
            "framework_operations": metadata.get("framework_operations", []),
            "decode_input_mode": metadata.get("decode_input_mode", "supplied_text"),
            "model": metadata.get("model"),
            "block_index": metadata.get("block_index_zero_based"),
            "prefill_seconds": _float(metadata.get("elapsed_prefill_s")),
            "decode_seconds": _float(metadata.get("elapsed_decode_s")),
            "total_seconds": _float(metadata.get("elapsed_prefill_s"))
            + _float(metadata.get("elapsed_decode_s")),
            "input_shape": metadata.get("prefill_input_shape"),
            "prefill_next_token_id": metadata.get("prefill_next_token_id", metadata.get("selected_block_prefill_next_token_id")),
            "prefill_next_token_text": metadata.get("prefill_next_token_text", metadata.get("selected_block_prefill_next_token_text", "")),
            "decode_next_token_id": metadata.get("decode_next_token_id", metadata.get("selected_block_decode_next_token_id")),
            "decode_next_token_text": metadata.get("decode_next_token_text", metadata.get("selected_block_decode_next_token_text", "")),
            "software_crossbar_metadata": str(output_paths["runtime_metadata"]),
        },
        "hardware_utilization": {
            "compiled_fp4llm_tile_tasks": _int(manifest_summary.get("fp4llm_tile_tasks")),
            "logical_pe_tiles_used": logical_pe_tiles,
            "linear_logical_pe_tiles": linear_pe_tiles,
            "attention_logical_pe_tiles": attention_pe_tiles,
            "compiled_static_programmed_cells": static_cells,
            "compiled_fabric_cell_capacity": cell_capacity,
            "compiled_cell_utilization_percent": (
                100.0 * static_cells / float(cell_capacity) if cell_capacity > 0 else 0.0
            ),
            "compiled_descriptors": descriptor_count,
            "descriptor_capacity": descriptor_capacity,
            "descriptor_utilization_percent": 100.0 * descriptor_count / descriptor_capacity,
            "resident_feasible": bool(stream_summary.get("resident_feasible", False)),
            "estimated_available_logical_tiles": available_logical_tiles,
            "estimated_pe_utilization_percent": 100.0 * pe_utilization,
            "runtime_macs_executed_blocks": int(runtime_macs),
            "x1_macros_per_die": int(args.x1_per_die),
            "note": "PE utilization is a logical tile estimate; physical bank allocation is in the stream plan.",
        },
        "nonlinearity_and_hardware_effects": {
            "iv_nonlinearity_enabled": bool(
                saturation_config.get("enabled", False)
            ),
            "iv_mode": saturation_config.get("iv_nonlinearity_mode"),
            "iv_upper_saturation_events": _int(saturation_total.get("iv_upper_saturation_events")),
            "iv_lower_clip_events": _int(saturation_total.get("iv_lower_clip_events")),
            "iv_saturated_tiles": _int(saturation_total.get("iv_saturated_tiles")),
            "iv_clamped_column_fraction": _float(saturation_total.get("iv_clamped_column_fraction")),
            "iv_max_branch_level": _float(saturation_total.get("iv_max_branch_level")),
            "reliability_changed_cells": _int(effects_total.get("reliability_changed_cells")),
            "effective_weight_changed_entries": _int(effects_total.get("effective_weight_changed_entries")),
            "noise_mean_branch_sigma_levels": _float(effects_total.get("noise_mean_branch_sigma_levels")),
            "noise_max_branch_sigma_levels": _float(effects_total.get("noise_max_branch_sigma_levels")),
            "tdc_clipped_branch_conversions": _int(effects_total.get("tdc_clipped_branch_conversions")),
            "tdc_mean_abs_int_error": _float(effects_total.get("tdc_mean_abs_int_error")),
            "tdc_max_abs_int_error": _float(effects_total.get("tdc_max_abs_int_error")),
            "tdc_timing": tdc_timing,
        },
    }
    return report


def print_report(report: dict[str, Any]) -> None:
    prompt = report["prompt"]
    mapping = report["mapping"]
    estimate = report["compiler_estimate"]
    runtime = report["software_crossbar_run"]
    utilization = report["hardware_utilization"]
    effects = report["nonlinearity_and_hardware_effects"]
    tdc_timing = effects["tdc_timing"]

    print("\n=== X1 LLM end-to-end summary ===")
    print(f"Prompt: {prompt['text']}")
    input_ids = prompt["prefill_input_ids"]
    print(f"Prefill token count: {len(input_ids[0]) if input_ids and isinstance(input_ids[0], list) else len(input_ids)}")
    print(f"Decode stimulus: {prompt['decode_text']!r} -> {prompt['decode_input_ids']}")
    print(f"Best mapping score: {mapping['best_score']:.6f}")
    print(f"Best prefill .m: {mapping['prefill_mapping']}")
    print(f"Best decode .m: {mapping['decode_mapping']}")
    print(f"Compiled PE tile tasks (all mapped layers): {utilization['compiled_fp4llm_tile_tasks']:,}")
    print(f"Estimated latency: {estimate['latency_cycles']:,} cycles")
    print(f"Estimated energy: {estimate['energy_pj']:,.3f} pJ")
    print(f"Software crossbar wall time: {runtime['total_seconds']:.4f} s")
    print(
        f"Greedy output ({runtime['executed_block_count']} blocks): "
        f"prefill -> {runtime['prefill_next_token_text']!r}; "
        f"decode -> {runtime['decode_next_token_text']!r}"
    )
    print(f"Logical PE/crossbar tiles used: {utilization['logical_pe_tiles_used']:,}")
    print(
        "Estimated PE utilization: "
        f"{utilization['estimated_pe_utilization_percent']:.3f}% "
        f"({utilization['estimated_available_logical_tiles']:,} logical tiles available)"
    )
    print(
        "Compiled fabric cell utilization: "
        f"{utilization['compiled_cell_utilization_percent']:.3f}% "
        f"({utilization['compiled_static_programmed_cells']:,} / "
        f"{utilization['compiled_fabric_cell_capacity']:,} cells)"
    )
    print(
        "Descriptor utilization: "
        f"{utilization['descriptor_utilization_percent']:.3f}% "
        f"({utilization['compiled_descriptors']:,} / {utilization['descriptor_capacity']:,})"
    )
    print("\nNonlinearity and hardware effects:")
    print(f"  IV/ADC mode: {effects['iv_mode']}")
    print(f"  IV upper saturation events: {effects['iv_upper_saturation_events']:,}")
    print(f"  IV saturated tiles: {effects['iv_saturated_tiles']:,}")
    print(f"  IV clamped-column fraction: {effects['iv_clamped_column_fraction']:.6f}")
    print(f"  Noise mean/max sigma: {effects['noise_mean_branch_sigma_levels']:.6f} / {effects['noise_max_branch_sigma_levels']:.6f}")
    print(f"  Reliability-changed cells: {effects['reliability_changed_cells']:,}")
    print(f"  TDC clipped conversions: {effects['tdc_clipped_branch_conversions']:,}")
    print(f"  TDC mean/max integer error: {effects['tdc_mean_abs_int_error']:.6f} / {effects['tdc_max_abs_int_error']:.6f}")
    print("\nTDC timing / first-ready columns:")
    if tdc_timing["enabled"]:
        print(
            "  Most frequent first-ready logical column: "
            f"{tdc_timing['first_ready_column']}"
        )
        print(f"  Timing samples: {tdc_timing['samples']:,}")
        print(
            "  First-ready arrival range: "
            f"{tdc_timing['first_ready_arrival_min_ps']:.3f} - "
            f"{tdc_timing['first_ready_arrival_max_ps']:.3f} ps"
        )
        print(f"  Maximum arrival spread: {tdc_timing['arrival_spread_max_ps']:.3f} ps")
        for phase_role, role_timing in sorted(tdc_timing["by_phase_role"].items()):
            if role_timing["enabled"]:
                print(
                    f"  {phase_role}: first-ready column {role_timing['first_ready_column']} "
                    f"({role_timing['samples']:,} samples)"
                )
    else:
        print("  unavailable (hardware-effects/TDC timing audit disabled or no samples recorded)")
    print(f"\nSoftware-crossbar scope: {runtime['scope']} ({runtime['executed_block_count']} blocks).")
    print(f"Framework operations: {', '.join(runtime['framework_operations'])}")
    print("Decode uses the supplied --decode-text token; this run is not a free-running generation loop.")
    print(f"Detailed report: {report['software_crossbar_run']['software_crossbar_metadata']}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.execution_target == "fpga6":
        from src.qwen_fpga6 import run_from_args
        try:
            if not math.isfinite(args.fpga_timeout) or args.fpga_timeout <= 0:
                raise ValueError("--fpga-timeout must be positive and finite")
            return run_from_args(args)
        except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
            parser.error(str(exc))
    try:
        validate_inputs(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    out_dir = _path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result, output_paths = optimize_mapping(args, out_dir)
    metadata_path = run_prompt_crossbar(args, output_paths, out_dir)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    output_paths["runtime_metadata"] = metadata_path

    report = build_report(args, result, metadata, output_paths)
    report_path = out_dir / "main_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_report(report)
    print(f"\nCombined report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
