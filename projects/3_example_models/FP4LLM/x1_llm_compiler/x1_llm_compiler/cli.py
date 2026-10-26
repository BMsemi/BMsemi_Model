from __future__ import annotations

import argparse
import json
from pathlib import Path

from .deployment import X1SystemConfig, write_deployment_report
from .fabric import FabricConfig, StreamPlan, write_stream_plan
from .firmware_driver import write_firmware_driver
from .mapping import parse_mapping_file
from .planner import CompilerConfig, ProgramManifest, compile_model
from .qwen_mapping import qwen_phase_layers
from .weight_image import WeightImageConfig, emit_weight_image, quantize_weight_from_index
from .cost_model import LatencyEnergyEngine
from .mapping_optimizer import (
    MappingOptimizer,
    OptimizationConfig,
    apply_mapping_candidate,
    write_mapping_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile FP4LLM/Qwen weights and schedules for X1 hardware.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    compile_p = sub.add_parser("compile", help="compile FP4LLM prefill/decode mappings")
    compile_p.add_argument("--prefill", type=Path, default=Path("../mappings/Qwen3_1_7B_prefill.m"))
    compile_p.add_argument("--decode", type=Path, default=Path("../mappings/Qwen3_1_7B_decode.m"))
    compile_p.add_argument("--out", type=Path, default=Path("artifacts/qwen3_1_7b_x1_manifest.json"))
    compile_p.add_argument("--model-name", default="Qwen3-1.7B")
    compile_p.add_argument("--max-tasks", type=int, default=20000, help="materialized task cap; use -1 for full task list")
    compile_p.add_argument("--dst", type=int, default=0)
    compile_p.add_argument("--vc", type=int, default=1)

    qwen_p = sub.add_parser("compile-qwen-config", help="compile a Qwen Hugging Face config without a hand-written .m file")
    qwen_p.add_argument("--model-id", default="Qwen/Qwen3-0.6B", help="Hugging Face repo id used to fetch config.json")
    qwen_p.add_argument("--revision", default="main", help="Hugging Face revision/branch/tag for config.json")
    qwen_p.add_argument("--config", type=Path, default=None, help="Optional local config.json or model directory override")
    qwen_p.add_argument("--out", type=Path, default=Path("artifacts/qwen3_0_6b_x1_manifest.json"))
    qwen_p.add_argument("--model-name", default="Qwen3-0.6B")
    qwen_p.add_argument("--seq-len", type=int, default=512)
    qwen_p.add_argument("--decode-context", type=int, default=513)
    qwen_p.add_argument("--max-tasks", type=int, default=20000, help="materialized task cap; use -1 for full task list")
    qwen_p.add_argument("--dst", type=int, default=0)
    qwen_p.add_argument("--vc", type=int, default=1)

    weight_p = sub.add_parser("emit-weight-image", help="emit packed X1 uint32 payload words for one extracted weight tensor")
    weight_p.add_argument("--index", type=Path, required=True, help="index.csv from extract_qwen_layer_weights.py")
    weight_p.add_argument("--layer", type=int, required=True)
    weight_p.add_argument("--role", required=True, help="Role from index.csv, e.g. self_attn.q_proj")
    weight_p.add_argument("--out-bin", type=Path, required=True)
    weight_p.add_argument("--out-json", type=Path, required=True)
    weight_p.add_argument("--weight-bits", type=int, default=7)
    weight_p.add_argument("--block-size", type=int, default=16)
    weight_p.add_argument("--tile-row", type=int, default=-1, help="X1 tile row to emit; -1 emits all rows")
    weight_p.add_argument("--tile-col", type=int, default=-1, help="X1 tile col to emit; -1 emits all cols")
    weight_p.add_argument("--max-tiles", type=int, default=0, help="Optional cap after filtering; 0 means no cap")
    weight_p.add_argument("--skip-zero-cells", action="store_true", help="Skip zero cell writes; only safe after X1 clear")

    report_p = sub.add_parser("capacity-report", help="estimate X1 resident fit and streamed deployment needs")
    report_p.add_argument("--manifest", type=Path, default=Path("artifacts/qwen3_0_6b_x1_manifest.json"))
    report_p.add_argument("--out-json", type=Path, default=Path("artifacts/deployment_report.json"))
    report_p.add_argument("--out-md", type=Path, default=Path("artifacts/deployment_report.md"))
    report_p.add_argument("--max-x1", type=int, default=20000)
    report_p.add_argument("--x1-per-endpoint", type=int, default=4)
    report_p.add_argument("--peripheral-gb", type=float, default=8.0)
    report_p.add_argument("--dma", action="store_true")

    stream_p = sub.add_parser("emit-stream-plan", help="emit DMA/descriptor streaming plan for X1 and memory")
    stream_p.add_argument("--manifest", type=Path, default=Path("artifacts/qwen3_0_6b_x1_manifest.json"))
    stream_p.add_argument("--out-json", type=Path, default=Path("artifacts/qwen3_0_6b_stream_plan.json"))
    stream_p.add_argument("--out-csv", type=Path, default=Path("artifacts/qwen3_0_6b_stream_plan.csv"))
    stream_p.add_argument("--die-count", type=int, default=1)
    stream_p.add_argument("--x1-per-die", type=int, default=20000)
    stream_p.add_argument("--max-descriptors", type=int, default=4096)
    stream_p.add_argument("--exec-per-window", type=int, default=2)
    stream_p.add_argument("--peripheral-gb", type=float, default=8.0)

    firmware_p = sub.add_parser("emit-firmware", help="emit a Caravel management-core firmware descriptor driver")
    firmware_p.add_argument("--stream-plan", type=Path, default=Path("artifacts/qwen3_0_6b_stream_plan.json"))
    firmware_p.add_argument("--out-c", type=Path, default=Path("artifacts/x1_streaming_descriptor_firmware.c"))
    firmware_p.add_argument("--out-json", type=Path, default=Path("artifacts/x1_streaming_descriptor_firmware.json"))
    firmware_p.add_argument("--out-yaml", type=Path, default=None)
    firmware_p.add_argument("--test-name", default="x1_streaming_descriptor_firmware")

    cost_p = sub.add_parser("estimate-cost", help="estimate latency and energy for a manifest and stream plan")
    cost_p.add_argument("--manifest", type=Path, required=True)
    cost_p.add_argument("--stream-plan", type=Path, required=True)
    cost_p.add_argument("--out-json", type=Path, default=Path("artifacts/plan_cost.json"))

    optimize_p = sub.add_parser("optimize-mapping", help="search mapping choices using latency and energy")
    optimize_p.add_argument("--prefill", type=Path, default=Path("../mappings/Qwen3_1_7B_prefill.m"))
    optimize_p.add_argument("--decode", type=Path, default=Path("../mappings/Qwen3_1_7B_decode.m"))
    optimize_p.add_argument("--out-json", type=Path, default=Path("artifacts/mapping_optimization.json"))
    optimize_p.add_argument("--out-manifest", type=Path, default=None)
    optimize_p.add_argument("--out-stream-json", type=Path, default=None)
    optimize_p.add_argument("--out-stream-csv", type=Path, default=None)
    optimize_p.add_argument(
        "--out-prefill-mapping",
        type=Path,
        default=Path("artifacts/best_prefill_mapping.m"),
        help="output optimized prefill .m mapping",
    )
    optimize_p.add_argument(
        "--out-decode-mapping",
        type=Path,
        default=Path("artifacts/best_decode_mapping.m"),
        help="output optimized decode .m mapping",
    )
    optimize_p.add_argument("--model-name", default="Qwen3-1.7B")
    optimize_p.add_argument("--max-tasks", type=int, default=20000, help="materialized task cap; use -1 for full task list")
    optimize_p.add_argument("--max-candidates", type=int, default=64, help="candidate cap; use -1 for exhaustive search")
    optimize_p.add_argument("--latency-weight", type=float, default=1.0)
    optimize_p.add_argument("--energy-weight", type=float, default=1.0)
    optimize_p.add_argument("--die-count", type=int, default=1)
    optimize_p.add_argument("--x1-per-die", type=int, default=20000)
    optimize_p.add_argument("--max-descriptors", type=int, default=4096)

    return parser


def _parse_optional_index(value: int) -> int | None:
    return None if int(value) < 0 else int(value)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "compile":
        max_tasks = None if args.max_tasks < 0 else args.max_tasks
        prefill = parse_mapping_file(args.prefill, "prefill")
        decode = parse_mapping_file(args.decode, "decode")
        cfg = CompilerConfig(
            model_name=args.model_name,
            max_materialized_tasks=max_tasks,
            default_dst=args.dst,
            default_vc=args.vc,
        )
        manifest = compile_model({"prefill": prefill, "decode": decode}, cfg)
        manifest.write_json(args.out)
        print(json.dumps({"out": str(args.out), "summary": manifest.summary}, indent=2))
        return 0

    if args.cmd == "compile-qwen-config":
        max_tasks = None if args.max_tasks < 0 else args.max_tasks
        config_source = args.config if args.config is not None else args.model_id
        cfg = CompilerConfig(
            model_name=args.model_name,
            max_materialized_tasks=max_tasks,
            default_dst=args.dst,
            default_vc=args.vc,
        )
        manifest = compile_model(
            {
                "prefill": qwen_phase_layers(
                    config_source,
                    phase="prefill",
                    seq_len=args.seq_len,
                    decode_context=args.decode_context,
                    revision=args.revision,
                ),
                "decode": qwen_phase_layers(
                    config_source,
                    phase="decode",
                    seq_len=args.seq_len,
                    decode_context=args.decode_context,
                    revision=args.revision,
                ),
            },
            cfg,
        )
        manifest.write_json(args.out)
        print(json.dumps({"config_source": str(config_source), "out": str(args.out), "summary": manifest.summary}, indent=2))
        return 0

    if args.cmd == "emit-weight-image":
        cfg = WeightImageConfig(
            weight_bits=args.weight_bits,
            block_size=args.block_size,
            include_zero_cells=not bool(args.skip_zero_cells),
        )
        quant = quantize_weight_from_index(args.index, layer=args.layer, role=args.role, config=cfg)
        metadata = emit_weight_image(
            quant,
            out_bin=args.out_bin,
            out_json=args.out_json,
            config=cfg,
            tile_row=_parse_optional_index(args.tile_row),
            tile_col=_parse_optional_index(args.tile_col),
            max_tiles=None if int(args.max_tiles) <= 0 else int(args.max_tiles),
        )
        print(json.dumps({"out_bin": str(args.out_bin), "out_json": str(args.out_json), "payload": metadata["payload"]}, indent=2))
        return 0

    if args.cmd == "capacity-report":
        manifest = ProgramManifest.read_json(args.manifest)
        system = X1SystemConfig(
            max_x1_macros=args.max_x1,
            x1_macros_per_endpoint=args.x1_per_endpoint,
            x1_rows=int(manifest.compiler_config.get("x1_macro_rows", 32)),
            x1_cols=int(manifest.compiler_config.get("x1_macro_cols", 32)),
            fp4llm_tile_rows=int(manifest.compiler_config.get("fp4llm_tile_rows", 64)),
            fp4llm_tile_cols=int(manifest.compiler_config.get("fp4llm_tile_cols", 32)),
            stored_cell_banks_per_weight=int(manifest.compiler_config.get("stored_cell_banks_per_weight", 4)),
            peripheral_memory_bytes=int(args.peripheral_gb * 1024**3),
            dma_available=bool(args.dma),
        )
        report = write_deployment_report(manifest, args.out_json, args.out_md, system)
        print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "summary": report["summary"]}, indent=2))
        return 0

    if args.cmd == "emit-stream-plan":
        manifest = ProgramManifest.read_json(args.manifest)
        fabric = FabricConfig(
            die_count=args.die_count,
            x1_macros_per_die=args.x1_per_die,
            x1_macros_per_endpoint=int(manifest.compiler_config.get("x1_macros_per_endpoint", 4)),
            x1_rows=int(manifest.compiler_config.get("x1_macro_rows", 32)),
            x1_cols=int(manifest.compiler_config.get("x1_macro_cols", 32)),
            stored_cell_banks_per_weight=int(manifest.compiler_config.get("stored_cell_banks_per_weight", 4)),
            peripheral_memory_bytes=int(args.peripheral_gb * 1024**3),
            max_descriptors=args.max_descriptors,
            max_execute_descriptors_per_window=args.exec_per_window,
        )
        plan = write_stream_plan(manifest, args.out_json, args.out_csv, fabric)
        print(json.dumps({"out_json": str(args.out_json), "out_csv": str(args.out_csv), "summary": plan.summary}, indent=2))
        return 0

    if args.cmd == "emit-firmware":
        plan = StreamPlan.read_json(args.stream_plan)
        driver = write_firmware_driver(plan, args.out_c, args.out_json, args.out_yaml, test_name=args.test_name)
        print(
            json.dumps(
                {
                    "out_c": str(args.out_c),
                    "out_json": str(args.out_json),
                    "out_yaml": str(args.out_yaml) if args.out_yaml is not None else None,
                    "test_name": driver["test_name"],
                    "descriptor_op_ids": driver["descriptor_op_ids"],
                    "packed_window_words": driver["packed_window_words"],
                    "expected": driver["expected"],
                },
                indent=2,
            )
        )
        return 0

    if args.cmd == "estimate-cost":
        manifest = ProgramManifest.read_json(args.manifest)
        stream_plan = StreamPlan.read_json(args.stream_plan)
        cost = LatencyEnergyEngine().estimate(manifest, stream_plan)
        cost.write_json(args.out_json)
        print(
            json.dumps(
                {
                    "out_json": str(args.out_json),
                    "latency_cycles": cost.latency.total_cycles,
                    "energy_pj": cost.energy.total_pj,
                },
                indent=2,
            )
        )
        return 0

    if args.cmd == "optimize-mapping":
        max_tasks = None if args.max_tasks < 0 else args.max_tasks
        max_candidates = None if args.max_candidates < 0 else args.max_candidates
        prefill = parse_mapping_file(args.prefill, "prefill")
        decode = parse_mapping_file(args.decode, "decode")
        compiler = CompilerConfig(
            model_name=args.model_name,
            max_materialized_tasks=max_tasks,
        )
        fabric = FabricConfig(
            die_count=args.die_count,
            x1_macros_per_die=args.x1_per_die,
            max_descriptors=args.max_descriptors,
        )
        optimizer = MappingOptimizer(
            optimization=OptimizationConfig(
                latency_weight=args.latency_weight,
                energy_weight=args.energy_weight,
                max_candidates=max_candidates,
            )
        )
        result = optimizer.optimize(
            {"prefill": prefill, "decode": decode},
            compiler_config=compiler,
            fabric_config=fabric,
        )
        best_mapped_layers = apply_mapping_candidate(
            {"prefill": prefill, "decode": decode},
            result.best_candidate,
        )
        write_mapping_file(
            best_mapped_layers["prefill"],
            args.out_prefill_mapping,
            candidate=result.best_candidate,
        )
        write_mapping_file(
            best_mapped_layers["decode"],
            args.out_decode_mapping,
            candidate=result.best_candidate,
        )
        result.write_json(args.out_json)
        if args.out_manifest is not None:
            result.best_manifest.write_json(args.out_manifest)
        if args.out_stream_json is not None:
            result.best_stream_plan.write_json(args.out_stream_json)
        if args.out_stream_csv is not None:
            result.best_stream_plan.write_csv(args.out_stream_csv)
        print(
            json.dumps(
                {
                    "out_json": str(args.out_json),
                    "evaluated_candidates": len(result.evaluations),
                    "best_score": result.best_score,
                    "best_candidate": result.best_candidate.as_dict(),
                    "latency_cycles": result.best_cost.latency.total_cycles,
                    "energy_pj": result.best_cost.energy.total_pj,
                    "out_prefill_mapping": str(args.out_prefill_mapping),
                    "out_decode_mapping": str(args.out_decode_mapping),
                },
                indent=2,
            )
        )
        return 0

    parser.error(f"Unhandled command {args.cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
