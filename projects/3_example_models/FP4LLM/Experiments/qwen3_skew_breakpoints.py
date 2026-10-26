from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from Experiments.qwen3_feature_effects import (  # noqa: E402
    DEFAULT_KV_CACHE_CONTEXT_TOKENS,
    DEFAULT_KV_CACHE_SERVICE_LIFE_YEARS,
    DEFAULT_KV_CACHE_TOKEN_TIME_S,
    DEFAULT_DYNAMIC_KV_ENDURANCE_CYCLES,
    DEFAULT_DYNAMIC_MLP_ENDURANCE_CYCLES,
    DEFAULT_MLP_DYNAMIC_RETENTION_TIME_S,
    DEFAULT_WEIGHT_RETENTION_TIME_S,
    KV_CACHE_RETENTION_STRATEGIES,
    MODEL_ID,
    EvalCase,
    FeatureCase,
    TinyCharTokenizer,
    default_eval_cases,
    load_qwen_weight_slice,
    mapping_projection,
    mean_abs_delta,
    resolve_snapshot,
    run_feature_case,
)
from src.qwen_tensor_select import (  # noqa: E402
    DEFAULT_QWEN_TENSOR_ROLE,
    qwen_tensor_role_choices,
    qwen_tensor_selection_metadata,
)


Row = dict[str, Any]
CaseFactory = Callable[[float], FeatureCase]


@dataclass(frozen=True)
class SweepSpec:
    name: str
    group: str
    parameter: str
    unit: str
    description: str
    values: tuple[float, ...]
    stress_direction: str
    make_case: CaseFactory


def base_noise(**overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "enabled": True,
        "vread": 0.1,
        "temp_k": 0.0,
        "fano": 0.0,
        "r_lrs_ohm": 100_000.0,
        "r_hrs_ohm": 10_000_000.0,
        "flicker_af_lrs": 0.0,
        "flicker_af_hrs": 0.0,
        "flicker_gamma_lrs": 1.0,
        "flicker_gamma_hrs": 1.0,
        "rtn_amp_frac_lrs": 0.0,
        "rtn_amp_frac_hrs": 0.0,
        "readout_noise_a_rms": 0.0,
        "bandwidth_hz": 1.0e9,
        "flicker_fmin_hz": 0.5,
    }
    cfg.update(overrides)
    return cfg


def reliability_case(
    *,
    name: str,
    description: str,
    seed: int,
    **reliability: Any,
) -> FeatureCase:
    return FeatureCase(
        name=name,
        group="reliability",
        description=description,
        reliability={"enabled": True, **reliability},
        reliability_seed=seed,
    )


def noise_case(
    *,
    name: str,
    description: str,
    seed: int,
    **noise: Any,
) -> FeatureCase:
    return FeatureCase(
        name=name,
        group="noise",
        description=description,
        noise=base_noise(**noise),
        noise_seed=seed,
    )


def sweep_specs(*, kv_cache_token_time_s: float = DEFAULT_KV_CACHE_TOKEN_TIME_S) -> list[SweepSpec]:
    return [
        SweepSpec(
            name="adc_linear_rows_before_clamp",
            group="adc",
            parameter="iv.linear_rows_before_clamp",
            unit="rows",
            description="ADC/I-V hard clamp severity. Smaller values clip column sums earlier.",
            values=(64, 32, 16, 8, 4, 2, 1),
            stress_direction="lower",
            make_case=lambda value: FeatureCase(
                name=f"adc_linear_rows_before_clamp={value:g}",
                group="adc",
                description="Column-current accumulation is hard-clamped before ADC quantization.",
                iv={"enabled": True, "mode": "hard_clamp", "linear_rows_before_clamp": int(value)},
            ),
        ),
        SweepSpec(
            name="stuck_at_lrs_prob",
            group="reliability",
            parameter="reliability.stuck_at_lrs_prob",
            unit="probability",
            description="Cells forced to the LRS/high-conductance state.",
            values=(0.0, 1.0e-5, 1.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2),
            stress_direction="higher",
            make_case=lambda value: reliability_case(
                name=f"stuck_at_lrs_prob={value:g}",
                description="Programmed cells have a probability of being stuck at LRS.",
                seed=2101,
                stuck_at_lrs_prob=value,
            ),
        ),
        SweepSpec(
            name="stuck_at_hrs_prob",
            group="reliability",
            parameter="reliability.stuck_at_hrs_prob",
            unit="probability",
            description="Cells forced to the HRS/low-conductance state.",
            values=(0.0, 1.0e-5, 1.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2),
            stress_direction="higher",
            make_case=lambda value: reliability_case(
                name=f"stuck_at_hrs_prob={value:g}",
                description="Programmed cells have a probability of being stuck at HRS.",
                seed=2102,
                stuck_at_hrs_prob=value,
            ),
        ),
        SweepSpec(
            name="raw_ber_no_ecc_ppm",
            group="reliability",
            parameter="reliability.raw_ber_ppm",
            unit="ppm",
            description="Raw cell bit-error rate with ECC disabled.",
            values=(0.0, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0),
            stress_direction="higher",
            make_case=lambda value: reliability_case(
                name=f"raw_ber_no_ecc_ppm={value:g}",
                description="Raw cell bit errors are injected without ECC.",
                seed=3101,
                raw_ber_ppm=value,
                use_ecc=False,
            ),
        ),
        SweepSpec(
            name="raw_ber_with_ecc_ppm",
            group="reliability",
            parameter="reliability.raw_ber_ppm",
            unit="ppm",
            description="Raw cell bit-error rate passed through the optional ECC model.",
            values=(0.0, 100.0, 1_000.0, 10_000.0, 30_000.0, 100_000.0, 300_000.0),
            stress_direction="higher",
            make_case=lambda value: reliability_case(
                name=f"raw_ber_with_ecc_ppm={value:g}",
                description="Raw cell bit errors are injected with single-error correction enabled.",
                seed=3101,
                raw_ber_ppm=value,
                use_ecc=True,
            ),
        ),
        SweepSpec(
            name="retention_drift_per_decade",
            group="reliability",
            parameter="reliability.retention_log10_resistance_drift_per_decade",
            unit="log10(R)/decade",
            description="Retention drift coefficient at 30 days and 85 C.",
            values=(0.0, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1),
            stress_direction="higher",
            make_case=lambda value: reliability_case(
                name=f"retention_drift_per_decade={value:g}",
                description="Log-resistance retention drift over 30 days at elevated temperature.",
                seed=4101,
                retention_time_s=30.0 * 24.0 * 3600.0,
                retention_reference_s=1.0,
                retention_temp_c=85.0,
                retention_temp_reference_c=25.0,
                retention_temp_acceleration_per_100c=2.0,
                retention_log10_resistance_drift_per_decade=value,
            ),
        ),
        SweepSpec(
            name="kv_cache_retention_context_tokens",
            group="reliability",
            parameter="retention_calibration.kv_cache_context_tokens",
            unit="tokens",
            description="K/V-cache retention residency at the configured token cadence.",
            values=(0.0, 1.0, 16.0, 128.0, 512.0, 2048.0, 4096.0, 8192.0),
            stress_direction="higher",
            make_case=lambda value: reliability_case(
                name=f"kv_cache_retention_context_tokens={value:g}",
                description="Retention drift with K/V-cache residency rather than weight-stationary storage time.",
                seed=4105,
                retention_time_s=max(float(value), 0.0) * max(float(kv_cache_token_time_s), 0.0),
                retention_reference_s=1.0,
                retention_temp_c=85.0,
                retention_temp_reference_c=25.0,
                retention_temp_acceleration_per_100c=2.0,
                retention_log10_resistance_drift_per_decade=0.03,
            ),
        ),
        SweepSpec(
            name="relaxation_amplitude_ln",
            group="reliability",
            parameter="reliability.relaxation_resistance_amplitude_ln",
            unit="ln(R)",
            description="Short-term relaxation amplitude after one hour.",
            values=(0.0, 1.0e-2, 3.0e-2, 1.0e-1, 2.0e-1, 4.0e-1, 8.0e-1),
            stress_direction="higher",
            make_case=lambda value: reliability_case(
                name=f"relaxation_amplitude_ln={value:g}",
                description="Short-term post-programming relaxation drift.",
                seed=4102,
                relaxation_time_s=3600.0,
                relaxation_tau_s=2000.0,
                relaxation_resistance_amplitude_ln=value,
            ),
        ),
        SweepSpec(
            name="state_sigma_ln",
            group="reliability",
            parameter="reliability.state_sigma_ln",
            unit="ln(R) sigma",
            description="Per-cell programmed state / conductance variation.",
            values=(0.0, 5.0e-3, 1.0e-2, 3.0e-2, 8.0e-2, 1.5e-1, 3.0e-1),
            stress_direction="higher",
            make_case=lambda value: reliability_case(
                name=f"state_sigma_ln={value:g}",
                description="Per-cell state spread through state_sigma_ln.",
                seed=4103,
                state_sigma_ln=value,
            ),
        ),
        SweepSpec(
            name="endurance_life_fraction",
            group="reliability",
            parameter="reliability.endurance_write_cycles / endurance_cycles",
            unit="life_fraction",
            description="Consumed write-cycle fraction with drift amplitude fixed at ln(R)=0.4 per full lifetime.",
            values=(0.0, 1.0e-2, 1.0e-1, 2.5e-1, 5.0e-1, 8.0e-1, 1.0),
            stress_direction="higher",
            make_case=lambda value: reliability_case(
                name=f"endurance_life_fraction={value:g}",
                description="Crossbar conductance drift from endurance wear.",
                seed=4104,
                endurance_write_cycles=int(round(value * 100_000)),
                endurance_cycles=100_000,
                endurance_resistance_drift_ln_per_life_fraction=0.40,
            ),
        ),
        SweepSpec(
            name="thermal_temp_k",
            group="noise",
            parameter="noise.temp_k",
            unit="K",
            description="Thermal white-noise temperature with 1 GHz read bandwidth.",
            values=(0.0, 300.0, 500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0),
            stress_direction="higher",
            make_case=lambda value: noise_case(
                name=f"thermal_temp_k={value:g}",
                description="Thermal white noise only, integrated before ADC.",
                seed=5101,
                temp_k=value,
                bandwidth_hz=1.0e9,
            ),
        ),
        SweepSpec(
            name="shot_fano",
            group="noise",
            parameter="noise.fano",
            unit="Fano factor",
            description="Shot-noise Fano factor with 1 GHz read bandwidth.",
            values=(0.0, 1.0e-1, 1.0, 3.0, 10.0, 30.0, 100.0),
            stress_direction="higher",
            make_case=lambda value: noise_case(
                name=f"shot_fano={value:g}",
                description="Shot noise only, integrated before ADC.",
                seed=5102,
                fano=value,
                bandwidth_hz=1.0e9,
            ),
        ),
        SweepSpec(
            name="flicker_af_scale",
            group="noise",
            parameter="noise.flicker_af_lrs/hrs scale",
            unit="scale",
            description="1/f noise scale; LRS uses 1e-6*scale, HRS uses 1e-4*scale.",
            values=(0.0, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1_000.0),
            stress_direction="higher",
            make_case=lambda value: noise_case(
                name=f"flicker_af_scale={value:g}",
                description="Flicker / 1/f noise only, integrated before ADC.",
                seed=5103,
                flicker_af_lrs=1.0e-6 * value,
                flicker_af_hrs=1.0e-4 * value,
                bandwidth_hz=1.0e6,
            ),
        ),
        SweepSpec(
            name="rtn_amp_frac_hrs",
            group="noise",
            parameter="noise.rtn_amp_frac_hrs",
            unit="fraction",
            description="RTN amplitude fraction; LRS amplitude is set to one eighth of HRS.",
            values=(0.0, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1),
            stress_direction="higher",
            make_case=lambda value: noise_case(
                name=f"rtn_amp_frac_hrs={value:g}",
                description="Random telegraph noise only, converted to INT4-level perturbation before ADC.",
                seed=5104,
                rtn_amp_frac_lrs=value / 8.0,
                rtn_amp_frac_hrs=value,
            ),
        ),
        SweepSpec(
            name="readout_noise_a_rms",
            group="noise",
            parameter="noise.readout_noise_a_rms",
            unit="A RMS",
            description="Readout RMS current noise converted to INT4-level perturbation before ADC.",
            values=(0.0, 1.0e-10, 3.0e-10, 1.0e-9, 3.0e-9, 1.0e-8, 3.0e-8),
            stress_direction="higher",
            make_case=lambda value: noise_case(
                name=f"readout_noise_a_rms={value:g}",
                description="Readout RMS current noise only, converted to INT4-level perturbation before ADC.",
                seed=5105,
                readout_noise_a_rms=value,
                bandwidth_hz=1.0e9,
            ),
        ),
        SweepSpec(
            name="combined_noise_scale",
            group="noise",
            parameter="noise.combined_scale",
            unit="scale",
            description="Combined thermal, shot, flicker, RTN, and readout noise multiplier.",
            values=(0.0, 1.0e-1, 3.0e-1, 1.0, 3.0, 10.0, 30.0),
            stress_direction="higher",
            make_case=lambda value: FeatureCase(
                name=f"combined_noise_scale={value:g}",
                group="noise",
                description="Combined read-noise terms before ADC rounding.",
                noise={
                    "enabled": True,
                    "vread": 0.1,
                    "temp_k": 400.0 * value,
                    "fano": 1.0 * value,
                    "r_lrs_ohm": 100_000.0,
                    "r_hrs_ohm": 10_000_000.0,
                    "flicker_af_lrs": 1.0e-5 * value,
                    "flicker_af_hrs": 1.0e-3 * value,
                    "rtn_amp_frac_lrs": min(1.0, 0.02 * value),
                    "rtn_amp_frac_hrs": min(1.0, 0.20 * value),
                    "readout_noise_a_rms": 2.5e-8 * value,
                    "bandwidth_hz": 2.5e8,
                    "flicker_fmin_hz": 0.5,
                },
                noise_seed=5106,
            ),
        ),
    ]


def select_specs(all_specs: list[SweepSpec], names: str) -> list[SweepSpec]:
    if names.strip().lower() in {"", "all"}:
        return all_specs
    requested = {item.strip() for item in names.split(",") if item.strip()}
    selected = [spec for spec in all_specs if spec.name in requested]
    missing = sorted(requested - {spec.name for spec in selected})
    if missing:
        raise ValueError(f"Unknown sweep spec(s): {', '.join(missing)}")
    return selected


def format_number(value: Any) -> str:
    if value is None or value == "":
        return ""
    value_f = float(value)
    if value_f == 0.0:
        return "0"
    if abs(value_f) >= 1.0e4 or abs(value_f) < 1.0e-3:
        return f"{value_f:.3e}"
    return f"{value_f:.6g}"


def safe_range_text(spec: SweepSpec, boundary: float | None, first_break: float | None) -> str:
    if boundary is None:
        return "no non-breaking point observed"
    if first_break is None:
        return f"no break through {format_number(boundary)} {spec.unit}".strip()
    op = ">=" if spec.stress_direction == "lower" else "<="
    return f"{spec.parameter} {op} {format_number(boundary)} {spec.unit}".strip()


def break_reason(row: Row, args: argparse.Namespace) -> str:
    reasons: list[str] = []
    if row["delta_nll"] >= args.break_nll_delta:
        reasons.append(f"delta_nll>={args.break_nll_delta:g}")
    if row["ppl_rel_increase"] >= args.break_ppl_rel:
        reasons.append(f"ppl_rel>={args.break_ppl_rel:g}")
    if row["mean_abs_logit_delta_vs_baseline"] >= args.break_logit_delta:
        reasons.append(f"logit_delta>={args.break_logit_delta:g}")
    if row["greedy_drop"] >= args.break_greedy_drop:
        reasons.append(f"greedy_drop>={args.break_greedy_drop:g}")
    return ";".join(reasons)


def annotate_row(
    row: Row,
    *,
    baseline: Row,
    baseline_logits: list[np.ndarray],
    logits: list[np.ndarray],
    spec: SweepSpec | None,
    value: float,
    args: argparse.Namespace,
) -> Row:
    out = dict(row)
    out["sweep"] = spec.name if spec else "baseline"
    out["parameter"] = spec.parameter if spec else ""
    out["parameter_value"] = value
    out["parameter_unit"] = spec.unit if spec else ""
    out["stress_direction"] = spec.stress_direction if spec else ""
    out["delta_avg_loglikelihood"] = out["avg_loglikelihood"] - float(baseline["avg_loglikelihood"])
    out["delta_nll"] = out["nll"] - float(baseline["nll"])
    out["ppl_rel_increase"] = (out["ppl_like"] / max(float(baseline["ppl_like"]), 1.0e-30)) - 1.0
    out["greedy_drop"] = float(baseline["greedy_char_acc"]) - float(out["greedy_char_acc"])
    out["mean_abs_logit_delta_vs_baseline"] = mean_abs_delta(baseline_logits, logits)
    out["behavior_changed_vs_baseline"] = bool(out["mean_abs_logit_delta_vs_baseline"] > 1.0e-8)
    reason = break_reason(out, args)
    out["breaks"] = bool(reason)
    out["break_reason"] = reason
    return out


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            flat: Row = {}
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    flat[key] = json.dumps(value, sort_keys=True)
                else:
                    flat[key] = value
            writer.writerow(flat)


def build_thresholds(all_rows: list[Row], specs: list[SweepSpec]) -> list[Row]:
    thresholds: list[Row] = []
    by_sweep = {spec.name: [row for row in all_rows if row["sweep"] == spec.name] for spec in specs}
    for spec in specs:
        rows = sorted(
            by_sweep[spec.name],
            key=lambda row: spec.values.index(float(row["parameter_value"])),
        )
        first_break = next((row for row in rows if row["breaks"]), None)
        if first_break is None:
            safe_row = rows[-1] if rows else None
        else:
            before = rows[: rows.index(first_break)]
            safe_row = before[-1] if before else None
        threshold = {
            "sweep": spec.name,
            "group": spec.group,
            "parameter": spec.parameter,
            "unit": spec.unit,
            "stress_direction": spec.stress_direction,
            "tested_values": list(spec.values),
            "last_non_breaking_value": safe_row["parameter_value"] if safe_row else None,
            "first_breaking_value": first_break["parameter_value"] if first_break else None,
            "safe_range": safe_range_text(
                spec,
                float(safe_row["parameter_value"]) if safe_row else None,
                float(first_break["parameter_value"]) if first_break else None,
            ),
            "first_break_reason": first_break["break_reason"] if first_break else "",
            "last_non_breaking_delta_nll": safe_row["delta_nll"] if safe_row else None,
            "first_break_delta_nll": first_break["delta_nll"] if first_break else None,
            "last_non_breaking_logit_delta": safe_row["mean_abs_logit_delta_vs_baseline"] if safe_row else None,
            "first_break_logit_delta": first_break["mean_abs_logit_delta_vs_baseline"] if first_break else None,
            "description": spec.description,
        }
        thresholds.append(threshold)
    return thresholds


def projection_rows(projection: dict[str, Any]) -> list[Row]:
    dense = projection["dense_linear_layers_mapped_to_crossbars"]
    qkt = projection["attention_qkt_k_cache_programmed_to_crossbars"]
    av = projection["attention_av_v_cache_programmed_to_crossbars"]
    mlp_dyn = projection["dynamic_mlp_programming_cost"]
    amort = projection["static_model_weight_programming_amortization"]
    dyn = projection["dynamic_kv_cache_programming_cost"]
    endurance = projection["endurance_lifetime_projection"]
    retention = projection["retention_calibration"]
    return [
        {
            "test": "dense_linear_layers_mapped_to_crossbars",
            "passed": dense["layers"] > 0 and dense["crossbar_tiles"] > 0,
            "value": dense,
        },
        {
            "test": "attention_qkt_k_cache_programmed_to_crossbars",
            "passed": qkt["layers"] > 0,
            "value": qkt,
        },
        {
            "test": "attention_av_v_cache_programmed_to_crossbars",
            "passed": av["layers"] > 0,
            "value": av,
        },
        {
            "test": "static_model_weight_programming_amortization",
            "passed": amort["static_programmed_elements_once"] > 0,
            "value": amort,
        },
        {
            "test": "dynamic_kv_cache_programming_cost",
            "passed": dyn["dynamic_elements_per_decode_token"] > 0,
            "value": dyn,
        },
        {
            "test": "dynamic_mlp_gate_up_down_programming_cost",
            "passed": mlp_dyn["dynamic_elements_per_decode_token"] > 0,
            "value": mlp_dyn,
        },
        {
            "test": "endurance_lifetime_projection",
            "passed": endurance["projected_kv_write_cycles_per_slot"] > 0 and endurance["dynamic_mlp_is_endurance_limited"],
            "value": endurance,
        },
        {
            "test": "mapping_strategy_dependent_retention_calibration",
            "passed": (
                retention["kv_cache"]["retention_time_s"] < retention["weight_stationary"]["retention_time_s"]
                and retention["mlp_dynamic"]["retention_time_s"] < retention["weight_stationary"]["retention_time_s"]
            ),
            "value": retention,
        },
    ]


def build_markdown_summary(
    thresholds: list[Row],
    projection: dict[str, Any],
    args: argparse.Namespace,
    tensor_metadata: dict[str, str] | None = None,
) -> str:
    tensor_metadata = tensor_metadata or {}
    lines = [
        "# Qwen3 FP4LLM Skew Breakpoint Sweep",
        "",
        "This run uses lm-eval with deterministic loglikelihood cases, a real Qwen/Qwen3-1.7B safetensor projection slice, and FP4LLM's ReRAM crossbar wrapper.",
        "",
        "Break criteria:",
        f"- delta NLL >= {args.break_nll_delta:g}",
        f"- relative ppl-like increase >= {args.break_ppl_rel:g}",
        f"- mean absolute logit delta >= {args.break_logit_delta:g}",
        f"- greedy character accuracy drop >= {args.break_greedy_drop:g}",
        "",
        "| Sweep | Parameter | Safe range from tested grid | First breaking value | Reason |",
        "|---|---|---:|---:|---|",
    ]
    for row in thresholds:
        lines.append(
            "| {sweep} | {parameter} | {safe} | {first} | {reason} |".format(
                sweep=row["sweep"],
                parameter=row["parameter"],
                safe=row["safe_range"],
                first=format_number(row["first_breaking_value"]),
                reason=row["first_break_reason"] or "not reached",
            )
        )

    dyn = projection["dynamic_kv_cache_programming_cost"]
    mlp_dyn = projection["dynamic_mlp_programming_cost"]
    high_churn = projection["high_churn_dynamic_programming_cost"]
    dense = projection["dense_linear_layers_mapped_to_crossbars"]
    endurance = projection["endurance_lifetime_projection"]
    retention = projection["retention_calibration"]
    lines.extend(
        [
            "",
            "## Mapping / PE Checks",
            "",
            f"- Dense mapped layers: {dense['layers']} layers, {dense['crossbar_tiles']} tiles.",
            f"- Dynamic K/V programmed elements per decode token: {dyn['dynamic_elements_per_decode_token']}.",
            f"- Dynamic MLP gate/up/down programmed elements per decode token: {mlp_dyn['dynamic_elements_per_decode_token']}.",
            f"- K/V dynamic/amortized-static write ratio: {dyn['dynamic_to_amortized_static_write_ratio']:.3f}.",
            f"- High-churn dynamic/amortized-static write ratio: {high_churn['dynamic_to_amortized_static_write_ratio']:.3f}.",
            f"- Append-only K/V wear: {endurance['projected_kv_write_cycles_per_slot']:.6g} cycles/slot over {endurance['kv_cache_service_life_years']} years at {endurance['kv_cache_token_rate_tps']:.6g} tok/s.",
            f"- Projected decode tokens per reused dynamic cell at endurance limit: K/V {endurance['projected_decode_tokens_per_reused_kv_cell']}, MLP {endurance['projected_decode_tokens_per_reused_mlp_cell']}.",
            f"- Retention calibration: weight-stationary {retention['weight_stationary']['retention_time_s']:.6g}s, K/V cache {retention['kv_cache']['retention_time_s']:.6g}s ({retention['kv_cache']['strategy']}), MLP dynamic {retention['mlp_dynamic']['retention_time_s']:.6g}s.",
            f"- Verification tensor: {tensor_metadata.get('tensor_name', '')} ({tensor_metadata.get('tensor_role_description', '')})",
            "",
        ]
    )
    return "\n".join(lines)


def write_sweep_plan(path: Path, specs: list[SweepSpec]) -> None:
    rows = [
        {
            "sweep": spec.name,
            "group": spec.group,
            "parameter": spec.parameter,
            "unit": spec.unit,
            "stress_direction": spec.stress_direction,
            "values": list(spec.values),
            "description": spec.description,
        }
        for spec in specs
    ]
    write_csv(path, rows)


def write_eval_cases(path: Path, cases: list[EvalCase]) -> None:
    write_csv(path, [case.__dict__ for case in cases])


def parse_args() -> argparse.Namespace:
    default_cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    default_local_root = Path(os.environ.get("HF_MODEL_ROOT", Path.home() / "hf_models"))
    ts = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Sweep FP4LLM hardware skew knobs until lm-eval metrics break.")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--cache-dir", type=Path, default=default_cache)
    parser.add_argument("--local-root", type=Path, default=default_local_root)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--weight-bits", type=int, default=7)
    parser.add_argument("--act-bits", type=int, default=7)
    parser.add_argument(
        "--tensor-role",
        choices=qwen_tensor_role_choices(),
        default=DEFAULT_QWEN_TENSOR_ROLE,
        help="Qwen projection tensor used for breakpoint verification. Default is MLP output only.",
    )
    parser.add_argument("--sweeps", default="all", help="Comma-separated sweep names, or 'all'.")
    parser.add_argument("--break-nll-delta", type=float, default=0.01)
    parser.add_argument("--break-ppl-rel", type=float, default=0.01)
    parser.add_argument("--break-logit-delta", type=float, default=0.05)
    parser.add_argument("--break-greedy-drop", type=float, default=0.05)
    parser.add_argument("--prefill", type=Path, default=PROJECT_DIR / "mappings" / "Qwen3_1_7B_prefill.m")
    parser.add_argument("--decode", type=Path, default=PROJECT_DIR / "mappings" / "Qwen3_1_7B_decode.m")
    parser.add_argument("--endurance-cycles", type=int, default=100_000)
    parser.add_argument("--dynamic-kv-endurance-cycles", type=int, default=DEFAULT_DYNAMIC_KV_ENDURANCE_CYCLES)
    parser.add_argument("--dynamic-mlp-endurance-cycles", type=int, default=DEFAULT_DYNAMIC_MLP_ENDURANCE_CYCLES)
    parser.add_argument("--amortization-tokens", type=int, default=1_000_000)
    parser.add_argument("--weight-retention-time-s", type=float, default=DEFAULT_WEIGHT_RETENTION_TIME_S)
    parser.add_argument("--kv-cache-context-tokens", type=int, default=DEFAULT_KV_CACHE_CONTEXT_TOKENS)
    parser.add_argument("--kv-cache-token-time-s", type=float, default=DEFAULT_KV_CACHE_TOKEN_TIME_S)
    parser.add_argument("--kv-cache-service-life-years", type=float, default=DEFAULT_KV_CACHE_SERVICE_LIFE_YEARS)
    parser.add_argument(
        "--kv-cache-retention-strategy",
        choices=KV_CACHE_RETENTION_STRATEGIES,
        default="cache_residency",
    )
    parser.add_argument(
        "--kv-cache-retention-time-s",
        type=float,
        default=None,
        help="Override K/V retention residency directly. Otherwise it is derived from strategy, context tokens, and token time.",
    )
    parser.add_argument("--mlp-dynamic-retention-time-s", type=float, default=DEFAULT_MLP_DYNAMIC_RETENTION_TIME_S)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / f"qwen3_skew_breakpoints_{ts}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = select_specs(sweep_specs(kv_cache_token_time_s=args.kv_cache_token_time_s), args.sweeps)
    eval_cases = default_eval_cases()
    write_sweep_plan(out_dir / "skew_sweep_plan.csv", specs)
    write_eval_cases(out_dir / "lm_eval_test_cases.csv", eval_cases)

    snapshot_dir, reused_snapshot = resolve_snapshot(
        args.model,
        args.cache_dir,
        args.local_root,
        local_files_only=args.local_files_only,
        max_workers=args.max_workers,
    )
    tensor_name, tensor_file, weight_slice = load_qwen_weight_slice(snapshot_dir, args.dim, args.tensor_role)
    tensor_metadata = {
        "tensor_name": tensor_name,
        "tensor_file": tensor_file,
        **qwen_tensor_selection_metadata(tensor_name, args.tensor_role),
    }
    tokenizer = TinyCharTokenizer()

    baseline_case = FeatureCase(
        name="baseline_clean_w7a7",
        group="baseline",
        description="Clean W7/A7 crossbar slice with no added faults, drift, or read noise.",
    )
    print("[baseline] clean W7/A7", flush=True)
    baseline_row, baseline_logits, baseline_case_rows = run_feature_case(
        baseline_case,
        weight_slice,
        tokenizer,
        eval_cases,
        weight_bits=args.weight_bits,
        act_bits=args.act_bits,
    )
    baseline = annotate_row(
        baseline_row,
        baseline=baseline_row,
        baseline_logits=baseline_logits,
        logits=baseline_logits,
        spec=None,
        value=0.0,
        args=args,
    )

    all_rows: list[Row] = []
    per_case_rows: list[Row] = []
    total_cases = sum(len(spec.values) for spec in specs)
    done = 0

    for spec in specs:
        for value in spec.values:
            done += 1
            if float(value) == 0.0 and spec.stress_direction == "higher":
                row = dict(baseline)
                row.update(
                    {
                        "sweep": spec.name,
                        "parameter": spec.parameter,
                        "parameter_value": value,
                        "parameter_unit": spec.unit,
                        "stress_direction": spec.stress_direction,
                        "name": f"{spec.name}=0",
                        "group": spec.group,
                        "description": spec.description,
                    }
                )
                all_rows.append(row)
                for case_row in baseline_case_rows:
                    per_case = dict(case_row)
                    per_case["sweep"] = spec.name
                    per_case["parameter_value"] = value
                    per_case_rows.append(per_case)
                continue

            print(f"[{done}/{total_cases}] {spec.name} value={value:g}", flush=True)
            case = spec.make_case(float(value))
            row_raw, logits, case_rows = run_feature_case(
                case,
                weight_slice,
                tokenizer,
                eval_cases,
                weight_bits=args.weight_bits,
                act_bits=args.act_bits,
            )
            row = annotate_row(
                row_raw,
                baseline=baseline,
                baseline_logits=baseline_logits,
                logits=logits,
                spec=spec,
                value=float(value),
                args=args,
            )
            all_rows.append(row)

            for case_row in case_rows:
                per_case = dict(case_row)
                per_case["sweep"] = spec.name
                per_case["parameter"] = spec.parameter
                per_case["parameter_value"] = value
                per_case["feature"] = case.name
                per_case_rows.append(per_case)

            write_csv(out_dir / "skew_results.csv", all_rows)
            write_csv(out_dir / "per_case_scores.csv", per_case_rows)

    thresholds = build_thresholds(all_rows, specs)
    projection = mapping_projection(
        args.prefill.resolve(),
        args.decode.resolve(),
        endurance_cycles=args.endurance_cycles,
        amortization_tokens=args.amortization_tokens,
        dynamic_kv_endurance_cycles=args.dynamic_kv_endurance_cycles,
        dynamic_mlp_endurance_cycles=args.dynamic_mlp_endurance_cycles,
        weight_retention_time_s=args.weight_retention_time_s,
        kv_cache_context_tokens=args.kv_cache_context_tokens,
        kv_cache_token_time_s=args.kv_cache_token_time_s,
        kv_cache_retention_strategy=args.kv_cache_retention_strategy,
        kv_cache_retention_time_s=args.kv_cache_retention_time_s,
        kv_cache_service_life_years=args.kv_cache_service_life_years,
        mlp_dynamic_retention_time_s=args.mlp_dynamic_retention_time_s,
    )
    mapping_tests = projection_rows(projection)

    write_csv(out_dir / "skew_results.csv", all_rows)
    write_csv(out_dir / "per_case_scores.csv", per_case_rows)
    write_csv(out_dir / "breakpoint_thresholds.csv", thresholds)
    write_csv(out_dir / "mapping_pe_tests.csv", mapping_tests)
    (out_dir / "mapping_projection.json").write_text(json.dumps(projection, indent=2) + "\n", encoding="utf-8")

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "model": args.model,
        "snapshot_dir": str(snapshot_dir),
        "snapshot_reused_before_download": reused_snapshot,
        "cache_dir": str(args.cache_dir),
        "local_root": str(args.local_root),
        "tensor_name": tensor_name,
        "tensor_file": tensor_file,
        "tensor_selection": tensor_metadata,
        "slice_dim": args.dim,
        "weight_bits": args.weight_bits,
        "act_bits": args.act_bits,
        "break_criteria": {
            "break_nll_delta": args.break_nll_delta,
            "break_ppl_rel": args.break_ppl_rel,
            "break_logit_delta": args.break_logit_delta,
            "break_greedy_drop": args.break_greedy_drop,
        },
        "baseline": baseline,
        "thresholds": thresholds,
        "mapping_pe_tests": mapping_tests,
        "all_mapping_pe_tests_passed": all(bool(row["passed"]) for row in mapping_tests),
        "outputs": {
            "breakpoint_thresholds_csv": str(out_dir / "breakpoint_thresholds.csv"),
            "skew_results_csv": str(out_dir / "skew_results.csv"),
            "per_case_scores_csv": str(out_dir / "per_case_scores.csv"),
            "lm_eval_test_cases_csv": str(out_dir / "lm_eval_test_cases.csv"),
            "skew_sweep_plan_csv": str(out_dir / "skew_sweep_plan.csv"),
            "mapping_pe_tests_csv": str(out_dir / "mapping_pe_tests.csv"),
            "mapping_projection_json": str(out_dir / "mapping_projection.json"),
            "summary_md": str(out_dir / "summary.md"),
            "summary_json": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(
        build_markdown_summary(thresholds, projection, args, tensor_metadata),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "sweeps": len(specs),
                "evaluated_points": len(all_rows),
                "all_mapping_pe_tests_passed": summary["all_mapping_pe_tests_passed"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
