#Reliability/Noise modeling testing
import csv
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from Crossbar_array import (
    CrossbarArray,
    CrossbarIVNonlinearityConfig,
    CrossbarNoiseConfig,
    CrossbarReliabilityConfig,
    CrossbarTDCConfig,
)


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "layer0_pe_effect_test_results"
INPUT_ROWS = 64
PHYSICAL_OUTPUTS = 64
LOGICAL_OUTPUTS = 32
LAYER0_ACTIVE_ROWS = 27
NUM_LAYER0_PE_VECTORS = 32
MAPPING_REPORT = ROOT / "Yolo26n_auto_layer0_mapping_report.csv"
MAPPING_CYCLE_TIME_S = float(os.environ.get("LAYER0_MAPPING_CYCLE_TIME_S", "1e-9"))
RETENTION_REFERENCE_S = 1.0
RETENTION_STRATEGIES = ("weight_stationary", "fast_remap", "kv_cache")
RETENTION_DEFAULT_STRATEGY = os.environ.get("LAYER0_RETENTION_STRATEGY", "weight_stationary")
INCLUDE_RETENTION_STRATEGY_SWEEP = os.environ.get("LAYER0_INCLUDE_RETENTION_STRATEGY_SWEEP", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RERAM_RETENTION_125C_LRS_BETA = -0.02556
RERAM_RETENTION_125C_HRS_BETA = -0.04041


def _read_numeric_text(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return np.asarray(json.load(handle))
    except Exception:
        pass
    try:
        return np.loadtxt(path, comments=("R", "G", "B", "F", "L"))
    except Exception:
        return None


def _maybe_load_layer0_weights():
    candidates = [
        ROOT / "layer0_weights.txt",
        ROOT / "Layer0_weights.txt",
        ROOT / "weights_layer0.txt",
        ROOT / "layer0_weight.txt",
    ]
    for path in candidates:
        if not path.exists():
            continue
        raw = _read_numeric_text(path)
        if raw is None or raw.size == 0:
            continue

        tile = np.zeros((INPUT_ROWS, LOGICAL_OUTPUTS), dtype=np.int16)
        if raw.ndim == 4:
            # Expected YOLO layer-0 layout: out_channels x kernel_h x kernel_w x input_channels.
            out_ch = min(raw.shape[0], LOGICAL_OUTPUTS)
            flat_rows = min(int(np.prod(raw.shape[1:])), INPUT_ROWS)
            for out_idx in range(out_ch):
                tile[:flat_rows, out_idx] = np.asarray(raw[out_idx]).reshape(-1)[:flat_rows]
        else:
            values = np.asarray(raw).reshape(-1)
            if values.size >= INPUT_ROWS * LOGICAL_OUTPUTS:
                tile[:, :] = values[: INPUT_ROWS * LOGICAL_OUTPUTS].reshape(INPUT_ROWS, LOGICAL_OUTPUTS)
            elif values.size >= LAYER0_ACTIVE_ROWS * LOGICAL_OUTPUTS:
                tile[:LAYER0_ACTIVE_ROWS, :] = values[: LAYER0_ACTIVE_ROWS * LOGICAL_OUTPUTS].reshape(
                    LAYER0_ACTIVE_ROWS, LOGICAL_OUTPUTS
                )
            else:
                continue
        return np.clip(np.rint(tile), -128, 127).astype(np.int16), str(path)
    return None, None


def _read_channel_matrix(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped[0].isalpha():
                continue
            rows.append([int(value) for value in stripped.split()])
    if not rows:
        return None
    return np.asarray(rows, dtype=np.uint8)


def _maybe_load_bus_patches():
    channel_paths = [
        ROOT / "bus_R_channel_640x640.txt",
        ROOT / "bus_G_channel_640x640.txt",
        ROOT / "bus_B_channel_640x640.txt",
    ]
    if not all(path.exists() for path in channel_paths):
        return None, None

    channels = [_read_channel_matrix(path) for path in channel_paths]
    if any(channel is None for channel in channels):
        return None, None
    image = np.stack(channels, axis=-1)

    padding = _read_numeric_text(ROOT / "Layer0_padding.txt")
    if padding is not None and np.asarray(padding).shape == (4, 2):
        pad = np.asarray(padding, dtype=int)
        top, bottom = pad[1]
        left, right = pad[2]
    else:
        top = bottom = left = right = 1

    padded = np.pad(image, ((top, bottom), (left, right), (0, 0)), mode="constant", constant_values=0)
    activations = np.zeros((NUM_LAYER0_PE_VECTORS, INPUT_ROWS), dtype=np.uint8)
    idx = 0
    for y in range(0, padded.shape[0] - 2, 2):
        for x in range(0, padded.shape[1] - 2, 2):
            patch = padded[y : y + 3, x : x + 3, :].reshape(-1)
            activations[idx, :LAYER0_ACTIVE_ROWS] = patch[:LAYER0_ACTIVE_ROWS]
            idx += 1
            if idx == NUM_LAYER0_PE_VECTORS:
                source = ", ".join(str(path) for path in channel_paths)
                return activations, source
    return None, None


def _maybe_load_layer0_activations():
    candidates = [
        ROOT / "layer0_activations.txt",
        ROOT / "input_layer0.txt",
    ]
    for path in candidates:
        if not path.exists():
            continue
        raw = _read_numeric_text(path)
        if raw is None or raw.size == 0:
            continue
        values = np.asarray(raw).reshape(-1)
        needed = NUM_LAYER0_PE_VECTORS * LAYER0_ACTIVE_ROWS
        if values.size < needed:
            continue
        activations = np.zeros((NUM_LAYER0_PE_VECTORS, INPUT_ROWS), dtype=np.uint8)
        activations[:, :LAYER0_ACTIVE_ROWS] = values[:needed].reshape(NUM_LAYER0_PE_VECTORS, LAYER0_ACTIVE_ROWS)
        return np.clip(np.rint(activations), 0, 255).astype(np.uint8), str(path)

    return _maybe_load_bus_patches()


def _read_layer0_mapping_row():
    if not MAPPING_REPORT.exists():
        return None
    with open(MAPPING_REPORT, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else None


def _normalize_retention_strategy(strategy):
    normalized = str(strategy or RETENTION_DEFAULT_STRATEGY).strip().lower().replace("-", "_")
    aliases = {
        "stationary": "weight_stationary",
        "weights": "weight_stationary",
        "yolo": "weight_stationary",
        "remap": "fast_remap",
        "fast": "fast_remap",
        "kv": "kv_cache",
        "llm": "kv_cache",
        "llm_kv_cache": "kv_cache",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in RETENTION_STRATEGIES:
        raise ValueError(
            f"Unknown retention mapping strategy: {strategy}. "
            f"Expected one of {', '.join(RETENTION_STRATEGIES)}."
        )
    return normalized


def _env_retention_time_s(strategy):
    env_names = [
        f"LAYER0_{strategy.upper()}_RETENTION_S",
        f"RETENTION_{strategy.upper()}_S",
    ]
    for env_name in env_names:
        if env_name in os.environ:
            return max(float(os.environ[env_name]), 0.0)
    return None


def estimate_layer0_retention_time_s(strategy):
    strategy = _normalize_retention_strategy(strategy)
    env_time = _env_retention_time_s(strategy)
    if env_time is not None:
        return env_time

    row = _read_layer0_mapping_row()
    if row is None and strategy != "kv_cache":
        return 0.0

    if strategy == "weight_stationary":
        reuse_count = max(int(os.environ.get("LAYER0_WEIGHT_STATIONARY_REUSE_COUNT", "1")), 1)
        cycles = int(row.get("LatencyCycles", 0)) * reuse_count
    elif strategy == "fast_remap":
        cycles = sum(
            int(row.get(field, 0))
            for field in ("FoldProgram", "FoldCompute", "Reduction", "Finalize")
        )
    elif strategy == "kv_cache":
        context_tokens = max(int(os.environ.get("LAYER0_KV_CACHE_CONTEXT_TOKENS", "0")), 0)
        token_time_s = max(float(os.environ.get("LAYER0_KV_CACHE_TOKEN_TIME_S", "0")), 0.0)
        return context_tokens * token_time_s

    return max(cycles * MAPPING_CYCLE_TIME_S, 0.0)


def retention_reliability_config(retention_time_s):
    return CrossbarReliabilityConfig(
        enabled=True,
        retention_time_s=retention_time_s,
        retention_reference_s=RETENTION_REFERENCE_S,
        retention_temp_c=125.0,
        retention_temp_reference_c=125.0,
        retention_model="state_power_law",
        retention_lrs_beta_log10_per_decade=RERAM_RETENTION_125C_LRS_BETA,
        retention_hrs_beta_log10_per_decade=RERAM_RETENTION_125C_HRS_BETA,
    )


def retention_scenario_definitions():
    default_strategy = _normalize_retention_strategy(RETENTION_DEFAULT_STRATEGY)
    strategies = [default_strategy]
    if INCLUDE_RETENTION_STRATEGY_SWEEP:
        strategies.extend(strategy for strategy in RETENTION_STRATEGIES if strategy != default_strategy)

    return [
        {
            "name": f"retention_{strategy}_mapping",
            "topic": f"Retention drift from {strategy.replace('_', '-')} mapping residency",
            "kwargs": {
                "reliability_config": retention_reliability_config(
                    estimate_layer0_retention_time_s(strategy)
                ),
                "reliability_seed": 1004 + idx,
            },
        }
        for idx, strategy in enumerate(strategies)
    ]


def build_layer0_pe_inputs():
    weights, weight_source = _maybe_load_layer0_weights()
    activations, activation_source = _maybe_load_layer0_activations()

    if weights is None or activations is None:
        rng = np.random.default_rng(260)
        weights = np.zeros((INPUT_ROWS, LOGICAL_OUTPUTS), dtype=np.int16)
        weights[:LAYER0_ACTIVE_ROWS, :] = rng.integers(
            -96, 97, size=(LAYER0_ACTIVE_ROWS, LOGICAL_OUTPUTS), dtype=np.int16
        )
        activations = np.zeros((NUM_LAYER0_PE_VECTORS, INPUT_ROWS), dtype=np.uint8)
        activations[:, :LAYER0_ACTIVE_ROWS] = rng.integers(
            0, 256, size=(NUM_LAYER0_PE_VECTORS, LAYER0_ACTIVE_ROWS), dtype=np.uint8
        )
        fixture = "deterministic_layer0_shaped_fixture"
    else:
        fixture = "workspace_layer0_text_files"

    metadata = {
        "fixture": fixture,
        "weight_source": weight_source,
        "activation_source": activation_source,
        "input_rows": INPUT_ROWS,
        "physical_outputs": PHYSICAL_OUTPUTS,
        "logical_outputs_compared": LOGICAL_OUTPUTS,
        "layer0_active_rows": LAYER0_ACTIVE_ROWS,
        "num_layer0_pe_vectors": NUM_LAYER0_PE_VECTORS,
        "mapping_cycle_time_s": MAPPING_CYCLE_TIME_S,
        "mapping_report": str(MAPPING_REPORT) if MAPPING_REPORT.exists() else None,
        "retention_reference_s": RETENTION_REFERENCE_S,
        "default_retention_strategy": _normalize_retention_strategy(RETENTION_DEFAULT_STRATEGY),
        "retention_strategy_sweep_enabled": INCLUDE_RETENTION_STRATEGY_SWEEP,
        "available_retention_strategies": list(RETENTION_STRATEGIES),
        "retention_time_s_by_strategy": {
            strategy: estimate_layer0_retention_time_s(strategy) for strategy in RETENTION_STRATEGIES
        },
    }
    return weights, activations, metadata


def scenario_definitions():
    return [
        {
            "name": "clean_pe_execution",
            "topic": "Crossbar PE execution",
            "kwargs": {},
        },
        {
            "name": "stuck_at_lrs_hrs_faults",
            "topic": "Stuck-at LRS/HRS faults",
            "kwargs": {
                "reliability_config": CrossbarReliabilityConfig(
                    enabled=True,
                    stuck_at_lrs_prob=0.02,
                    stuck_at_hrs_prob=0.02,
                ),
                "reliability_seed": 1001,
            },
        },
        {
            "name": "raw_ber_no_ecc",
            "topic": "Raw BER without ECC",
            "kwargs": {
                "reliability_config": CrossbarReliabilityConfig(
                    enabled=True,
                    raw_ber_ppm=10000.0,
                    use_ecc=False,
                ),
                "reliability_seed": 1002,
            },
        },
        {
            "name": "raw_ber_with_ecc",
            "topic": "Raw BER with SEC-style ECC estimate",
            "kwargs": {
                "reliability_config": CrossbarReliabilityConfig(
                    enabled=True,
                    raw_ber_ppm=10000.0,
                    use_ecc=True,
                    ecc_codeword_bits=38,
                    ecc_data_bits=32,
                    ecc_encode_latency_cycles=1,
                    ecc_decode_latency_cycles=1,
                ),
                "reliability_seed": 1003,
            },
        },
        *retention_scenario_definitions(),
        {
            "name": "relaxation_drift",
            "topic": "Relaxation drift",
            "kwargs": {
                "reliability_config": CrossbarReliabilityConfig(
                    enabled=True,
                    relaxation_time_s=7200.0,
                    relaxation_tau_s=2000.0,
                    relaxation_resistance_amplitude_ln=0.16,
                ),
                "reliability_seed": 1005,
            },
        },
        {
            "name": "state_conductance_variation",
            "topic": "State / conductance variation",
            "kwargs": {
                "reliability_config": CrossbarReliabilityConfig(
                    enabled=True,
                    state_sigma_ln=0.12,
                ),
                "reliability_seed": 1006,
            },
        },
        {
            "name": "endurance_limit_failure_modes",
            "topic": "Endurance limit stuck-cell / multilevel-loss modes",
            "kwargs": {
                "reliability_config": CrossbarReliabilityConfig(
                    enabled=True,
                    endurance_write_cycles=100000,
                    endurance_cycles=100000,
                    endurance_limit_stuck_at_lrs_prob=0.02,
                    endurance_limit_stuck_at_hrs_prob=0.02,
                    endurance_limit_multilevel_loss_prob=0.12,
                    endurance_limit_multilevel_sigma_ln=0.30,
                ),
                "reliability_seed": 1007,
            },
        },
        {
            "name": "thermal_shot_noise",
            "topic": "Thermal + shot noise",
            "kwargs": {
                "noise_config": CrossbarNoiseConfig(
                    enabled=True,
                    temp_k=400.0,
                    fano=5.0,
                    bandwidth_hz=5.0e10,
                    flicker_af_lrs=0.0,
                    flicker_af_hrs=0.0,
                    rtn_amp_frac_lrs=0.0,
                    rtn_amp_frac_hrs=0.0,
                    readout_noise_a_rms=0.0,
                ),
                "noise_seed": 2001,
            },
        },
        {
            "name": "flicker_1f_noise",
            "topic": "Flicker / 1/f noise",
            "kwargs": {
                "noise_config": CrossbarNoiseConfig(
                    enabled=True,
                    temp_k=0.0,
                    fano=0.0,
                    bandwidth_hz=50000.0,
                    flicker_fmin_hz=0.1,
                    flicker_af_lrs=2.0e-4,
                    flicker_af_hrs=8.0e-4,
                    rtn_amp_frac_lrs=0.0,
                    rtn_amp_frac_hrs=0.0,
                    readout_noise_a_rms=0.0,
                ),
                "noise_seed": 2002,
            },
        },
        {
            "name": "rtn_noise",
            "topic": "RTN noise",
            "kwargs": {
                "noise_config": CrossbarNoiseConfig(
                    enabled=True,
                    temp_k=0.0,
                    fano=0.0,
                    flicker_af_lrs=0.0,
                    flicker_af_hrs=0.0,
                    rtn_amp_frac_lrs=0.08,
                    rtn_amp_frac_hrs=0.18,
                    readout_noise_a_rms=0.0,
                ),
                "noise_seed": 2003,
            },
        },
        {
            "name": "readout_noise",
            "topic": "Readout noise",
            "kwargs": {
                "noise_config": CrossbarNoiseConfig(
                    enabled=True,
                    temp_k=0.0,
                    fano=0.0,
                    flicker_af_lrs=0.0,
                    flicker_af_hrs=0.0,
                    rtn_amp_frac_lrs=0.0,
                    rtn_amp_frac_hrs=0.0,
                    readout_noise_a_rms=4.0e-7,
                ),
                "noise_seed": 2004,
            },
        },
        {
            "name": "adc_clipping_quantization",
            "topic": "ADC clipping / quantization",
            "kwargs": {
                "iv_config": CrossbarIVNonlinearityConfig(
                    enabled=True,
                    mode="hard_clamp",
                    adc_full_scale_levels=255.0,
                ),
            },
        },
        {
            "name": "sar_tdc_8b",
            "topic": "SAR-TDC differential output quantization",
            "kwargs": {
                "tdc_config": CrossbarTDCConfig(
                    enabled=True,
                    precision_mode="8b",
                    output_format="sar_fp",
                    gain_shift=0,
                ),
            },
        },
    ]


def run_pe(weights, activations, **crossbar_kwargs):
    with redirect_stdout(io.StringIO()):
        pe = CrossbarArray(
            input_size=INPUT_ROWS,
            output_size=PHYSICAL_OUTPUTS,
            adc_bits=8,
            cell_bits=4,
            safety_checks=True,
            **crossbar_kwargs,
        )
        pe.program_weights(weights)
        outputs = np.stack([pe.compute(vec) for vec in activations], axis=0)
    return outputs, pe.get_effective_weights(), pe.get_info()


def summarize(name, topic, outputs, effective_weights, baseline_outputs, baseline_weights, info):
    active = outputs[:, :LOGICAL_OUTPUTS].astype(np.int64)
    baseline_active = baseline_outputs[:, :LOGICAL_OUTPUTS].astype(np.int64)
    delta = active - baseline_active
    abs_delta = np.abs(delta)
    weight_delta = effective_weights.astype(np.int64) - baseline_weights.astype(np.int64)
    latency = info.get("latency_model", {})
    stats = info.get("reliability_stats", {})
    reliability_config = info.get("reliability_config", {})
    iv_stats = info.get("iv_saturation_stats", {})
    tdc_config = info.get("tdc_config", {})
    programmed_cells = max(int(stats.get("programmed_cells", 0)), 1)
    endurance_failed = (
        int(stats.get("endurance_limit_stuck_at_lrs_cells", 0))
        + int(stats.get("endurance_limit_stuck_at_hrs_cells", 0))
        + int(stats.get("endurance_limit_multilevel_loss_cells", 0))
    )
    return {
        "scenario": name,
        "topic": topic,
        "mae_vs_clean": float(np.mean(abs_delta)),
        "rmse_vs_clean": float(np.sqrt(np.mean(delta.astype(float) ** 2))),
        "max_abs_delta": int(np.max(abs_delta)),
        "changed_output_fraction": float(np.mean(delta != 0)),
        "mean_delta": float(np.mean(delta)),
        "output_min": int(np.min(active)),
        "output_max": int(np.max(active)),
        "delta_min": int(np.min(delta)),
        "delta_max": int(np.max(delta)),
        "effective_weight_changed_fraction": float(np.mean(weight_delta != 0)),
        "effective_weight_max_abs_delta": int(np.max(np.abs(weight_delta))),
        "ecc_program_latency_cycles": int(latency.get("ecc_program_latency_cycles", 0)),
        "ecc_compute_latency_cycles": int(latency.get("ecc_compute_latency_cycles", 0)),
        "ecc_total_latency_cycles": int(latency.get("ecc_total_latency_cycles", 0)),
        "endurance_limit_failed_cell_fraction": float(endurance_failed / programmed_cells),
        "retention_time_s": float(reliability_config.get("retention_time_s", 0.0)),
        "retention_model": str(reliability_config.get("retention_model", "")),
        "iv_upper_saturation_events": int(iv_stats.get("upper_saturation_events", 0)),
        "iv_unique_saturated_columns": int(iv_stats.get("unique_saturated_columns", 0)),
        "tdc_enabled": bool(info.get("tdc_enabled", False)),
        "tdc_precision_mode": str(tdc_config.get("precision_mode") or ""),
        "tdc_output_format": str(tdc_config.get("output_format", "")),
        "tdc_effective_coarse_bits": int(info.get("tdc_effective_coarse_bits", 0)),
        "tdc_effective_fine_bits": int(info.get("tdc_effective_fine_bits", 0)),
    }


def to_jsonable(value):
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    weights, activations, metadata = build_layer0_pe_inputs()

    scenarios = scenario_definitions()
    outputs_by_name = {}
    infos_by_name = {}
    rows = []

    baseline_outputs, baseline_weights, baseline_info = run_pe(weights, activations, **scenarios[0]["kwargs"])
    digital_ref = activations.astype(np.int32) @ weights.astype(np.int32)
    clean_matches_digital = bool(np.array_equal(baseline_outputs[:, :LOGICAL_OUTPUTS], digital_ref))

    for scenario in scenarios:
        outputs, effective_weights, info = run_pe(weights, activations, **scenario["kwargs"])
        outputs_by_name[scenario["name"]] = outputs
        infos_by_name[scenario["name"]] = info
        rows.append(
            summarize(
                scenario["name"],
                scenario["topic"],
                outputs,
                effective_weights,
                baseline_outputs,
                baseline_weights,
                info,
            )
        )

    np.save(RESULT_DIR / "layer0_pe_weights.npy", weights)
    np.save(RESULT_DIR / "layer0_pe_activations.npy", activations)
    np.save(RESULT_DIR / "clean_digital_reference.npy", digital_ref)
    np.savez_compressed(RESULT_DIR / "scenario_outputs.npz", **outputs_by_name)

    with open(RESULT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "metadata": metadata,
        "clean_matches_digital_reference": clean_matches_digital,
        "scenario_count": len(scenarios),
        "scenarios": [
            {
                "name": item["name"],
                "topic": item["topic"],
                "kwargs": to_jsonable(item["kwargs"]),
                "crossbar_info": to_jsonable(infos_by_name[item["name"]]),
            }
            for item in scenarios
        ],
        "summary": rows,
        "result_files": {
            "summary_csv": str(RESULT_DIR / "summary.csv"),
            "summary_json": str(RESULT_DIR / "summary.json"),
            "scenario_outputs_npz": str(RESULT_DIR / "scenario_outputs.npz"),
            "weights_npy": str(RESULT_DIR / "layer0_pe_weights.npy"),
            "activations_npy": str(RESULT_DIR / "layer0_pe_activations.npy"),
            "clean_digital_reference_npy": str(RESULT_DIR / "clean_digital_reference.npy"),
        },
    }
    with open(RESULT_DIR / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Saved layer-0 PE effect results to: {RESULT_DIR}")
    print(f"Scenario count: {len(scenarios)}")
    print(f"Clean PE matches digital reference: {clean_matches_digital}")
    for row in rows:
        print(
            f"{row['scenario']}: MAE={row['mae_vs_clean']:.3f}, "
            f"max_abs={row['max_abs_delta']}, changed={row['changed_output_fraction']:.3f}"
        )


if __name__ == "__main__":
    main()
