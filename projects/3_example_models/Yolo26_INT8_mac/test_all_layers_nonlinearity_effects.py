# Full-network reliability/noise/sensing audit for YOLO26n crossbar layers.
import contextlib
import csv
import io
import json
from pathlib import Path

import numpy as np

from Crossbar_array import (
    CrossbarArray,
    CrossbarIVNonlinearityConfig,
    CrossbarNoiseConfig,
    CrossbarReliabilityConfig,
    CrossbarTDCConfig,
)
from full_inference_executor import FullInferenceExecutor
from hybrid_inference_executor import _is_float_tensor, _is_quantized_int
from user import make_weight_provider


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "all_layer_nonlinearity_effect_test_results"
RETENTION_30_DAYS_S = 30 * 24 * 60 * 60
RERAM_RETENTION_125C_LRS_BETA = -0.02556
RERAM_RETENTION_125C_HRS_BETA = -0.04041
IV_ROW_GROUPS = 4
IV_FULL_SCALE_LEVELS = 2250.0
TDC_PRECISION_MODE = "9b"
TDC_CHUNK_ROWS = 4096
IV_CHUNK_ROWS = 4096


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _decompose_signed_int8(weights):
    w = np.clip(np.rint(weights), -128, 127).astype(np.int16)
    pos = np.maximum(w, 0).astype(np.int16)
    neg = np.maximum(-w, 0).astype(np.int16)
    return (
        (pos & 0xF).astype(np.uint8),
        ((pos >> 4) & 0xF).astype(np.uint8),
        (neg & 0xF).astype(np.uint8),
        ((neg >> 4) & 0xF).astype(np.uint8),
    )


def _reconstruct_signed_int8(banks):
    pos0, pos1, neg0, neg1 = banks
    return (
        pos0.astype(np.int16)
        + (pos1.astype(np.int16) << 4)
        - neg0.astype(np.int16)
        - (neg1.astype(np.int16) << 4)
    )


def _reliability_config():
    return CrossbarReliabilityConfig(
        enabled=True,
        raw_ber_ppm=10.0,
        use_ecc=True,
        ecc_codeword_bits=38,
        ecc_data_bits=32,
        ecc_encode_latency_cycles=1,
        ecc_decode_latency_cycles=1,
        retention_time_s=RETENTION_30_DAYS_S,
        retention_reference_s=1.0,
        retention_temp_c=125.0,
        retention_temp_reference_c=125.0,
        retention_model="state_power_law",
        retention_lrs_beta_log10_per_decade=RERAM_RETENTION_125C_LRS_BETA,
        retention_hrs_beta_log10_per_decade=RERAM_RETENTION_125C_HRS_BETA,
        state_sigma_ln=0.02,
    )


def _noise_config():
    return CrossbarNoiseConfig(
        enabled=True,
        bandwidth_hz=5000.0,
    )


def _iv_config():
    return CrossbarIVNonlinearityConfig(
        enabled=True,
        mode="hard_clamp",
        adc_full_scale_levels=IV_FULL_SCALE_LEVELS,
    )


def _tdc_config():
    return CrossbarTDCConfig(
        enabled=True,
        precision_mode=TDC_PRECISION_MODE,
        output_format="sar_fp",
        gain_shift=0,
    )


def _make_crossbar_quiet(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return CrossbarArray(**kwargs)


def _prepare_layer_tensors(executor, op, layer_name):
    runner = executor.crossbar_runner
    input_ids = [int(idx) for idx in op.get("inputs", [])]
    output_ids = [int(idx) for idx in op.get("outputs", [])]
    input_meta = executor.tensor_meta[input_ids[0]]
    weight_meta = executor.tensor_meta[input_ids[1]]
    output_meta = executor.tensor_meta[output_ids[0]]
    input_value = executor._get_reference_tensor(input_ids[0])
    weight_q = executor._get_reference_tensor(input_ids[1])

    mdims, _ = runner.mapper.map_layer_from_file(runner.mapping_path, layer_name=layer_name)
    conv_weights = runner._prepare_weights(str(op["op_name"]), np.asarray(weight_q))
    provider_mdims = mdims
    if str(op["op_name"]) == "DEPTHWISE_CONV_2D":
        provider_mdims = type(mdims)(
            K=mdims.K,
            C=int(conv_weights.shape[0]),
            R=mdims.R,
            S=mdims.S,
            Y=mdims.Y,
            X=mdims.X,
            stride_x=mdims.stride_x,
            stride_y=mdims.stride_y,
        )

    with contextlib.redirect_stdout(io.StringIO()):
        weight_provider, is_depthwise, actual_output_channels, actual_dims = make_weight_provider(
            conv_weights, provider_mdims
        )

    mdims = type(mdims)(
        K=actual_output_channels,
        C=actual_dims["C"],
        R=actual_dims["R"],
        S=actual_dims["S"],
        Y=mdims.Y,
        X=mdims.X,
        stride_x=mdims.stride_x,
        stride_y=mdims.stride_y,
    )
    out_y = (int(mdims.Y) + int(mdims.stride_y) - 1) // int(mdims.stride_y)
    out_x = (int(mdims.X) + int(mdims.stride_x) - 1) // int(mdims.stride_x)

    if _is_float_tensor(input_meta) and _is_float_tensor(output_meta):
        input_real = np.asarray(input_value, dtype=np.float32)
        max_abs = float(np.max(np.abs(input_real))) if input_real.size else 0.0
        activation_scale = max(max_abs / 127.0, 1.0 / 127.0)
        input_q = np.clip(np.rint(input_real / activation_scale), -128, 127).astype(np.int16)
        activation_offset = 128
        input_chw_u8 = np.clip(input_q[0] + activation_offset, 0, 255).astype(np.uint8).transpose(2, 0, 1)
        pad_fill = activation_offset
    elif _is_quantized_int(input_meta):
        input_i16 = np.asarray(input_value, dtype=np.int16)
        activation_offset = 0
        input_chw_u8 = np.clip(input_i16[0] - int(input_meta.zero_point), 0, 255).astype(np.uint8).transpose(2, 0, 1)
        pad_fill = 0
    else:
        raise ValueError(f"Unsupported activation tensor for {layer_name}: {input_meta.dtype}")

    pre_padded = runner._input_is_pre_padded(input_chw_u8, mdims)
    cols_u8 = runner._im2col_u8(
        input_chw_u8,
        mdims,
        pre_padded=pre_padded,
        pad_fill=pad_fill,
        out_y=out_y,
        out_x=out_x,
    ).astype(np.uint8, copy=False)
    weights = np.clip(
        np.rint(weight_provider(0, int(actual_output_channels))),
        -128,
        127,
    ).astype(np.int16, copy=False)

    if cols_u8.shape[1] != weights.shape[0]:
        raise ValueError(
            f"{layer_name}: im2col width {cols_u8.shape[1]} does not match weight rows {weights.shape[0]}"
        )

    return {
        "layer": layer_name,
        "op_name": str(op["op_name"]),
        "cols_u8": cols_u8,
        "weights": weights,
        "activation_offset": int(activation_offset),
        "out_y": int(out_y),
        "out_x": int(out_x),
        "output_channels": int(actual_output_channels),
        "patch_len": int(weights.shape[0]),
        "is_depthwise": bool(is_depthwise),
        "input_tensor": int(input_ids[0]),
        "weight_tensor": int(input_ids[1]),
        "output_tensor": int(output_ids[0]),
        "input_shape": tuple(int(v) for v in input_meta.shape),
        "weight_shape": tuple(int(v) for v in weight_meta.shape),
        "output_shape": tuple(int(v) for v in output_meta.shape),
    }


def _audit_reliability_and_noise(layer, reliability_totals, noise_totals):
    cols_u8 = layer["cols_u8"]
    weights = layer["weights"]
    activation_offset = int(layer["activation_offset"])
    banks_clean = _decompose_signed_int8(weights)

    low_passes = np.count_nonzero(cols_u8 & 0xF, axis=0).astype(np.int64)
    high_passes = np.count_nonzero(cols_u8 >> 4, axis=0).astype(np.int64)
    row_passes = low_passes + high_passes
    row_any_reads = np.count_nonzero(cols_u8, axis=0).astype(np.int64)
    k = int(weights.shape[1])
    reliability_totals["programmed_cell_reads_audited"] += int(np.sum(row_passes) * k * 4)

    reliability_pe = _make_crossbar_quiet(
        input_size=64,
        output_size=64,
        adc_bits=8,
        cell_bits=4,
        safety_checks=False,
        reliability_config=_reliability_config(),
        reliability_seed=31000 + int(reliability_totals["layers_audited"]),
    )
    banks_reliable = reliability_pe._decompose_signed_int8_to_4bit_banks(weights)
    stats = reliability_pe.get_info()["reliability_stats"]
    reliability_totals["raw_ber_bit_flips_after_ecc"] += int(stats.get("raw_ber_bit_flips", 0))
    reliability_totals["programmed_unique_cell_levels"] += int(stats.get("programmed_cells", 0))

    changed_bank_counts_by_row = np.zeros(weights.shape[0], dtype=np.int64)
    for clean, changed in zip(banks_clean, banks_reliable):
        changed_bank_counts_by_row += np.count_nonzero(clean != changed, axis=1).astype(np.int64)
    reliability_totals["changed_cell_level_reads"] += int(np.dot(row_passes, changed_bank_counts_by_row))

    clean_eff = _reconstruct_signed_int8(banks_clean)
    reliable_eff = _reconstruct_signed_int8(banks_reliable)
    delta_w = reliable_eff.astype(np.int32) - clean_eff.astype(np.int32)
    changed_eff_by_row = np.count_nonzero(delta_w != 0, axis=1).astype(np.int64)
    reliability_totals["effective_int8_weight_read_entries_changed"] += int(np.dot(row_any_reads, changed_eff_by_row))

    cols_signed_abs = np.abs(cols_u8.astype(np.int16) - activation_offset).astype(np.int32)
    row_abs_sum = np.sum(cols_signed_abs, axis=0, dtype=np.int64)
    row_abs_max = np.max(cols_signed_abs, axis=0).astype(np.int64) if cols_signed_abs.size else np.zeros(weights.shape[0], dtype=np.int64)
    abs_delta_w = np.abs(delta_w).astype(np.int64)
    reliability_totals["branch_abs_error_sum"] += int(np.dot(row_abs_sum, np.sum(abs_delta_w, axis=1, dtype=np.int64)))
    per_row_max_delta = np.max(abs_delta_w, axis=1).astype(np.int64) if abs_delta_w.size else np.zeros(weights.shape[0], dtype=np.int64)
    reliability_totals["branch_abs_error_max"] = max(
        int(reliability_totals["branch_abs_error_max"]),
        int(np.max(row_abs_max * per_row_max_delta)) if per_row_max_delta.size else 0,
    )
    reliability_totals["branch_error_denominator"] += int(cols_u8.shape[0] * weights.shape[0] * k)

    noise_pe = _make_crossbar_quiet(
        input_size=64,
        output_size=64,
        adc_bits=8,
        cell_bits=4,
        safety_checks=False,
        noise_config=_noise_config(),
    )
    for bank in banks_clean:
        sigma = noise_pe._read_noise_level_sigma(bank)
        noise_totals["weighted_sigma_sum"] += float(np.sum(row_passes[:, None] * sigma))
        noise_totals["weighted_sigma_count"] += int(np.sum(row_passes) * k)
        if np.any(row_passes > 0):
            active_sigma = sigma[row_passes > 0]
            noise_totals["max_sigma"] = max(float(noise_totals["max_sigma"]), float(np.max(active_sigma)))


def _audit_iv(layer, iv_totals):
    cols_u8 = layer["cols_u8"]
    banks_clean = _decompose_signed_int8(layer["weights"])
    n, k = int(cols_u8.shape[0]), int(layer["weights"].shape[1])
    row_groups = [group for group in np.array_split(np.arange(layer["weights"].shape[0]), IV_ROW_GROUPS) if group.size]
    full_scale = float(IV_FULL_SCALE_LEVELS)

    for start in range(0, n, IV_CHUNK_ROWS):
        stop = min(start + IV_CHUNK_ROWS, n)
        chunk = cols_u8[start:stop]
        chunk_tile_columns = np.zeros((stop - start, k, len(row_groups)), dtype=bool)
        chunk_tiles = np.zeros((stop - start, len(row_groups)), dtype=bool)

        for group_idx, rows in enumerate(row_groups):
            a0 = (chunk[:, rows] & 0xF).astype(np.int32, copy=False)
            a1 = (chunk[:, rows] >> 4).astype(np.int32, copy=False)
            for act in (a0, a1):
                for bank in banks_clean:
                    branch = act @ bank[rows, :].astype(np.int32, copy=False)
                    iv_totals["max_branch_level"] = max(
                        int(iv_totals["max_branch_level"]),
                        int(np.max(branch)) if branch.size else 0,
                    )
                    upper = branch > full_scale
                    upper_count = int(np.count_nonzero(upper))
                    iv_totals["upper_saturation_events"] += upper_count
                    if upper_count:
                        chunk_tile_columns[:, :, group_idx] |= upper
                        chunk_tiles[:, group_idx] |= np.any(upper, axis=1)

        iv_totals["clamped_tile_columns"] += int(np.count_nonzero(chunk_tile_columns))
        iv_totals["saturated_tiles"] += int(np.count_nonzero(chunk_tiles))


def _sar_pack_magnitude(values, tdc_pe):
    fine_bits = tdc_pe._tdc_effective_fine_bits()
    coarse_budget = max(tdc_pe._tdc_effective_coarse_bits(), 1)
    coarse_bits = min(max(int(tdc_pe.tdc_config.coarse_bits), 1), coarse_budget)
    max_coarse = (1 << coarse_bits) - 1
    mantissa_base = 1 << fine_bits
    max_mantissa = (1 << (fine_bits + 1)) - 1
    gain_shift = max(int(tdc_pe.tdc_config.gain_shift), 0)
    gain_scale = float(1 << gain_shift)

    scaled = np.maximum(np.asarray(values, dtype=np.float64), 0.0) / gain_scale
    active = scaled >= 0.5
    mantissa = np.zeros(scaled.shape, dtype=np.int32)
    exponent = np.zeros(scaled.shape, dtype=np.int32)
    clipped_total = 0
    if not np.any(active):
        return mantissa, exponent, clipped_total

    scaled_active = scaled[active]
    coarse_active = np.floor(np.log2(scaled_active)).astype(np.int32)
    coarse_active = np.maximum(coarse_active, 0)
    clipped_high = coarse_active > max_coarse
    coarse_active = np.minimum(coarse_active, max_coarse)

    step_exp = coarse_active - fine_bits
    mantissa_active = np.rint(np.ldexp(scaled_active, -step_exp)).astype(np.int32)
    mantissa_active = np.maximum(mantissa_active, mantissa_base)

    overflow = mantissa_active >= (1 << (fine_bits + 1))
    if np.any(overflow):
        can_increment = coarse_active < max_coarse
        inc = overflow & can_increment
        coarse_active[inc] += 1
        mantissa_active[inc] = mantissa_base
        clipped_high |= overflow & ~can_increment

    clipped_total = int(np.count_nonzero(clipped_high))
    mantissa_active = np.minimum(mantissa_active, max_mantissa)
    mantissa_active[clipped_high] = max_mantissa
    mantissa[active] = mantissa_active
    exponent[active] = coarse_active + gain_shift
    return mantissa, exponent, clipped_total


def _audit_tdc(layer, tdc_totals):
    cols_signed = layer["cols_u8"].astype(np.int16) - int(layer["activation_offset"])
    weights = layer["weights"].astype(np.int32, copy=False)
    n = int(cols_signed.shape[0])
    tdc_pe = _make_crossbar_quiet(
        input_size=64,
        output_size=64,
        adc_bits=8,
        cell_bits=4,
        safety_checks=False,
        iv_config=_iv_config(),
        tdc_config=_tdc_config(),
    )
    fine_bits = tdc_pe._tdc_effective_fine_bits()
    full_scale = float(IV_FULL_SCALE_LEVELS)

    row_folds = [
        np.arange(start, min(start + 64, weights.shape[0]))
        for start in range(0, weights.shape[0], 64)
    ]
    for rows in row_folds:
        w_fold = weights[rows, :]
        for start in range(0, n, TDC_CHUNK_ROWS):
            stop = min(start + TDC_CHUNK_ROWS, n)
            diff = cols_signed[start:stop, rows].astype(np.int32, copy=False) @ w_fold
            pos = np.clip(np.maximum(diff, 0), 0, full_scale)
            neg = np.clip(np.maximum(-diff, 0), 0, full_scale)
            target = np.rint(pos - neg).astype(np.int64)

            pos_mant, pos_exp, pos_clips = _sar_pack_magnitude(pos, tdc_pe)
            neg_mant, neg_exp, neg_clips = _sar_pack_magnitude(neg, tdc_pe)
            mantissa = pos_mant.astype(np.int64) - neg_mant.astype(np.int64)
            exponent = np.maximum(pos_exp, neg_exp).astype(np.int32)
            recon = np.rint(np.ldexp(mantissa.astype(np.float64), exponent - fine_bits)).astype(np.int64)
            error = np.abs(recon - target)

            values = int(error.size)
            tdc_totals["output_values_converted"] += values
            tdc_totals["wire_bits"] += values * int(tdc_pe.tdc_config.signed_payload_bits)
            tdc_totals["extra_sar_range_clips_after_iv_limiting"] += int(pos_clips + neg_clips)
            tdc_totals["abs_error_sum"] += int(np.sum(error, dtype=np.int64))
            tdc_totals["abs_error_max"] = max(
                int(tdc_totals["abs_error_max"]),
                int(np.max(error)) if error.size else 0,
            )


def _format_int(value):
    return f"{int(value):,}"


def _format_float(value):
    return f"{float(value):.6f}"


def _build_markdown(summary):
    rows = [
        {
            "Effect": "Reliability",
            "Enabled hardware model": (
                "30-day retention drift, 0.02 log-normal state spread, "
                "10 ppm raw BER with ECC"
            ),
            "Main measured error / stress": (
                f"{_format_int(summary['reliability']['programmed_cell_reads_audited'])} programmed cell reads audited; "
                f"{_format_int(summary['reliability']['changed_cell_level_reads'])} cell-level reads changed after the retention/state model; "
                f"{_format_int(summary['reliability']['raw_ber_bit_flips_after_ecc'])} raw BER flips after ECC; "
                f"{_format_int(summary['reliability']['effective_int8_weight_read_entries_changed'])} effective INT8 weight read entries changed; "
                f"mean absolute integer branch error {_format_float(summary['reliability']['mean_abs_integer_branch_error'])}; "
                f"max {_format_int(summary['reliability']['max_abs_integer_branch_error'])}"
            ),
        },
        {
            "Effect": "Noise",
            "Enabled hardware model": (
                "Thermal 4 k T / R, shot 2 q I * fano, flicker/1/f, RTN, "
                "readout RMS, 5 kHz bandwidth integration"
            ),
            "Main measured error / stress": (
                f"Mean equivalent branch read-noise sigma {_format_float(summary['noise']['mean_equivalent_branch_read_noise_sigma_levels'])} "
                f"conductance levels; max {_format_float(summary['noise']['max_equivalent_branch_read_noise_sigma_levels'])} levels"
            ),
        },
        {
            "Effect": "IV / ADC",
            "Enabled hardware model": (
                f"hard_clamp, {IV_ROW_GROUPS} row groups, full-scale {int(IV_FULL_SCALE_LEVELS):,} "
                "product levels before digitization"
            ),
            "Main measured error / stress": (
                f"{_format_int(summary['iv_adc']['upper_saturation_events'])} upper saturation events; "
                f"{_format_int(summary['iv_adc']['lower_clip_events'])} lower clips; "
                f"{_format_int(summary['iv_adc']['clamped_tile_columns'])} clamped tile-columns; "
                f"{_format_int(summary['iv_adc']['saturated_tiles'])} saturated tiles; "
                f"max branch level {_format_int(summary['iv_adc']['max_branch_level'])}"
            ),
        },
        {
            "Effect": "TDC",
            "Enabled hardware model": (
                "9-bit signed-payload SAR-FP TDC, gain_shift=0"
            ),
            "Main measured error / stress": (
                f"{_format_int(summary['tdc']['output_values_converted'])} output values converted; "
                f"{_format_int(summary['tdc']['wire_bits'])} wire bits; "
                f"{_format_int(summary['tdc']['extra_sar_range_clips_after_iv_limiting'])} extra SAR range clips after IV limiting; "
                f"mean absolute integer readout error {_format_float(summary['tdc']['mean_abs_integer_readout_error'])}; "
                f"max {_format_int(summary['tdc']['max_abs_integer_readout_error'])}"
            ),
        },
    ]
    lines = [
        "| Effect | Enabled hardware model | Main measured error / stress |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Effect']} | {row['Enabled hardware model']} | {row['Main measured error / stress']} |"
        )
    return "\n".join(lines) + "\n"


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    executor = FullInferenceExecutor(
        ROOT / "yolo26n_int8.tflite",
        mapping_path=ROOT / "Yolo26n_full_model_spec.m",
        preprocess_mode="plain-resize",
    )
    input_tensor, preprocess_meta = executor.preprocess_image(ROOT / "bus.jpg")
    executor._invoke_reference(input_tensor)
    eligible_layers, ineligible_layers = executor.crossbar_eligible_layers()

    reliability_totals = {
        "layers_audited": 0,
        "programmed_unique_cell_levels": 0,
        "programmed_cell_reads_audited": 0,
        "changed_cell_level_reads": 0,
        "raw_ber_bit_flips_after_ecc": 0,
        "effective_int8_weight_read_entries_changed": 0,
        "branch_abs_error_sum": 0,
        "branch_abs_error_max": 0,
        "branch_error_denominator": 0,
    }
    noise_totals = {
        "weighted_sigma_sum": 0.0,
        "weighted_sigma_count": 0,
        "max_sigma": 0.0,
    }
    iv_totals = {
        "upper_saturation_events": 0,
        "lower_clip_events": 0,
        "clamped_tile_columns": 0,
        "saturated_tiles": 0,
        "max_branch_level": 0,
    }
    tdc_totals = {
        "output_values_converted": 0,
        "wire_bits": 0,
        "extra_sar_range_clips_after_iv_limiting": 0,
        "abs_error_sum": 0,
        "abs_error_max": 0,
    }
    layer_rows = []

    conv_ops = [
        op for op in executor.ops
        if str(op["op_name"]) in {"CONV_2D", "DEPTHWISE_CONV_2D"}
        and executor.conv_layer_names.get(int(op["index"]), "") in eligible_layers
    ]
    for idx, op in enumerate(conv_ops, start=1):
        layer_name = executor.conv_layer_names[int(op["index"])]
        layer = _prepare_layer_tensors(executor, op, layer_name)
        print(
            f"[{idx:03d}/{len(conv_ops):03d}] {layer_name}: "
            f"{layer['out_y'] * layer['out_x']:,} positions x {layer['output_channels']} channels, "
            f"patch={layer['patch_len']}",
            flush=True,
        )

        before_reads = reliability_totals["programmed_cell_reads_audited"]
        before_iv = iv_totals["upper_saturation_events"]
        before_tdc_values = tdc_totals["output_values_converted"]

        _audit_reliability_and_noise(layer, reliability_totals, noise_totals)
        _audit_iv(layer, iv_totals)
        _audit_tdc(layer, tdc_totals)
        reliability_totals["layers_audited"] += 1

        layer_rows.append(
            {
                "layer": layer_name,
                "op_name": layer["op_name"],
                "output_positions": layer["out_y"] * layer["out_x"],
                "output_channels": layer["output_channels"],
                "patch_len": layer["patch_len"],
                "cell_reads_audited": reliability_totals["programmed_cell_reads_audited"] - before_reads,
                "iv_upper_saturation_events": iv_totals["upper_saturation_events"] - before_iv,
                "tdc_output_values_converted": tdc_totals["output_values_converted"] - before_tdc_values,
            }
        )

    reliability_summary = {
        **reliability_totals,
        "mean_abs_integer_branch_error": (
            float(reliability_totals["branch_abs_error_sum"]) / float(reliability_totals["branch_error_denominator"])
            if reliability_totals["branch_error_denominator"]
            else 0.0
        ),
        "max_abs_integer_branch_error": int(reliability_totals["branch_abs_error_max"]),
        "config": _reliability_config().as_dict(),
    }
    noise_summary = {
        "mean_equivalent_branch_read_noise_sigma_levels": (
            float(noise_totals["weighted_sigma_sum"]) / float(noise_totals["weighted_sigma_count"])
            if noise_totals["weighted_sigma_count"]
            else 0.0
        ),
        "max_equivalent_branch_read_noise_sigma_levels": float(noise_totals["max_sigma"]),
        "weighted_cell_level_reads": int(noise_totals["weighted_sigma_count"]),
        "config": _noise_config().as_dict(),
    }
    iv_summary = {
        **iv_totals,
        "row_groups": IV_ROW_GROUPS,
        "config": _iv_config().as_dict(),
    }
    tdc_summary = {
        "output_values_converted": int(tdc_totals["output_values_converted"]),
        "wire_bits": int(tdc_totals["wire_bits"]),
        "extra_sar_range_clips_after_iv_limiting": int(tdc_totals["extra_sar_range_clips_after_iv_limiting"]),
        "mean_abs_integer_readout_error": (
            float(tdc_totals["abs_error_sum"]) / float(tdc_totals["output_values_converted"])
            if tdc_totals["output_values_converted"]
            else 0.0
        ),
        "max_abs_integer_readout_error": int(tdc_totals["abs_error_max"]),
        "config": _tdc_config().as_dict(),
        "iv_limiter": _iv_config().as_dict(),
    }
    summary = {
        "metadata": {
            "model": str(ROOT / "yolo26n_int8.tflite"),
            "image": str(ROOT / "bus.jpg"),
            "mapping": str(ROOT / "Yolo26n_full_model_spec.m"),
            "preprocess": preprocess_meta,
            "crossbar_candidate_layers": len(executor.crossbar_candidate_layers()),
            "eligible_layers_audited": len(conv_ops),
            "ineligible_layers": ineligible_layers,
            "activation_source": "LiteRT reference tensors with the same INT8 bridge used by the crossbar executor",
        },
        "reliability": reliability_summary,
        "noise": noise_summary,
        "iv_adc": iv_summary,
        "tdc": tdc_summary,
        "layer_summary_csv": str(RESULT_DIR / "layer_summary.csv"),
    }

    with open(RESULT_DIR / "layer_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0].keys()))
        writer.writeheader()
        writer.writerows(layer_rows)

    markdown = _build_markdown(summary)
    (RESULT_DIR / "summary.md").write_text(markdown, encoding="utf-8")
    (RESULT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")

    print(f"Saved all-layer nonlinearity audit to: {RESULT_DIR}")
    print(markdown)


if __name__ == "__main__":
    main()
