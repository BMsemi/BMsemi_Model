"""
Functional tests for CrossbarLinear.

Tests:
  1. shape
  2. reference integer-matmul equivalence
  3. UQ3.2 vs INT8 finite/different outputs
  4. program-time sigma noise
  5. from_linear transpose handling
  6. activation quantization precision
  7. I-V / ADC saturation
  8. tile-count stats using LLM_ReRAM's configured tile size
  9. reliability metadata, ECC latency, and endurance-limit accounting
 10. differential A-B / B-A TDC output
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from crossbar_wrapper import (
    CrossbarArray,
    CrossbarLinear,
    CrossbarReliabilityConfig,
    CrossbarTDCConfig,
    TDCPackedOutput,
    quantize_block16,
    quantize_symmetric,
    round_float_format,
    weight_block16,
)

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"


def check(name: str, cond: bool, detail: str = ""):
    tag = PASS if cond else FAIL
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


def make_weight(in_dim: int, out_dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.02, size=(in_dim, out_dim)).astype(np.float32)


def make_input(tokens: int, in_dim: int, seed: int = 1) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.25, size=(tokens, in_dim)).astype(np.float32)
    return torch.from_numpy(x)


def clean_crossbar_kwargs():
    return {"noise_config": {"enabled": False}, "iv_config": {"enabled": False}}


def crossbar_reference(
    w_in_out: np.ndarray,
    x: np.ndarray,
    weight_bits: int = 5,
    act_bits: int = 5,
    block_size: int = 16,
    float_format: str = "bf16",
) -> np.ndarray:
    original_shape = x.shape[:-1]
    x_2d = round_float_format(x.reshape(-1, x.shape[-1]), float_format)
    w_in_out = round_float_format(w_in_out, float_format)

    if weight_bits < 8:
        w_q_pre = weight_block16(w_in_out, bits=weight_bits, block_size=block_size)
    else:
        w_q_pre = w_in_out.copy()

    w_q, w_s = quantize_symmetric(w_q_pre)
    x_a = quantize_block16(x_2d, bits=act_bits, block_size=block_size)
    x_q, x_s = quantize_symmetric(x_a)
    y_int = x_q.astype(np.int64) @ w_q.astype(np.int64)
    y = y_int.astype(np.float32) * x_s * w_s
    return round_float_format(y.reshape(*original_shape, w_in_out.shape[1]), float_format)


def test_shape():
    print("\n[1] Shape test")
    cases = [(4, 64, 32), (1, 64, 64), (8, 128, 64), (3, 64, 128), (1, 128, 256)]
    all_ok = True
    for tokens, in_dim, out_dim in cases:
        w = make_weight(in_dim, out_dim)
        x = make_input(tokens, in_dim)
        layer = CrossbarLinear(w)
        y = layer(x)
        ok = y.shape == (tokens, out_dim)
        all_ok &= check(f"({tokens},{in_dim})->({tokens},{out_dim})", ok, f"got {tuple(y.shape)}")
    return all_ok


def test_default_physical_models_enabled():
    print("\n[2] Default physical models test")
    w = make_weight(64, 32, seed=16)
    layer = CrossbarLinear(w)

    ok = True
    ok &= check("thermal/shot/flicker/RTN noise defaults enabled", layer.noise_config.enabled)
    ok &= check("thermal default is nonzero", layer.noise_config.temp_k > 0.0)
    ok &= check("shot default is nonzero", layer.noise_config.fano > 0.0)
    ok &= check(
        "flicker defaults are nonzero",
        layer.noise_config.flicker_af_lrs > 0.0 and layer.noise_config.flicker_af_hrs > 0.0,
    )
    ok &= check(
        "RTN defaults are nonzero",
        layer.noise_config.rtn_amp_frac_lrs > 0.0 and layer.noise_config.rtn_amp_frac_hrs > 0.0,
    )
    ok &= check("I-V / ADC saturation defaults enabled", layer.iv_config.enabled)
    ok &= check("default saturation mode is hard clamp", layer.iv_config.mode == "hard_clamp")
    return ok


def test_reference():
    print("\n[3] Reference match test")
    cases = [(4, 64, 32, 5, 5), (4, 64, 32, 8, 8), (2, 128, 64, 5, 8), (1, 64, 128, 5, 5)]
    all_ok = True
    for tokens, in_dim, out_dim, wb, ab in cases:
        w = make_weight(in_dim, out_dim, seed=2)
        x = make_input(tokens, in_dim, seed=3)
        layer = CrossbarLinear(w, weight_bits=wb, act_bits=ab, **clean_crossbar_kwargs())
        y_cross = layer(x).numpy()
        y_ref = crossbar_reference(w, x.numpy(), weight_bits=wb, act_bits=ab)
        max_err = np.abs(y_cross - y_ref).max()
        ok = max_err < 1e-3
        all_ok &= check(f"wb={wb} ab={ab} ({tokens},{in_dim})->out={out_dim}", ok, f"max_err={max_err:.2e}")
    return all_ok


def test_uq32_vs_int8():
    print("\n[4] UQ3.2 vs INT8")
    tokens, in_dim, out_dim = 4, 128, 64
    w = make_weight(in_dim, out_dim, seed=4)
    x = make_input(tokens, in_dim, seed=5)

    y5 = CrossbarLinear(w, weight_bits=5, **clean_crossbar_kwargs())(x).numpy()
    y8 = CrossbarLinear(w, weight_bits=8, **clean_crossbar_kwargs())(x).numpy()
    diff = np.abs(y5 - y8).mean()

    ok = True
    ok &= check("UQ3.2 output is finite", np.all(np.isfinite(y5)))
    ok &= check("INT8 output is finite", np.all(np.isfinite(y8)))
    ok &= check("outputs differ", diff > 0, f"mean |delta|={diff:.4f}")
    ok &= check("outputs are close", diff < 0.5, f"mean |delta|={diff:.4f}")
    return ok


def test_noise():
    print("\n[5] Noise injection test")
    tokens, in_dim, out_dim = 4, 64, 32
    w = make_weight(in_dim, out_dim, seed=6)
    x = make_input(tokens, in_dim, seed=7)

    y_clean = CrossbarLinear(w, sigma=0.00, **clean_crossbar_kwargs())(x).numpy()
    y_clean2 = CrossbarLinear(w, sigma=0.00, **clean_crossbar_kwargs())(x).numpy()
    y_noisy = CrossbarLinear(w, sigma=0.03, **clean_crossbar_kwargs())(x).numpy()
    y_noisy2 = CrossbarLinear(w, sigma=0.03, seed=99, **clean_crossbar_kwargs())(x).numpy()

    ok = True
    ok &= check("sigma=0.0 is deterministic", np.allclose(y_clean, y_clean2, atol=1e-6))
    ok &= check(
        "sigma=0.03 changes output",
        not np.allclose(y_clean, y_noisy, atol=1e-4),
        f"mean |delta|={np.abs(y_clean - y_noisy).mean():.4f}",
    )
    ok &= check("different seeds give different noise", not np.allclose(y_noisy, y_noisy2, atol=1e-4))
    return ok


def test_from_linear():
    print("\n[6] from_linear() test")
    in_dim, out_dim = 64, 32
    linear = nn.Linear(in_dim, out_dim, bias=False)
    nn.init.normal_(linear.weight, 0.0, 0.02)
    x = make_input(4, in_dim)

    layer_from = CrossbarLinear.from_linear(linear, **clean_crossbar_kwargs())
    w_in_out = linear.weight.detach().float().numpy().T
    layer_manual = CrossbarLinear(w_in_out, **clean_crossbar_kwargs())

    y_from = layer_from(x).numpy()
    y_manual = layer_manual(x).numpy()

    ok = True
    ok &= check("from_linear matches direct construction", np.allclose(y_from, y_manual, atol=1e-5))
    ok &= check("output shape", y_from.shape == (4, out_dim), f"got {y_from.shape}")
    ok &= check("in_features / out_features set correctly", layer_from.in_features == in_dim and layer_from.out_features == out_dim)
    return ok


def test_act_quant():
    print("\n[7] Activation quantization precision test")
    tokens, in_dim, out_dim = 4, 64, 32
    w = make_weight(in_dim, out_dim, seed=8)
    x = make_input(tokens, in_dim, seed=9)
    ref = x.numpy() @ w

    y5 = CrossbarLinear(w, weight_bits=8, act_bits=5, **clean_crossbar_kwargs())(x).numpy()
    y8 = CrossbarLinear(w, weight_bits=8, act_bits=8, **clean_crossbar_kwargs())(x).numpy()
    err5 = np.abs(y5 - ref).mean()
    err8 = np.abs(y8 - ref).mean()

    ok = True
    ok &= check("A8 closer to float ref than A5", err8 < err5, f"err_A5={err5:.4f} err_A8={err8:.4f}")
    ok &= check("A5 error < 5% of signal RMS", err5 < 0.05 * np.sqrt((ref**2).mean()))
    return ok


def test_iv_saturation():
    print("\n[8] I-V nonlinearity / ADC saturation test")
    tokens, in_dim, out_dim = 2, 64, 32
    w = make_weight(in_dim, out_dim, seed=10)
    x = make_input(tokens, in_dim, seed=11)

    clean = CrossbarLinear(w, weight_bits=8, act_bits=8, **clean_crossbar_kwargs())(x).numpy()
    clamped_layer = CrossbarLinear(
        w,
        weight_bits=8,
        act_bits=8,
        noise_config={"enabled": False},
        enable_iv_nonlinearity=True,
        iv_nonlinearity_mode="hard_clamp",
        iv_adc_full_scale_levels=1.0,
    )
    clamped = clamped_layer(x).numpy()

    ok = True
    ok &= check("iv_config is enabled", clamped_layer.iv_config.enabled)
    ok &= check("clamped output is finite", np.all(np.isfinite(clamped)))
    ok &= check(
        "ADC clamp changes output",
        not np.allclose(clean, clamped, atol=1e-6),
        f"mean |delta|={np.abs(clean - clamped).mean():.4f}",
    )
    return ok


def test_stats():
    print("\n[9] Tile count / stats test")
    cases = [(64, 32), (128, 64), (64, 96), (200, 100)]
    all_ok = True
    for in_dim, out_dim in cases:
        w = make_weight(in_dim, out_dim)
        layer = CrossbarLinear(w)
        st = layer.stats(tokens=1)
        expected_tiles = math.ceil(in_dim / layer._executor.row_tile) * math.ceil(out_dim / layer._executor.col_tile)
        ok = st.crossbar_tiles == expected_tiles
        all_ok &= check(f"in={in_dim} out={out_dim}: tiles={st.crossbar_tiles}", ok, f"expected {expected_tiles}")
    return all_ok


def test_reliability_accounting():
    print("\n[10] Reliability accounting / retention model test")
    rng = np.random.default_rng(12)
    weights = rng.integers(-64, 65, size=(64, 32), dtype=np.int16).astype(np.int8)
    activations = rng.integers(0, 256, size=(64,), dtype=np.uint8)
    cfg = CrossbarReliabilityConfig(
        enabled=True,
        raw_ber_ppm=10_000.0,
        use_ecc=True,
        ecc_codeword_bits=38,
        ecc_data_bits=32,
        ecc_encode_latency_cycles=2,
        ecc_decode_latency_cycles=3,
        retention_time_s=30.0 * 24.0 * 3600.0,
        retention_reference_s=1.0,
        retention_model="state_power_law",
        retention_lrs_beta_log10_per_decade=-0.02556,
        retention_hrs_beta_log10_per_decade=-0.04041,
        endurance_write_cycles=100_000,
        endurance_cycles=100_000,
        endurance_limit_multilevel_loss_prob=1.0,
        endurance_limit_multilevel_sigma_ln=0.30,
    )

    crossbar = CrossbarArray(reliability_config=cfg, reliability_seed=13)
    crossbar.program_weights(weights)
    output = crossbar.compute(activations)
    info = crossbar.get_info()
    stats = info["reliability_stats"]
    latency = info["latency_model"]

    ok = True
    ok &= check("output shape", output.shape == (64,), f"got {output.shape}")
    ok &= check("output is finite", np.all(np.isfinite(output)))
    ok &= check("state-power retention config round-trips", info["reliability_config"]["retention_model"] == "state_power_law")
    ok &= check("programmed cell count tracked", stats["programmed_cells"] == 64 * 32 * 4)
    ok &= check("endurance multilevel-loss cells tracked", stats["endurance_limit_multilevel_loss_cells"] == 64 * 32 * 4)
    ok &= check("ECC program latency tracked", latency["ecc_program_latency_cycles"] > 0)
    ok &= check("ECC compute latency tracked", latency["ecc_compute_latency_cycles"] > 0)
    ok &= check(
        "ECC total latency sums components",
        latency["ecc_total_latency_cycles"] == latency["ecc_program_latency_cycles"] + latency["ecc_compute_latency_cycles"],
    )

    try:
        crossbar.compute(np.full((64,), 300, dtype=np.int16))
        rejected_bad_input = False
    except ValueError:
        rejected_bad_input = True
    ok &= check("PWM input safety check rejects >255", rejected_bad_input)
    return ok


def test_differential_tdc():
    print("\n[11] Differential A-B / B-A TDC test")
    weights = np.zeros((64, 32), dtype=np.int8)
    weights[:8, 0] = np.array([2, -7, 1, 0, -3, 5, -2, 1], dtype=np.int8)
    activations = np.zeros((64,), dtype=np.uint8)
    activations[:8] = np.array([13, 230, 77, 26, 0, 179, 51, 102], dtype=np.uint8)

    crossbar = CrossbarArray(
        noise_config={"enabled": False},
        iv_config={"enabled": False},
        tdc_config=CrossbarTDCConfig(
            enabled=True,
            tdc_bits=8,
            coarse_bits=5,
            fine_bits=3,
            gain_shift=3,
        )
    )
    crossbar.program_weights(weights)
    output = crossbar.compute(activations)
    packed = crossbar.compute_tdc_packed(activations)
    corrected = crossbar.compute_tdc_packed_corrected(activations)
    info = crossbar.get_info()

    raw_plus = 1100
    raw_minus = 1712
    raw = raw_plus - raw_minus
    expected_coarse = 6
    expected_fine = 2
    expected_mantissa = -(2**3 + expected_fine)
    expected_exponent = 3 + expected_coarse
    expected_reconstruction = expected_mantissa * (2 ** (expected_exponent - 3))
    exact_corrected = int((activations.astype(np.int16) - 128)[:8] @ weights[:8, 0].astype(np.int16))

    ok = True
    ok &= check("tdc config is enabled", info["tdc_enabled"])
    ok &= check("tdc gain scale tracked", info["tdc_gain_scale"] == 8)
    ok &= check("SAR-FP is the default TDC output format", packed.format == "tdc_sar_fp")
    ok &= check("local differential TDC reconstruction", int(output[0]) == expected_reconstruction, f"got {int(output[0])}, expected {expected_reconstruction}")
    ok &= check("packed output object returned", isinstance(packed, TDCPackedOutput))
    ok &= check("coarse selects exponent", int(packed.negative_coarse[0]) == expected_coarse and int(packed.exponent[0]) == expected_exponent)
    ok &= check("fine selects mantissa residual", int(packed.negative_fine[0]) == expected_fine and int(packed.mantissa[0]) == expected_mantissa)
    ok &= check("SAR-FP reconstruction is close to raw", abs(int(packed.reconstruct_integer()[0]) - raw) <= 32)
    ok &= check("zero-corrected TDC subtracts offset before FP packing", abs(int(corrected.reconstruct_integer()[0]) - exact_corrected) <= 32)
    ok &= check("unused columns remain zero", np.all(output[1:32] == 0))
    ok &= check("TDC arrival metadata is present", packed.arrival_time_ps is not None)
    ok &= check(
        "larger column output arrives before smaller output",
        float(packed.arrival_time_ps[0]) < float(packed.arrival_time_ps[1]),
    )
    ok &= check(
        "PE completion follows slowest column",
        math.isclose(packed.pe_completion_time_ps, float(np.max(packed.arrival_time_ps))),
    )
    ok &= check("arrival spread captures large-small timing difference", packed.arrival_spread_ps > 0.0)
    ok &= check("completion order reports the early column first", int(packed.completion_order[0]) == 0)
    ok &= check("TDC timing is exposed in PE latency metadata", info["latency_model"]["tdc_timing"]["enabled"])

    legacy = CrossbarArray(
        noise_config={"enabled": False},
        iv_config={"enabled": False},
        tdc_config=CrossbarTDCConfig(
            enabled=True,
            tdc_bits=8,
            coarse_bits=5,
            fine_bits=3,
            gain_shift=3,
            output_format="bfp",
        )
    )
    legacy.program_weights(weights)
    legacy_packed = legacy.compute_tdc_packed(activations)
    ok &= check("legacy BFP TDC remains available", int(legacy_packed.mantissa[0]) == -76 and legacy_packed.exponent == 3)
    return ok


def test_bf16_default_float_boundary():
    print("\n[12] BF16 default PE float-boundary test")
    tokens, in_dim, out_dim = 4, 64, 32
    w = make_weight(in_dim, out_dim, seed=14)
    x = make_input(tokens, in_dim, seed=15)

    default_layer = CrossbarLinear(w, weight_bits=8, act_bits=8, **clean_crossbar_kwargs())
    bf16_layer = CrossbarLinear(w, weight_bits=8, act_bits=8, float_format="bf16", **clean_crossbar_kwargs())
    fp32_layer = CrossbarLinear(w, weight_bits=8, act_bits=8, float_format="fp32", **clean_crossbar_kwargs())

    y_default = default_layer(x).numpy()
    y_bf16 = bf16_layer(x).numpy()
    y_fp32 = fp32_layer(x).numpy()

    ok = True
    ok &= check("default float_format is BF16", default_layer.float_format == "bf16")
    ok &= check("default matches explicit BF16", np.array_equal(y_default, y_bf16))
    ok &= check("FP32 override remains available", fp32_layer.float_format == "fp32")
    ok &= check(
        "BF16 boundary can differ from FP32 boundary",
        not np.allclose(y_bf16, y_fp32, atol=0.0),
        f"mean |delta|={np.abs(y_bf16 - y_fp32).mean():.6g}",
    )
    ok &= check("BF16 boundary output is finite", np.all(np.isfinite(y_default)))
    return ok


def main():
    print("=" * 60)
    print("CrossbarLinear wrapper tests")
    print("=" * 60)

    results = {
        "shape": test_shape(),
        "default_physical_models_enabled": test_default_physical_models_enabled(),
        "reference": test_reference(),
        "uq32_vs_int8": test_uq32_vs_int8(),
        "noise": test_noise(),
        "from_linear": test_from_linear(),
        "act_quant": test_act_quant(),
        "iv_saturation": test_iv_saturation(),
        "stats": test_stats(),
        "reliability_accounting": test_reliability_accounting(),
        "differential_tdc": test_differential_tdc(),
        "bf16_default_float_boundary": test_bf16_default_float_boundary(),
    }

    print("\n" + "=" * 60)
    passed = sum(results.values())
    total = len(results)
    print(f"Result: {passed}/{total} test groups passed")
    if passed < total:
        failed = [k for k, v in results.items() if not v]
        print(f"Failed: {failed}")
    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
