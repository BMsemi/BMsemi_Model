"""Microscaling with the local CrossbarArray simulator; --backend ideal for reference.

Run: python quantized_user.py
Custom arrays: python quantized_user.py --weights W.npy --inputs x.npy
W has shape (input_rows, output_columns); x has shape (input_rows,).
"""
import argparse
import json
from pathlib import Path
import runpy

import numpy as np

from src.microscaling import execute, quantize_inputs, quantize_weights, requantize_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", action="store_true", help="Run original PE_quantization example")
    parser.add_argument("--backend", choices=("crossbar", "ideal"), default="crossbar")
    parser.add_argument("--array-rows", type=int, default=32)
    parser.add_argument("--physical-columns", type=int, default=32)
    parser.add_argument("--adc-bits", type=int, default=10, help="Unsigned column ADC width (default 10 fits 16 rows at 2-bit DAC)")
    parser.add_argument("--disable-noise", action="store_true")
    parser.add_argument("--disable-iv", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crossbar-config", type=Path, help="JSON object with noise, reliability, and iv config sections")
    parser.add_argument("--weights", type=Path, help="Floating-point .npy matrix [input, output]")
    parser.add_argument("--inputs", type=Path, help="Floating-point .npy vector [input]")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dac-bits", type=int, default=2)
    parser.add_argument("--input-scale-frac-bits", type=int)
    parser.add_argument("--output-frac-bits", type=int, help="Optional signed 16-bit output binary point")
    parser.add_argument("--report", type=Path, help="Write scale metadata and per-cycle results to JSON")
    args = parser.parse_args()
    if args.legacy:
        if any((args.weights, args.inputs, args.report, args.output_frac_bits is not None,
                args.input_scale_frac_bits is not None, args.block_size != 16, args.dac_bits != 2,
                args.backend != "crossbar", args.array_rows != 32, args.physical_columns != 32,
                args.adc_bits != 10, args.disable_noise, args.disable_iv,
                args.crossbar_config, args.seed != 42)):
            parser.error("--legacy cannot be combined with microscaling options")
        try:
            runpy.run_path(str(Path(__file__).with_name("quantized_user_legacy.py")), run_name="__main__")
        except ModuleNotFoundError as exc:
            parser.exit(2, f"Legacy mode requires its original dependencies: {exc}\n")
        return
    if (args.weights is None) != (args.inputs is None):
        parser.error("Provide --weights and --inputs together")
    try:
        if args.weights:
            w, x = np.load(args.weights, allow_pickle=False), np.load(args.inputs, allow_pickle=False)
        else:
            # Multiple blocks, unequal scales, signs, and partial edge tiles.
            rng = np.random.default_rng(42)
            w = rng.normal(size=(35, 19))
            w[16:32] *= 0.03
            x = rng.normal(size=35)
            x[16:32] *= 5
        qw = quantize_weights(w, args.block_size)
        qx = quantize_inputs(x, args.block_size, args.input_scale_frac_bits)
        adapter = None
        if args.backend == "crossbar":
            from src.microscaling_crossbar import MicroscalingCrossbar
            from reram_crossbar import CrossbarNoiseConfig, CrossbarReliabilityConfig, CrossbarIVNonlinearityConfig
            from dataclasses import fields
            config = json.loads(args.crossbar_config.read_text(encoding="utf-8")) if args.crossbar_config else {}
            if not isinstance(config, dict) or set(config) - {"noise", "reliability", "iv"}:
                raise ValueError("Crossbar config must contain only noise, reliability, and iv objects")
            classes = {"noise": CrossbarNoiseConfig, "reliability": CrossbarReliabilityConfig,
                       "iv": CrossbarIVNonlinearityConfig}
            for key, cls in classes.items():
                section = config.setdefault(key, {})
                if not isinstance(section, dict) or set(section) - {f.name for f in fields(cls)}:
                    raise ValueError(f"Unknown fields or invalid object in {key} config")
            if args.disable_noise:
                config["noise"]["enabled"] = False
            if args.disable_iv:
                config["iv"]["enabled"] = False
            adapter = MicroscalingCrossbar(rows=args.array_rows, physical_columns=args.physical_columns,
                dac_bits=args.dac_bits, adc_bits=args.adc_bits, seed=args.seed,
                noise_config=config["noise"], reliability_config=config["reliability"], iv_config=config["iv"])
            output, trace = execute(qw, qx, args.dac_bits, read=adapter)
        else:
            if args.crossbar_config or args.disable_noise or args.disable_iv:
                raise ValueError("Crossbar settings require --backend crossbar")
            output, trace = execute(qw, qx, args.dac_bits)
        reference = qx.reconstruct() @ qw.reconstruct()
        if adapter is None:
            np.testing.assert_allclose(output, reference, rtol=1e-12, atol=1e-12)
        original = x @ w
        report = {
            "backend": "CrossbarArray simulator, single-pair INT5 ADC path" if adapter else "ideal differential numerical reference",
            "dac_bits": args.dac_bits,
            "weight_shape": list(w.shape), "block_size": args.block_size,
            "weight_tensor_exponent": qw.tensor_exponent,
            "weight_scale_e4m3_codes": qw.scale_codes.tolist(),
            "weight_block_scales": qw.block_scales.tolist(),
            "input_scale_uint16_codes": qx.scale_codes.tolist(),
            "input_scale_fractional_bits": qx.fractional_bits,
            "input_binary_point_mode": "per-vector reference" if args.input_scale_frac_bits is None else "explicit",
            "quantized_weights": qw.values.tolist(), "quantized_inputs": qx.values.tolist(),
            "output": output.tolist(), "unquantized_reference": original.tolist(),
            "quantized_reference_max_abs_error": float(np.max(np.abs(output - reference))),
            "unquantized_max_abs_error": float(np.max(np.abs(output - original))),
            "blocks": trace,
        }
        if adapter is not None:
            report["crossbar"] = adapter.report()
        if args.output_frac_bits is not None:
            codes, reconstructed, saturated = requantize_output(output, args.output_frac_bits)
            report.update(output_int16=codes.tolist(), output_fractional_bits=args.output_frac_bits,
                          output_reconstructed=reconstructed.tolist(), output_saturated_count=saturated)
        print("Backend:", report["backend"])
        print(f"INT5 weights {w.shape}; INT6 inputs {x.shape}; block size {args.block_size}")
        print("Input scale fractional bits:", qx.fractional_bits)
        print("Scaled output:", output)
        print("Quantized-reference max error:", report["quantized_reference_max_abs_error"])
        print("Original float max error:", report["unquantized_max_abs_error"])
        if adapter is not None:
            print("Crossbar arrays:", report["crossbar"]["programmed_arrays"],
                  "reads:", report["crossbar"]["read_calls"],
                  "ADC clips:", report["crossbar"]["adc_clip_events"])
        if args.output_frac_bits is not None:
            print("INT16 output:", codes, "saturated:", saturated)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print("Report:", args.report.resolve())
    except (ValueError, OSError, AssertionError) as exc:
        parser.exit(2, f"Microscaling error: {exc}\n")


if __name__ == "__main__":
    main()
