"""Deep_gate (AD01 autoencoder) INT8 inference as a platform operation.

Runs the exact integer path from projects/3_example_models/Deep_gate
(schema.json weights + torch_quantised_linear.py math), i.e. the same
"correctness path" User.py uses.  The multi-PE crossbar simulation is not run
here (it only produces utilisation stats, not the numbers).

params:
  case       int 0-4      use bundled test_cases.json example i as the input
  input      list[640]    OR raw int8 input vector (-128..127)
  mfcc       20x32 floats OR 640 floats, quantised like App.py:
             round(mfcc/input_scale)+input_zp, clipped to int8
  threshold  float        optional; adds is_anomaly = recon_mse > threshold
  layers     bool         include per-layer int8 outputs (default False)
"""
from __future__ import annotations

import importlib.util
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "projects" / "3_example_models" / "Deep_gate"


def _dir() -> Path:
    return DEFAULT_DIR


@lru_cache(maxsize=1)
def _load():
    d = _dir()
    spec = importlib.util.spec_from_file_location("_dg_quant", d / "torch_quantised_linear.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    layers = []
    for L in json.loads((d / "schema.json").read_text()):
        m, so, si = L["module_specs"], L["shape_out"][0], L["shape_in"][0]
        layers.append({
            "W": np.array(m["weights_int8"]["data"], dtype=np.int8).reshape(so, si),
            "bias": np.array(m["bias_int32"]["data"], dtype=np.int32),
            "mult": np.array(m["requant_multiplier"]["data"], dtype=np.int32),
            "shift": np.array(m["requant_shift"]["data"], dtype=np.int32),
            "in_zp": int(m["input_zp"]["data"]), "out_zp": int(m["output_zp"]["data"]),
            "in_scale": float(m["input_scale"]["data"]), "out_scale": float(m["output_scale"]["data"]),
            "n_in": si, "n_out": so,
        })
    cases = json.loads((d / "test_cases.json").read_text())["examples"]
    return mod.torch_quantised_linear_forward_vectorised, layers, cases


def infer(params: dict) -> dict:
    fwd, layers, cases = _load()
    if "mfcc" in params:
        m = np.asarray(params["mfcc"], dtype=np.float64).reshape(-1)
        if m.size != layers[0]["n_in"] or not np.all(np.isfinite(m)):
            raise ValueError(f"mfcc must be {layers[0]['n_in']} finite floats (20x32)")
        x = np.clip(np.round(m / layers[0]["in_scale"]) + layers[0]["in_zp"], -128, 127).astype(np.int8)
        expected = None
    elif "input" in params:
        x = np.asarray(params["input"])
        if x.shape != (layers[0]["n_in"],) or x.size == 0 or x.min() < -128 or x.max() > 127 \
                or not np.all(x == np.rint(x)):
            raise ValueError(f"input must be {layers[0]['n_in']} integers in [-128,127]")
        x = x.astype(np.int8)
        expected = None
    else:
        i = int(params.get("case", 0))
        if not 0 <= i < len(cases):
            raise ValueError(f"case must be 0..{len(cases) - 1}")
        x = np.array(cases[i][0]["output"], dtype=np.int8)
        expected = cases[i]

    x_in, outs = x, []
    max_err = 0.0
    for idx, L in enumerate(layers):
        y = fwd(x.astype(np.int32) - L["in_zp"], L["W"], L["bias"], L["mult"], L["shift"], L["out_zp"], np.int8)
        yf = (y.astype(np.float32) - L["out_zp"]) * L["out_scale"]
        if expected is not None:
            ref = np.array(expected[idx + 1]["output"], dtype=np.float32)
            max_err = max(max_err, float(np.abs(yf - ref).max()))
        outs.append(y)
        x = y

    first, last = layers[0], layers[-1]
    x_f = (x_in.astype(np.float32) - first["in_zp"]) * first["in_scale"]
    y_f = (x.astype(np.float32) - last["out_zp"]) * last["out_scale"]
    mse = float(np.mean((x_f - y_f) ** 2))
    res = {"recon_mse": mse, "output_int8": x.tolist(), "n_layers": len(layers)}
    if params.get("threshold") is not None:     # canvas default is null
        res["is_anomaly"] = mse > float(params["threshold"])
    if expected is not None:
        res["reference_max_abs_err"] = max_err
        res["reference_ok"] = max_err < 0.02
    if params.get("layers"):
        res["layer_outputs_int8"] = [o.tolist() for o in outs]
    return res
