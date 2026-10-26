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
    if "audio_dataset" in params or "audio_base64" in params:
        return infer_audio(params)
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


def infer_audio(params):
    """WAV dataset -> mono MFCC(20x32) -> existing INT8 autoencoder."""
    import base64, binascii, io
    import soundfile as sf
    import librosa
    items = params.get("audio_dataset")
    if items is None:
        items = [{"name": "audio.wav", "audio_base64": params.get("audio_base64")}]
    if not isinstance(items, list) or not 1 <= len(items) <= 20:
        raise ValueError("audio_dataset must contain 1..20 WAV files")
    records = []; total = 0
    for i, item in enumerate(items):
        if not isinstance(item, dict): raise ValueError("Each audio item needs name and audio_base64")
        encoded = item.get("audio_base64")
        if not isinstance(encoded, str) or len(encoded) > 14_000_000: raise ValueError("WAV file must be base64, maximum 10 MB")
        try: raw=base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error): raise ValueError("Invalid audio_base64")
        total += len(raw)
        if len(raw)>10_000_000 or total>20_000_000: raise ValueError("Audio limit: 10 MB per file and 20 MB per dataset")
        try:
            with sf.SoundFile(io.BytesIO(raw)) as audio:
                sr=audio.samplerate
                if audio.format not in ("WAV", "WAVEX") or not 8000 <= sr <= 96000 or audio.channels not in (1,2):
                    raise ValueError("Use mono/stereo WAV at 8..96 kHz")
                if not 1 <= len(audio) <= sr*30: raise ValueError("WAV duration must be greater than 0 and at most 30 seconds")
                y=audio.read(dtype="float32",always_2d=True).mean(axis=1)
        except (RuntimeError, sf.LibsndfileError): raise ValueError("Cannot decode WAV audio")
        if not np.all(np.isfinite(y)): raise ValueError("Audio contains invalid samples")
        mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=20,n_fft=2048,hop_length=512,n_mels=128)
        mfcc=np.pad(mfcc,((0,0),(0,max(0,32-mfcc.shape[1]))))[:,:32]
        result=infer({"mfcc":mfcc.tolist(),"threshold":params.get("threshold"),"layers":bool(params.get("layers",False))})
        records.append({"name":str(item.get("name",f"audio_{i+1}.wav"))[:200],"sample_rate":sr,
                        "duration_seconds":round(len(y)/sr,4),"result":result})
    return {"count":len(records),"samples":records,"preprocessing":{
        "mono":"stereo channels averaged", "sample_rate":"original; no resampling",
        "n_mfcc":20,"n_fft":2048,"hop_length":512,"n_mels":128,
        "frames":32,"selection":"first 32 frames, zero-padded if shorter",
        "scope":"MFCC reconstruction score; not transcription, classification or audio denoising"}}
