"""TCAM search (TCAM_Search_FPGA / zcu106_tcam_rag_flow) as an operation.

Runs the project's own DualPETcamSearch (differential q/qbar x A/B encoding on
two 32x32 PEs, B-A + TDC readout) with the bit-exact NumpyMvmEngine stand-in
for the FPGA PEs.  Stateless: the caller supplies the entries to search.
The real board path (--mode hardware) is not used here.

params:
  entries   list of up to 32 {key, care?, chunk_id, slot?}   keys 16-bit ints or "0x36C8"
  query     16-bit int / "0x..."      OR
  image_base64   a PNG/JPEG (<= 5 MB); fingerprinted with the project's 16-bit dHash
  query_care      16-bit mask of bits to compare (default 0xFFFF)
  top_k           1..32 (default 4)
  max_hamming     a match within this distance counts as a hit (default 0)
"""
from __future__ import annotations

import base64
import binascii
import importlib
import io
import os
import sys
import types
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_DIR = (Path(__file__).resolve().parents[2] / "projects" / "2_building_blocks"
               / "TCAM_Search_FPGA" / "zcu106_tcam_rag_flow" / "integrated")
MAX_IMAGE_BYTES = 5_000_000
_PKG = __package__ + "._tcam_integrated"


@lru_cache(maxsize=1)
def _mods():
    d = DEFAULT_DIR
    pkg = types.ModuleType(_PKG)            # namespace only: skip integrated/__init__ (pulls in YOLO/MLIR)
    pkg.__path__ = [str(d)]
    sys.modules[_PKG] = pkg
    return (importlib.import_module(f"{_PKG}.pe_tcam"),
            importlib.import_module(f"{_PKG}.fingerprint"))


def _word(v, name: str) -> int:
    try:
        x = int(v, 0) if isinstance(v, str) else (int(v) if int(v) == v and not isinstance(v, bool) else None)
    except (TypeError, ValueError):
        x = None
    if x is None or not 0 <= x <= 0xFFFF:
        raise ValueError(f"{name} must be a 16-bit integer (0..65535 or '0x....')")
    return x


def _fingerprint(b64: str, fp) -> int:
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("image_base64 is not valid base64")
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"image must be 1..{MAX_IMAGE_BYTES} bytes")
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except Exception:
        raise ValueError("could not decode image")
    if im.width * im.height > 40_000_000:
        raise ValueError("image too large")
    bgr = np.asarray(im.convert("RGB"))[:, :, ::-1]      # same BGR order identify_image uses
    return fp.visual_fingerprint(bgr)


def search(params: dict) -> dict:
    pe, fp = _mods()
    raw = params.get("entries")
    if not isinstance(raw, list) or not 1 <= len(raw) <= pe.TOTAL_ENTRIES:
        raise ValueError(f"entries must be a list of 1..{pe.TOTAL_ENTRIES} items")
    entries, used = [], set()
    for i, e in enumerate(raw):
        if not isinstance(e, dict) or "key" not in e or "chunk_id" not in e:
            raise ValueError(f"entries[{i}] needs key and chunk_id")
        slot = i if "slot" not in e else int(e["slot"])
        if slot in used:
            raise ValueError(f"duplicate slot {slot}")
        used.add(slot)
        cid = int(e["chunk_id"])
        te = pe.TcamEntry(slot, _word(e["key"], "key"), _word(e.get("care", 0xFFFF), "care"), cid)
        te.validate()
        entries.append(te)

    if ("query" in params) == ("image_base64" in params):
        raise ValueError("give exactly one of query or image_base64")
    query = _word(params["query"], "query") if "query" in params else _fingerprint(params["image_base64"], fp)
    care = _word(params.get("query_care", 0xFFFF), "query_care")
    top_k = int(params.get("top_k", 4))
    max_h = int(params.get("max_hamming", 0))
    if not 1 <= top_k <= pe.TOTAL_ENTRIES or max_h < 0:
        raise ValueError("top_k must be 1..32 and max_hamming >= 0")

    engines = [pe.NumpyMvmEngine("pe0"), pe.NumpyMvmEngine("pe1")]
    s = pe.DualPETcamSearch(engines)
    s.program(entries)
    matches = s.search(query, query_care=care, top_k=top_k)
    best = matches[0] if matches else None
    hit = bool(best and best.hamming_distance <= max_h and best.compared_bits > 0)
    return {"query": f"0x{query:04X}", "hit": hit,
            "chunk_id": best.chunk_id if hit else None,
            "matches": [asdict(m) for m in matches],
            "engine": "software (NumpyMvmEngine, bit-exact stand-in for the FPGA PEs)",
            "pe_launches": s.pe_launches}
