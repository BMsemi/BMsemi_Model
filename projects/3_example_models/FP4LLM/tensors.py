import json
import struct
from pathlib import Path


MODEL_PATH = Path(r"C:\Users\Avadh\Downloads\model.safetensors")
OUT_PATH = Path(r"C:\Users\Avadh\Downloads\model_tensor_list.txt")


def read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(header_size))


header = read_safetensors_header(MODEL_PATH)
tensors = [(name, info) for name, info in header.items() if name != "__metadata__"]

with OUT_PATH.open("w", encoding="utf-8") as out:
    out.write(f"file: {MODEL_PATH}\n")
    out.write(f"size_bytes: {MODEL_PATH.stat().st_size}\n")
    out.write(f"metadata: {header.get('__metadata__', {})}\n")
    out.write(f"num_tensors: {len(tensors)}\n\n")
    out.write(f"{'tensor_name':80s}  {'shape':24s}  dtype\n")
    out.write("-" * 120 + "\n")

    for name, info in tensors:
        shape = str(info.get("shape"))
        dtype = str(info.get("dtype"))
        out.write(f"{name:80s}  {shape:24s}  {dtype}\n")

print(f"Wrote {len(tensors)} tensor entries to {OUT_PATH}")
