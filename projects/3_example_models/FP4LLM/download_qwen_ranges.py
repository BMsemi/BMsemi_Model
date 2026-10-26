"""Resumable HTTP-range fallback for Qwen weights; verifies Hub SHA-256."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path("D:/hf_models/Qwen--Qwen3-1.7B")
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
CHUNK = 16 * 1024 * 1024


def fetch(item):
    name, start, end = item
    path = ROOT / ".ranges" / f"{name}.{start}"
    if path.exists() and path.stat().st_size == end - start + 1:
        return path
    url = f"https://huggingface.co/Qwen/Qwen3-1.7B/resolve/{REVISION}/{name}?range_start={start}"
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=45) as response:
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                    raise ValueError("Server did not return requested byte range")
                data = response.read()
            if len(data) != end - start + 1:
                raise ValueError("Incomplete range")
            path.write_bytes(data)
            return path
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    (ROOT / ".ranges").mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://huggingface.co/api/models/Qwen/Qwen3-1.7B/revision/{REVISION}?blobs=true", timeout=30) as response:
        info = json.load(response)
    for entry in info["siblings"]:
        name = entry["rfilename"]
        if name not in {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}:
            continue
        target = ROOT / name
        if target.exists():
            with target.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() == entry["lfs"]["sha256"]:
                    continue
        size = entry["size"]
        jobs = [(name, start, min(start + CHUNK, size) - 1) for start in range(0, size, CHUNK)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            for i, _ in enumerate(pool.map(fetch, jobs), 1):
                print(f"{name}: {i}/{len(jobs)} chunks", flush=True)
        digest = hashlib.sha256()
        temporary = target.with_suffix(".partial")
        with temporary.open("wb") as handle:
            for job in jobs:
                chunk_path = fetch(job)
                data = chunk_path.read_bytes()
                digest.update(data)
                handle.write(data)
                handle.flush()
                if chunk_path.resolve().parent != (ROOT / ".ranges").resolve():
                    raise ValueError("Unexpected chunk path")
                chunk_path.unlink()
        if digest.hexdigest() != entry["lfs"]["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {name}")
        temporary.replace(target)
        print(f"Verified {name}", flush=True)
    (ROOT / "download_revision.json").write_text(json.dumps({"revision": REVISION}), encoding="utf-8")


if __name__ == "__main__":
    main()
