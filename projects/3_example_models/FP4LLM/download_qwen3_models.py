from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download


MODELS = [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen3 model snapshots.")
    parser.add_argument("--cache-dir", default=r"D:\hf_cache\hub")
    parser.add_argument("--local-root", default=r"D:\hf_models")
    parser.add_argument("--model", action="append", choices=MODELS)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    local_root = Path(args.local_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_root.mkdir(parents=True, exist_ok=True)
    models = args.model or MODELS

    for model_id in models:
        print(f"\n=== Downloading {model_id} ===", flush=True)
        local_dir = local_root / model_id.replace("/", "--")
        path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_dir=local_dir,
            max_workers=args.max_workers,
            ignore_patterns=[
                "*.h5",
                "*.msgpack",
                "*.ot",
                "flax_model.*",
                "tf_model.*",
            ],
        )
        print(f"cached: {path}", flush=True)


if __name__ == "__main__":
    main()
