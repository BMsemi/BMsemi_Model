#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${FP4LLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

VENV="${FP4LLM_VENV:-$HOME/venvs/fp4llm-qwen3}"
PYTHON_BIN="$VENV/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"

if ! python -m pip show torch transformers huggingface-hub accelerate sentencepiece >/dev/null 2>&1; then
  python -m pip install --upgrade pip wheel 'setuptools<82'
  python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu126
  python -m pip install --upgrade huggingface_hub transformers accelerate sentencepiece
fi

RUN_NAME="Qwen3-1.7B_Transformer_block_1_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/results/$RUN_NAME"
CACHE_DIR="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
LOCAL_ROOT="${HF_MODEL_ROOT:-$HOME/hf_models}"

echo "root=$ROOT_DIR"
echo "venv=$VENV"
echo "out_dir=$OUT_DIR"
echo "cache_dir=$CACHE_DIR"
echo "local_root=$LOCAL_ROOT"

RUN_CMD=$(cat <<EOF
set -euo pipefail
cd "$ROOT_DIR"
source "$VENV/bin/activate"
python Experiments/qwen3_transformer_block1_gpu_reference.py \
  --model Qwen/Qwen3-1.7B \
  --run-name "$RUN_NAME" \
  --block-index 0 \
  --device cuda \
  --dtype auto \
  --cache-dir "$CACHE_DIR" \
  --local-root "$LOCAL_ROOT" \
  --out-dir "$OUT_DIR"
EOF
)

if [ "$(hostname)" = "bose01" ]; then
  bash -lc "$RUN_CMD"
elif [ -n "${DISPLAY:-}" ]; then
  srun --x11 --pty --nodelist=bose01 bash -lc "$RUN_CMD"
else
  echo "DISPLAY is not set; running non-graphical GPU reference without --x11."
  srun --pty --nodelist=bose01 bash -lc "$RUN_CMD"
fi

echo "wrote $OUT_DIR"
