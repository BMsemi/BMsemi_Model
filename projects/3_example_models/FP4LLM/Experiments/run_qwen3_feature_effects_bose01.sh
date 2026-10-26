#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV="${FP4LLM_VENV:-$HOME/venvs/fp4llm-qwen3}"
PYTHON_BIN="$VENV/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install --upgrade huggingface_hub safetensors numpy pandas transformers datasets accelerate sentencepiece tqdm lm-eval

RUN_NAME="qwen3_feature_effects_bose01_$(date +%Y%m%d_%H%M%S)"
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
python test_crossbar_wrapper.py
python verify_crossbar_qwen_lm_eval.py \
  --model Qwen/Qwen3-1.7B \
  --cache-dir "$CACHE_DIR" \
  --local-root "$LOCAL_ROOT" \
  --dim 16 \
  --out "$OUT_DIR/verify_crossbar_qwen_lm_eval.json"
python Experiments/qwen3_feature_effects.py \
  --model Qwen/Qwen3-1.7B \
  --cache-dir "$CACHE_DIR" \
  --local-root "$LOCAL_ROOT" \
  --out-dir "$OUT_DIR"
EOF
)

if [ "$(hostname)" = "bose01" ]; then
  bash -lc "$RUN_CMD"
else
  srun --x11 --pty --nodelist=bose01 bash -lc "$RUN_CMD"
fi

echo "wrote $OUT_DIR"
