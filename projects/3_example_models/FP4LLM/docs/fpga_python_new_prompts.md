# Running a new prompt through FPGA block 0 and Python blocks 1–27

Use the project-root `run_qwen_fpga_continuation.py`. The script saved under
`results/qwen_model_execution_x2/` remains the historical single-token Hello
runner and is not the general-purpose entry point.

This workflow supports unpadded prefill sequences of 1–128 tokens for
Qwen3-1.7B. It outputs the next-token prediction, not a multi-token generated
response. The FPGA still runs only transformer block 0. No bitstream rebuild
is needed for a different supported prompt.

## 1. Export the new input

On the preparation PC, use the exact checkpoint that will be loaded by the
Python continuation. The exporter needs NumPy and Hugging Face tokenizers.

```sh
python export_fpga_bundle.py --checkpoint /path/to/Qwen3-1.7B --prompt "Explain matrix multiplication." --out block1_new_prompt.npz --max-tokens 128
```

Use the same prompt text in the board command. Token counts are not word counts;
the exporter rejects an oversized prompt rather than silently truncating it.

## 2. Run the first block on the board

Transfer the new bundle to the board. In the prepared board project directory,
using the validated DMA-heap runtime:

```sh
sudo -n /usr/local/share/pynq-venv/bin/python3 fpga/qwen6/board_dma_heap.py main.py --execution-target fpga6 --block-index 0 --prompt "Explain matrix multiplication." --fpga-bundle block1_new_prompt.npz --fpga-bitstream fpga/qwen6/overlay/qwen6.bit --fpga-driver fpga/qwen6/axi_x2_mvm_driver.py --out-dir results/new_prompt_fpga
```

Retrieve both `fpga_block1_prefill.npz` and `fpga_summary.json` into a new input
folder on the Python machine. The summary contains the actual prompt, token IDs,
positions and model configuration, so a Hello-specific sidecar is unnecessary.
Never pair a new summary with an old output array. Optional `source.json` and
legacy `hello.source.json` files must agree with the summary when present.

## 3. Continue in Python

From the FP4LLM project root, in an environment with PyTorch, Transformers and
the existing software-crossbar dependencies:

```sh
python run_qwen_fpga_continuation.py --input-package results/new_prompt_fpga --out-dir results/new_prompt_continuation --model /path/to/Qwen3-1.7B --threads 8
```

Use a new, empty output directory. The default `wrapper` backend retains the
existing detailed crossbar simulation. This change does not implement a faster
backend. CPU threads do not make dependent transformer blocks execute in parallel.

The runner validates the physical block-0 prefill summary, 1–128 token IDs,
positions, finite `[tokens, 2048]` output, local tokenizer/configuration and mapping
capacity. It builds `[1, tokens]` masks and positions `0..tokens-1`; the model
constructs the causal attention mask. It executes blocks 1–27, final norm and the
LM head, choosing the next token from the final prompt position.

Use identical checkpoint weights on both sides. Tokenizer/configuration checks
alone cannot establish weight identity. For provenance containing a pinned
Hugging Face revision, `--expected-revision <40-character-revision>` additionally
checks the revision recorded by the FPGA; it does not hash local model weights.

FPGA W8/A8 arithmetic and the default software W7/A7 continuation remain a
mixed-arithmetic experiment. Decode and multi-token response generation remain
unsupported by this continuation runner.

## Validation

```sh
python -m unittest test_fpga_continuation_input -v
```

These tests validate real Hello handoff compatibility and synthetic multi-token
handoff boundaries, shapes, positions and mismatch rejection. They do not run a
physical board or a full 28-block model.
