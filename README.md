# BMsemi Model Library

Canonical source of model adapters, weights, reference projects and hardware assets for the Neuromorphic Platform.

## Contents
- `catalog.json`: runnable built-in operations and module names.
- `bm_models/`: platform-compatible software adapters and mapping code.
- `projects/`: original foundation, building-block and example-model projects, including Deep Gate, MNIST, TCAM, YOLO26 and FP4LLM/Qwen.
- `hardware/`: ZCU106 graph runner. The current runner supports only the PE self-test; archived model code is not automatically FPGA-compatible.
- `templates/simple-model/`: minimal contribution package.

## Contribute through the platform
Create a GitHub repository containing `model.json` and `model.py` at its root, following the template. Sign into the platform, open Models, import that repository, and submit a contribution. The platform publishes a contribution branch here. An administrator reviews the files and publishes the exact submitted commit. Only published main-branch code is activated for canvas and API users.

`run(params)` must accept a JSON object and return a JSON-serializable dictionary. New contributions belong to the `community` module; an administrator grants that module to users. No dependency installation or source code execution occurs during import. Dependencies must already exist on the platform; request administrator installation when needed. Contributions are reviewed Python code running with the platform process permissions, not a sandbox.

## Licensing
Original copyright and license notices are retained in project files. No blanket license or new rights over third-party models are implied. Contributors must have permission to publish their submitted code, data and weights. Do not submit passwords, access tokens, personal datasets or environment files.

## Runtime provenance
The platform uses a commit-pinned checkout of this repository. Git fetch/import never changes active code until publication or administrator sync. Built-in model adapters use assets within that checkout; local project directories are not a fallback.
