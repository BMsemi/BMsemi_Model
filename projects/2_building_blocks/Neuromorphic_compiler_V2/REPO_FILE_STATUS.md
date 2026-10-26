# Repo file status

This file tells you which files should be pushed, which files are generated,
and which files are only used by tests.

## Push these source files

- `README.md`
- `REPO_FILE_STATUS.md`
- `optimize_onnx_and_map.py`
- `models/tiny_mobilenet_v2.py`
- `utils/checkpoints.py`
- `executorch/executorch_compile_spec.py`
- `executorch/delegate_preprocess.py`
- `executorch/lower_to_execu_runtime.py`
- `compiler/preprocess_onnx.py`
- `compiler/onnx_ir.py`
- `compiler/graph_level_optimizer.py`
- `compiler/weight_legalization.py`
- `compiler/weight_cluster.py`
- `compiler/weight_quant.py`
- `compiler/activation_quant.py`
- `compiler/op_tiler.py`
- `compiler/mapping_assign.py`
- `maestro/dataflow_builder.py`
- `maestro/MobilenetV2_clustered.m`
- `maestro/layer_dataflow_reference.json`
- `maestro/Mapper.py`
- `maestro/Scheduler.py`
- `runtime/global_memory_planner.py`
- `runtime/dispatch.py`
- `runtime/verification_profiling_harness.py`
- `deployment/export_tflite.py`
- `tests/test_tiny_flow.py`
- package marker files: all `__init__.py`
- `.gitignore`

## Generated artifacts

These can be committed if you want reproducible reference outputs, but they are
not required as source code:

- `artifacts/sample_digit.png`
- `artifacts/executorch_compile_spec.json`
- `artifacts/compiler_graph.json`
- `artifacts/fx_graph.json`
- `artifacts/onnx_export_status.json`
- `artifacts/maestro_layer_dataflow.json`
- `artifacts/MobilenetV2_clustered_generated.m`
- `artifacts/maestro_integration_status.json`
- `artifacts/maestro_schedule.json`
- `artifacts/tflite_export_status.json`
- `artifacts/checkpoint_report.json`
- `artifacts/test_run_summary.json`
- `artifacts/run_console.log`
- `checkpoints/tiny_mobilenetv2_digits_best.pt`

## Not used by the main Python flow

- none among the pushed source files

The main flow now actively uses the MAESTRO source references:

- `maestro/MobilenetV2_clustered.m`
- `maestro/layer_dataflow_reference.json`

They are cross-checked against the generated MAESTRO artifacts during runtime.

## Only used in tests

- `tests/test_tiny_flow.py`

## Optional package marker files

- `compiler/__init__.py`
- `executorch/__init__.py`
- `maestro/__init__.py`
- `models/__init__.py`
- `runtime/__init__.py`
- `utils/__init__.py`
- `deployment/__init__.py`
