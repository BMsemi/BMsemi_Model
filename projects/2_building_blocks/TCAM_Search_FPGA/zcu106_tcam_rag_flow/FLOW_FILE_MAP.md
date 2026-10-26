# Flow Block to File Map

| Flow block | Files |
|---|---|
| Input image -> fingerprint key | `integrated/fingerprint.py` |
| TCAM RAG controller | `integrated/cache.py`, `integrated/pipeline.py` |
| 32x32 TCAM on Engine 0/1 | `integrated/pe_tcam.py`, `rtl/xforge_shared_pe/axi_x2_mvm_accel.sv` |
| Constants & Manifest + TCAM Cache Index | `integrated/cache.py` |
| TCAM hit skip path | `integrated/pipeline.py` |
| Cached chunk storage | `integrated/cache.py` |
| TCAM query scheduling and bank/page capacity policy | `integrated/hybrid_backend.py`, `integrated/pe_tcam.py` |
| YOLO + MLIR miss path | `run_tcam_yolo.py`, `xforge_zcu106/x2_zcu106/x2compiler/`, `xforge_zcu106/x2_zcu106/runtime/axi_x2_mvm_driver.py` |
| MLIR spatial/temporal output-channel tiling | `integrated/hybrid_backend.py`, `scripts/generate_channel_temporal_manifest.py`, `xforge_zcu106/x2_zcu106/compiler_artifacts/yolo26n_hybrid_calibrated/channel_temporal_manifest.json` |
| Upstream YOLO layers-on-FPGA six-engine experiment | `integrated/hybrid_backend.py`, `run_tcam_yolo.py`, `xforge_zcu106/x2_zcu106/runtime/` |
| Product API layers-on-FPGA contract | `../YOLO26n-TFLite-FPGA-Partner-Project/api.py`, `integrated/hybrid_backend.py` |
| FPGA layer tensor mirror into DDR/L2 | `integrated/hybrid_backend.py`, `integrated/legacy_emulator.py` |
| Neuromorphic and CPU fallback | `integrated/hybrid_backend.py`, `integrated/legacy_emulator.py` |
| Legacy emulator L2 layer storage | `integrated/legacy_emulator.py`, `legacy_neuromorphic_core/Crossbar_array.py`, `legacy_neuromorphic_core/L2_buffer.py` |
| Current MLIR neuromorphic emulator backend | `xforge_zcu106/x2_zcu106/x2compiler/runtime.py` (`CrossbarX2Backend`) |
| Legacy neuromorphic X2 core | `legacy_neuromorphic_core/neuromorphic_x2.py`, `legacy_neuromorphic_core/Multiple_PE_core.py`, `legacy_neuromorphic_core/Crossbar_array.py`, `legacy_neuromorphic_core/L2_buffer.py` |
| INT4 RRAM programming accuracy model | PS: `scripts/model_rram_int4_programming_accuracy.py`, `scripts/run_rram_int4_accuracy_ps.py`; PL: `rtl/rram_int4_programming_accuracy_lut.sv`, `rtl/axi_rram_int4_accuracy.sv`, `rtl/axi_rram_int4_accuracy_bd.v`, `scripts/probe_rram_int4_accuracy_pl.py`; test: `tests/test_rram_int4_model.py` |
| FPGA bitstream and smoke | `deploy/` |
| Six-engine same-flow diagram | `docs/simplified_same_flow_tcam_mlir.png` |
| General, simplified, and detailed diagrams | `docs/` |
