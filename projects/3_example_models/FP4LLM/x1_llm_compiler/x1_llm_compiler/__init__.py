"""FP4LLM to X1 memory compiler NoC toolchain."""

from .mapping import LayerSpec, parse_mapping_file
from .noc import RolePolicy, build_noc_flit, decode_header, role_policy
from .planner import (
    CompilerConfig,
    LayerPlan,
    ProgramManifest,
    TileTask,
    compile_model,
    iter_layer_tasks,
    lower_task_to_csr_ops,
    plan_layer,
)
from .qwen_mapping import qwen_phase_layers
from .fabric import FabricConfig, FabricDescriptor, StreamPlan, build_stream_plan
from .cost_model import (
    CostEstimate,
    CostModelConfig,
    EnergyBreakdown,
    EnergyEngine,
    LatencyBreakdown,
    LatencyEnergyEngine,
    LatencyEngine,
)
from .mapping_optimizer import (
    CandidateEvaluation,
    MappingCandidate,
    MappingOptimizer,
    MappingSearchSpace,
    OptimizationConfig,
    OptimizationResult,
    render_mapping_file,
    write_mapping_file,
)
from .weight_image import WeightImageConfig, emit_weight_image, quantize_weight_from_index

__all__ = [
    "CompilerConfig",
    "LayerSpec",
    "LayerPlan",
    "TileTask",
    "FabricConfig",
    "FabricDescriptor",
    "ProgramManifest",
    "RolePolicy",
    "StreamPlan",
    "WeightImageConfig",
    "CandidateEvaluation",
    "CostEstimate",
    "CostModelConfig",
    "EnergyBreakdown",
    "EnergyEngine",
    "LatencyBreakdown",
    "LatencyEnergyEngine",
    "LatencyEngine",
    "MappingCandidate",
    "MappingOptimizer",
    "MappingSearchSpace",
    "OptimizationConfig",
    "OptimizationResult",
    "render_mapping_file",
    "write_mapping_file",
    "build_stream_plan",
    "build_noc_flit",
    "compile_model",
    "decode_header",
    "emit_weight_image",
    "parse_mapping_file",
    "iter_layer_tasks",
    "lower_task_to_csr_ops",
    "plan_layer",
    "quantize_weight_from_index",
    "qwen_phase_layers",
    "role_policy",
]
