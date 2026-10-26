from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class GraphLayerSpec:
    name: str
    layer_type: str
    k: int
    c: int
    r: int
    s: int
    y: int
    x: int
    stride_x: int
    stride_y: int
    source_op: str
    source_index: int
    input_name: str
    output_name: str
    weight_name: str


def sanitize_identifier(text: str, default: str = "ParsedNetwork") -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", text.strip())
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = default
    if name[0].isdigit():
        name = f"_{name}"
    return name


def detect_model_format(path: Path) -> str:
    suffix = path.suffix.lower()
    header = path.read_bytes()[:8]
    if len(header) >= 8 and header[4:8] == b"TFL3":
        return "tflite"
    if suffix in {".onnx", ".pb"}:
        return "onnx"
    if suffix in {".tflite", ".lite"}:
        return "tflite"
    raise ValueError(f"Cannot detect model format for {path}")


def parse_model(
    path: str | Path,
    *,
    network_name: str | None = None,
    include_depthwise: bool = True,
) -> tuple[str, list[GraphLayerSpec]]:
    model_path = Path(path)
    model_format = detect_model_format(model_path)
    parser = ONNXGraphParser(model_path, network_name=network_name, include_depthwise=include_depthwise)
    if model_format == "onnx":
        return parser.parse_onnx()
    if model_format == "tflite":
        return parser.parse_tflite()
    raise ValueError(f"Unsupported model format: {model_format}")


def render_maestro_spec(network_name: str, layers: Iterable[GraphLayerSpec]) -> str:
    lines = [f"Network {sanitize_identifier(network_name)} {{"]
    for layer in layers:
        k_tile_hint = max(1, min(8, layer.k))
        lines.extend(
            [
                f"    Layer {sanitize_identifier(layer.name, 'Layer')} {{",
                f"        Type: {layer.layer_type}",
                f"        Stride {{ X: {layer.stride_x}, Y: {layer.stride_y} }}",
                f"        Dimensions {{ K: {layer.k}, C: {layer.c}, R: {layer.r}, S: {layer.s}, Y: {layer.y}, X: {layer.x} }}",
                "        Dataflow {",
                f"            // Parsed from {layer.source_op} op {layer.source_index}.",
                f"            SpatialMap({k_tile_hint},{k_tile_hint}) K;",
                "            TemporalMap(1,1) C;",
            ]
        )

        if layer.r > 1 and layer.s > 1:
            lines.extend(
                [
                    f"            TemporalMap({layer.r},{layer.r}) R;",
                    f"            TemporalMap({layer.s},{layer.s}) S;",
                    f"            TemporalMap({layer.r},1) Y;",
                    f"            TemporalMap({layer.s},1) X;",
                ]
            )
        else:
            lines.extend(
                [
                    "            TemporalMap(1,1) Y;",
                    "            TemporalMap(1,1) X;",
                    f"            TemporalMap({layer.r},{layer.r}) R;",
                    f"            TemporalMap({layer.s},{layer.s}) S;",
                ]
            )

        lines.extend(["        }", "    }", ""])
    lines.append("}")
    return "\n".join(lines)


class ONNXGraphParser:
    """Extract convolution-like layers from ONNX or TFLite graphs.

    The optimizer only needs layer dimensions and strides, so this parser keeps
    the graph walk deliberately small: it reads tensor metadata, finds Conv2D
    and depthwise Conv2D operators, and emits LayerSpec-compatible records.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        network_name: str | None = None,
        include_depthwise: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.network_name = network_name or sanitize_identifier(self.model_path.stem)
        self.include_depthwise = include_depthwise

    def parse_tflite(self) -> tuple[str, list[GraphLayerSpec]]:
        import os
        import warnings

        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import tensorflow as tf

            interpreter = tf.lite.Interpreter(model_path=str(self.model_path))

        ops = interpreter._get_ops_details()
        tensors = {int(tensor["index"]): tensor for tensor in interpreter.get_tensor_details()}

        layers: list[GraphLayerSpec] = []
        conv_index = 0
        for op in ops:
            op_name = str(op.get("op_name", ""))
            if op_name not in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
                continue
            if op_name == "DEPTHWISE_CONV_2D" and not self.include_depthwise:
                continue

            inputs = [int(idx) for idx in op.get("inputs", [])]
            outputs = [int(idx) for idx in op.get("outputs", [])]
            if len(inputs) < 2 or not outputs:
                continue

            input_tensor = tensors.get(inputs[0])
            weight_tensor = tensors.get(inputs[1])
            output_tensor = tensors.get(outputs[0])
            if input_tensor is None or weight_tensor is None or output_tensor is None:
                continue

            input_shape = _shape_list(input_tensor)
            weight_shape = _shape_list(weight_tensor)
            output_shape = _shape_list(output_tensor)
            if len(input_shape) != 4 or len(weight_shape) != 4 or len(output_shape) != 4:
                continue

            if op_name == "CONV_2D":
                k, r, s, c = weight_shape
                layer_type = "Conv"
                layer_name = "Conv2D" if conv_index == 0 else f"Conv2D_{conv_index}"
            else:
                _, r, s, depthwise_k = weight_shape
                k = output_shape[3] if output_shape[3] > 0 else depthwise_k
                c = 1
                layer_type = "DepthwiseConv"
                layer_name = f"DepthwiseConv2D_{conv_index}"

            input_h, input_w = input_shape[1], input_shape[2]
            output_h, output_w = output_shape[1], output_shape[2]
            stride_y = _derive_stride(input_h, output_h, r)
            stride_x = _derive_stride(input_w, output_w, s)

            layers.append(
                GraphLayerSpec(
                    name=layer_name,
                    layer_type=layer_type,
                    k=max(1, int(k)),
                    c=max(1, int(c)),
                    r=max(1, int(r)),
                    s=max(1, int(s)),
                    y=max(1, int(output_h) * stride_y),
                    x=max(1, int(output_w) * stride_x),
                    stride_x=stride_x,
                    stride_y=stride_y,
                    source_op=op_name,
                    source_index=int(op.get("index", conv_index)),
                    input_name=str(input_tensor.get("name", inputs[0])),
                    output_name=str(output_tensor.get("name", outputs[0])),
                    weight_name=str(weight_tensor.get("name", inputs[1])),
                )
            )
            conv_index += 1

        if not layers:
            raise ValueError(f"No convolution layers found in {self.model_path}")
        return self.network_name, layers

    def parse_onnx(self) -> tuple[str, list[GraphLayerSpec]]:
        import onnx
        from onnx import helper, shape_inference

        model = onnx.load(str(self.model_path))
        try:
            graph = shape_inference.infer_shapes(model).graph
        except Exception:
            graph = model.graph

        shape_map = _onnx_shape_map(graph)
        initializer_shapes = {init.name: [int(dim) for dim in init.dims] for init in graph.initializer}

        layers: list[GraphLayerSpec] = []
        conv_index = 0
        for node_index, node in enumerate(graph.node):
            if node.op_type != "Conv" or len(node.input) < 2:
                continue

            weight_shape = initializer_shapes.get(node.input[1]) or shape_map.get(node.input[1])
            input_shape = shape_map.get(node.input[0])
            output_shape = shape_map.get(node.output[0]) if node.output else None
            if not weight_shape or len(weight_shape) != 4:
                continue

            attributes = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
            strides = attributes.get("strides", [1, 1])
            stride_y, stride_x = [int(v) for v in strides[:2]]
            groups = int(attributes.get("group", 1))

            k, c_per_group, r, s = [int(v) for v in weight_shape]
            c = max(1, c_per_group * groups)
            layer_type = "Conv"
            if groups > 1 and c_per_group == 1:
                if not self.include_depthwise:
                    continue
                c = 1
                layer_type = "DepthwiseConv"

            input_h, input_w = _onnx_spatial_dims(input_shape, output_shape, c_per_group * groups)
            output_h, output_w = _onnx_output_spatial(output_shape, input_h, input_w, stride_y, stride_x)

            op_name = node.name or f"Conv_{conv_index}"
            layer_name = "Conv2D" if conv_index == 0 else f"Conv2D_{conv_index}"
            if layer_type == "DepthwiseConv":
                layer_name = f"DepthwiseConv2D_{conv_index}"

            layers.append(
                GraphLayerSpec(
                    name=layer_name,
                    layer_type=layer_type,
                    k=max(1, k),
                    c=max(1, c),
                    r=max(1, r),
                    s=max(1, s),
                    y=max(1, int(output_h) * stride_y),
                    x=max(1, int(output_w) * stride_x),
                    stride_x=max(1, stride_x),
                    stride_y=max(1, stride_y),
                    source_op=f"ONNX:{op_name}",
                    source_index=node_index,
                    input_name=str(node.input[0]),
                    output_name=str(node.output[0]) if node.output else "",
                    weight_name=str(node.input[1]),
                )
            )
            conv_index += 1

        if not layers:
            raise ValueError(f"No Conv nodes found in {self.model_path}")
        return self.network_name, layers


def _shape_list(tensor_detail: dict) -> list[int]:
    shape = tensor_detail.get("shape_signature")
    if shape is None or not len(shape):
        shape = tensor_detail.get("shape")
    return [int(dim) if int(dim) > 0 else 1 for dim in shape]


def _derive_stride(input_len: int, output_len: int, kernel_len: int) -> int:
    input_len = max(1, int(input_len))
    output_len = max(1, int(output_len))
    kernel_len = max(1, int(kernel_len))

    for stride in range(1, 9):
        if input_len >= kernel_len and ((input_len - kernel_len) // stride + 1) == output_len:
            return stride
    for stride in range(1, 9):
        if math.ceil(input_len / stride) == output_len:
            return stride
    return max(1, int(round(input_len / output_len)))


def _onnx_shape_map(graph) -> dict[str, list[int]]:
    shape_map: dict[str, list[int]] = {}
    for value in list(graph.input) + list(graph.value_info) + list(graph.output):
        dims: list[int] = []
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value") and dim.dim_value > 0:
                dims.append(int(dim.dim_value))
            else:
                dims.append(1)
        shape_map[value.name] = dims
    return shape_map


def _onnx_spatial_dims(
    input_shape: list[int] | None,
    output_shape: list[int] | None,
    expected_c: int,
) -> tuple[int, int]:
    if input_shape and len(input_shape) == 4:
        if input_shape[1] == expected_c or input_shape[1] != 1:
            return max(1, input_shape[2]), max(1, input_shape[3])
        return max(1, input_shape[1]), max(1, input_shape[2])
    if output_shape and len(output_shape) == 4:
        return max(1, output_shape[2]), max(1, output_shape[3])
    return 1, 1


def _onnx_output_spatial(
    output_shape: list[int] | None,
    input_h: int,
    input_w: int,
    stride_y: int,
    stride_x: int,
) -> tuple[int, int]:
    if output_shape and len(output_shape) == 4:
        return max(1, output_shape[2]), max(1, output_shape[3])
    return max(1, math.ceil(input_h / max(1, stride_y))), max(1, math.ceil(input_w / max(1, stride_x)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse ONNX/TFLite convolution graphs into MAESTRO layer specs.")
    parser.add_argument("model", help="Input .onnx or .tflite model.")
    parser.add_argument("--output", default="parsed_model.m", help="Output MAESTRO-style layer spec.")
    parser.add_argument("--network-name", default=None)
    parser.add_argument("--skip-depthwise", action="store_true", help="Skip depthwise convolution layers.")
    args = parser.parse_args()

    network_name, layers = parse_model(
        args.model,
        network_name=args.network_name,
        include_depthwise=not args.skip_depthwise,
    )
    Path(args.output).write_text(render_maestro_spec(network_name, layers), encoding="utf-8")
    print(f"{network_name}: parsed {len(layers)} convolution layers -> {args.output}")


if __name__ == "__main__":
    main()
