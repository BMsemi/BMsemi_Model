"""
User script with:
1. RGB image loading (224×224) from test_file.py approach
2. Binary weight retraining and loading from binary_weight_retraining.py
3. Crossbar programming with auto-mode detection
"""

import numpy as np
import torch
import hashlib
import json
import cv2
import time
import argparse
from PIL import Image
from pathlib import Path
from dataclasses import replace
import json

from neuromorphic_x2 import NeuromorphicX2
from Scheduler import MaestroScheduler, MaestroClusterScheduler
from Mapper import MaestroMapper, MaestroClusterMapper
from energy import EnergyCalculator
from Multiple_PE_core import MultiPENeuromorphicCore
from L2_buffer import L2_Buffer
from extract_yolo26n_int8_layer0 import run_litert_layer0_step

# # Import binary weight retraining functions
# from binary_weight_retraining import (
#     mobilenet_v2,
#     convert_to_binary_model,
#     initialize_binary_weights,
#     retrain_binary_model,
#     create_test_dataloader,
#     binarize_weights
# )

import sys
import io
import contextlib
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# =====================================================
# Image Loading (replaces MNIST loading)
# =====================================================
def load_rgb_image_plain_resize(img_path: str, size: tuple = (640, 640), threshold: int = 128, save_dir: str = None):
    """
    Legacy helper: plain-resize RGB image and return uint8 [0,255] values per channel.
    
    Args:
        img_path: Path to input image
        size: Target size (width, height)
        threshold: Unused (kept for API compatibility)
        save_dir: Directory to save channel text files (optional)
    
    Returns:
        rgb: (H, W, 3) uint8 array
        rgb_channels: dict with 'R', 'G', 'B' uint8 matrices (values 0-255)
    """
    print(f"[IMAGE] Loading from: {img_path}")
    
    # Load and resize
    im_original = Image.open(img_path).convert("RGB") 
    print(f"[IMAGE] Original size: {im_original.size}")
    
    im = im_original.resize(size)
    print(f"[IMAGE] Resized to: {im.size}")
    
    # Convert to numpy and keep full 8-bit pixel values (0-255)
    rgb = np.asarray(im, dtype=np.uint8)  # (H, W, 3)
    print(f"[IMAGE] Array shape: {rgb.shape}")
    
    # Separate channels — raw uint8 values, no binarization
    rgb_channels = {
        'R': rgb[:, :, 0].astype(np.int16),
        'G': rgb[:, :, 1].astype(np.int16),
        'B': rgb[:, :, 2].astype(np.int16)
    }
    
    print(f"[IMAGE] Loaded uint8 pixels (no binarization)")
    print(f"[IMAGE] R channel: min={rgb_channels['R'].min()}, max={rgb_channels['R'].max()}, mean={rgb_channels['R'].mean():.1f}")
    print(f"[IMAGE] G channel: min={rgb_channels['G'].min()}, max={rgb_channels['G'].max()}, mean={rgb_channels['G'].mean():.1f}")
    print(f"[IMAGE] B channel: min={rgb_channels['B'].min()}, max={rgb_channels['B'].max()}, mean={rgb_channels['B'].mean():.1f}")
    
    # Save each channel as text file for PE mapping
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        for channel_name, channel_data in rgb_channels.items():
            channel_file = save_path / f"{channel_name}_channel_{size[0]}x{size[1]}.txt"
            np.savetxt(channel_file, channel_data, fmt="%d", delimiter=" ")
            print(f"[SAVE] {channel_name} channel saved: {channel_file}")
    
    print()
    return rgb, rgb_channels


def letterbox_ultralytics(
    image: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    *,
    auto: bool = False,
    scale_fill: bool = False,
    scaleup: bool = True,
    center: bool = True,
    stride: int = 32,
    padding_value: int = 114,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[np.ndarray, dict]:
    """
    Resize and pad an image using the same geometry as Ultralytics LetterBox.

    Args:
        image: Input image in HWC order. Channel order is preserved.
        new_shape: Target shape as (height, width).
        auto: Use minimum rectangle padding rounded by stride.
        scale_fill: Stretch image to new_shape without preserving aspect ratio.
        scaleup: Allow enlarging the image.
        center: Center image in the padded canvas.
        stride: Model stride used when auto=True.
        padding_value: Constant border value. Ultralytics default is 114.
        interpolation: Resize interpolation.

    Returns:
        letterboxed_image, metadata with ratio/padding information.
    """
    if image is None:
        raise ValueError("letterbox_ultralytics received an empty image")
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    shape = image.shape[:2]  # current shape: height, width
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = (r, r)
    new_unpad = (round(shape[1] * r), round(shape[0] * r))  # width, height
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    if center:
        dw /= 2
        dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=interpolation)
    if image.ndim == 2:
        image = image[..., None]

    top = round(dh - 0.1) if center else 0
    bottom = round(dh + 0.1)
    left = round(dw - 0.1) if center else 0
    right = round(dw + 0.1)

    h, w, c = image.shape
    if c == 3:
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(padding_value,) * 3,
        )
    else:
        padded = np.full(
            (h + top + bottom, w + left + right, c),
            fill_value=padding_value,
            dtype=image.dtype,
        )
        padded[top : top + h, left : left + w] = image
        image = padded

    meta = {
        "original_shape": shape,
        "new_shape": new_shape,
        "resized_unpadded_shape": (new_unpad[1], new_unpad[0]),
        "ratio": ratio,
        "pad": (left, top),
        "border": (top, bottom, left, right),
        "padding_value": padding_value,
    }
    return image, meta


def _write_channel_file(path: Path, channel_name: str, channel_data: np.ndarray, meta: dict) -> None:
    height, width = channel_data.shape
    top, bottom, left, right = meta["border"]
    ratio_x, ratio_y = meta["ratio"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{channel_name} channel, Ultralytics letterbox crossbar activation: {width}x{height}\n")
        f.write(f"Format: {height} rows x {width} columns, zero-point-removed uint8 values in [0,255]\n")
        f.write(
            "LetterBox: "
            f"ratio=({ratio_x:.8f},{ratio_y:.8f}), "
            f"pad_left/right=({left},{right}), pad_top/bottom=({top},{bottom}), "
            f"pad_value={meta['padding_value']}\n\n"
        )
        np.savetxt(f, channel_data, fmt="%d", delimiter=" ")


def load_rgb_image(
    img_path: str,
    size: tuple = (640, 640),
    threshold: int = 128,
    save_dir: str = None,
    return_preprocessed: bool = False,
    input_scale: float = 1.0 / 255.0,
    input_zero_point: int = -128,
):
    """
    Load an image and apply the Ultralytics inference preprocessing order.

    Args:
        img_path: Path to input image.
        size: Target size as (width, height).
        threshold: Unused; kept for API compatibility.
        save_dir: Optional directory for letterboxed R/G/B channel text files.
        return_preprocessed: Also return quantized INT8 input and metadata.
        input_scale: Quantized model input scale. YOLO26n TFLite uses ~1/255.
        input_zero_point: Quantized model input zero point. YOLO26n TFLite uses -128.

    Returns:
        rgb: (H, W, 3) uint8 RGB after letterbox.
        rgb_channels: dict with R/G/B uint8 matrices in [0,255].

        If return_preprocessed=True, also returns:
        model_input_i8: (1, H, W, 3) signed INT8 input for the quantized model.
        preprocess_meta: ratio/pad metadata for postprocess box scaling.
    """
    _ = threshold
    print(f"[IMAGE] Loading from: {img_path}")

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")

    target_hw = (int(size[1]), int(size[0]))
    print(f"[IMAGE] Original shape (H,W,C BGR): {bgr.shape}")

    bgr_lb, preprocess_meta = letterbox_ultralytics(bgr, new_shape=target_hw)
    print(f"[IMAGE] Letterboxed shape: {bgr_lb.shape}")
    print(
        "[IMAGE] LetterBox "
        f"ratio={preprocess_meta['ratio']}, "
        f"pad(left,top)={preprocess_meta['pad']}, "
        f"border(top,bottom,left,right)={preprocess_meta['border']}"
    )

    # Quantized model path after LetterBox: BGR -> RGB, then quantize.
    # For YOLO26n full-integer TFLite, scale ~= 1/255 and zero_point = -128,
    # so the int8 model tensor is q = pixel - 128.
    rgb = np.ascontiguousarray(bgr_lb[:, :, ::-1])
    model_input_i16 = np.rint((rgb.astype(np.float32) / 255.0) / float(input_scale) + int(input_zero_point))
    model_input_i8 = np.clip(model_input_i16, -128, 127).astype(np.int8)[None, ...]
    crossbar_activation_u8 = np.clip(
        model_input_i8.astype(np.int16) - int(input_zero_point),
        0,
        255,
    ).astype(np.uint8)[0]
    crossbar_bchw_u8 = np.ascontiguousarray(crossbar_activation_u8.transpose((2, 0, 1)))[None, ...]
    print(
        f"[IMAGE] Quantized model input: {model_input_i8.shape}, "
        f"int8 range=[{model_input_i8.min()}, {model_input_i8.max()}], "
        f"scale={input_scale:.9f}, zero_point={input_zero_point}"
    )
    print(
        f"[IMAGE] Crossbar activation input: {crossbar_bchw_u8.shape}, "
        f"uint8 range=[{crossbar_bchw_u8.min()}, {crossbar_bchw_u8.max()}]"
    )

    # The custom PE flow consumes zero-point-removed 8-bit activation magnitudes:
    # a_u8 = q_int8 - zero_point. For zero_point=-128, this is the original pixel.
    rgb_channels = {
        "R": crossbar_activation_u8[:, :, 0].astype(np.uint8),
        "G": crossbar_activation_u8[:, :, 1].astype(np.uint8),
        "B": crossbar_activation_u8[:, :, 2].astype(np.uint8),
    }

    print("[IMAGE] Loaded letterboxed uint8 pixels (no binarization)")
    print(f"[IMAGE] R channel: min={rgb_channels['R'].min()}, max={rgb_channels['R'].max()}, mean={rgb_channels['R'].mean():.1f}")
    print(f"[IMAGE] G channel: min={rgb_channels['G'].min()}, max={rgb_channels['G'].max()}, mean={rgb_channels['G'].mean():.1f}")
    print(f"[IMAGE] B channel: min={rgb_channels['B'].min()}, max={rgb_channels['B'].max()}, mean={rgb_channels['B'].mean():.1f}")

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        image_stem = Path(img_path).stem

        for channel_name, channel_data in rgb_channels.items():
            channel_file = save_path / f"{image_stem}_{channel_name}_channel_{size[0]}x{size[1]}.txt"
            _write_channel_file(channel_file, channel_name, channel_data, preprocess_meta)
            print(f"[SAVE] {channel_name} channel saved: {channel_file}")

    print()
    if return_preprocessed:
        return rgb, rgb_channels, model_input_i8, preprocess_meta
    return rgb, rgb_channels


def load_rgb_channels_from_files(r_path: str, g_path: str, b_path: str) -> dict:
    """
    Load pre-saved R, G, B channel text files (640×640, uint8).
    File format: 3-line header then 640 rows of space-separated integers.
    """
    def _read_channel(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Skip header lines (non-numeric lines at top)
        data_lines = [ln.strip() for ln in lines if ln.strip() and ln.strip()[0].isdigit()]
        rows = [list(map(int, ln.split())) for ln in data_lines]
        arr = np.array(rows, dtype=np.int16)
        print(f"[CHANNEL] Loaded {path}: shape={arr.shape}, min={arr.min()}, max={arr.max()}")
        return arr

    return {
        'R': _read_channel(r_path),
        'G': _read_channel(g_path),
        'B': _read_channel(b_path),
    }


def load_layer0_weights(weights_path: str) -> np.ndarray:
    """Load first-layer YOLO weights from JSON text and return (K,C,R,S)."""
    with open(weights_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    weights = np.asarray(raw, dtype=np.float32)
    if weights.ndim != 4:
        raise ValueError(f"Expected 4D weights (K,C,R,S), got shape {weights.shape}")

    print(f"[WEIGHTS] Loaded layer0 weights: {weights.shape} from {weights_path}")
    print(f"[WEIGHTS] Range: [{weights.min():.2f}, {weights.max():.2f}]")
    return weights


def load_layer0_padding(padding_path: str) -> tuple[int, int, int, int]:
    """
    Load padding from Layer0_padding.txt style JSON:
      [[n0,n1],[top,bottom],[left,right],[c0,c1]]
    """
    with open(padding_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    arr = np.asarray(raw, dtype=int)
    if arr.shape != (4, 2):
        raise ValueError(f"Expected padding shape (4,2), got {arr.shape}")

    pad_top, pad_bottom = int(arr[1, 0]), int(arr[1, 1])
    pad_left, pad_right = int(arr[2, 0]), int(arr[2, 1])
    print(
        f"[PADDING] Loaded from {padding_path}: "
        f"top/bottom=({pad_top},{pad_bottom}), left/right=({pad_left},{pad_right})"
    )
    return pad_top, pad_bottom, pad_left, pad_right


def apply_padding_to_rgb_channels(
    rgb_channels: dict,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
) -> dict:
    """Apply spatial padding to each RGB channel independently."""
    padded = {}
    for key in ("R", "G", "B"):
        ch = np.asarray(rgb_channels[key], dtype=np.uint8)
        padded[key] = np.pad(
            ch,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        ).astype(np.uint8)
    return padded


def write_layer_output_dump(path: str, l2_buffer: L2_Buffer, layer_id: int, output_shape: tuple[int, int, int]):
    """Write full layer output tensor channel-by-channel to a text file."""
    out_y, out_x, out_c = output_shape
    out_path = Path(path)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"LayerID: {layer_id}\n")
        f.write(f"OutputShape: ({out_y}, {out_x}, {out_c})\n")
        f.write("=" * 80 + "\n")

        for c in range(out_c):
            ch = l2_buffer.read_output_channel(layer_id, c)
            if ch is None:
                ch = np.zeros((out_y, out_x), dtype=np.float32)
            ch = np.asarray(ch, dtype=np.float32)

            if ch.shape != (out_y, out_x):
                tmp = np.zeros((out_y, out_x), dtype=np.float32)
                h = min(out_y, ch.shape[0])
                w = min(out_x, ch.shape[1])
                tmp[:h, :w] = ch[:h, :w]
                ch = tmp

            f.write(f"\nCHANNEL {c}\n")
            for y in range(out_y):
                row = " ".join(f"{v:.6f}" for v in ch[y])
                f.write(row + "\n")

    print(f"[SAVE] Output dump written: {out_path}")


# =====================================================
# Binary Weight Retraining & Loading
# =====================================================
# def retrain_and_export_weights(epochs: int = 5, output_dir: str = "binary_retrained"):
#     """
#     Retrain MobileNetV2 with binary weights and export to .npy files.
    
#     Args:
#         epochs: Number of retraining epochs
#         output_dir: Directory to save weights
    
#     Returns:
#         model: Retrained binary model
#         output_dir: Path to saved weights
#     """
#     print("\n" + "="*80)
#     print("BINARY WEIGHT RETRAINING")
#     print("="*80 + "\n")
    
#     output_path = Path(output_dir)
#     output_path.mkdir(exist_ok=True)
    
#     # # Load pretrained MobileNetV2
#     # print("[STEP 1] Loading MobileNetV2...")
#     # model = mobilenet_v2(pretrained=True)
    
#     # # Convert to binary model
#     # print("[STEP 2] Converting to binary layers...")
#     # model = convert_to_binary_model(model)
#     # initialize_binary_weights(model)
    
#     # # Create synthetic data for quick training
#     # print("[STEP 3] Creating training data...")
#     # train_loader, val_loader = create_test_dataloader(batch_size=32)
    
#     # # Retrain
#     # print("[STEP 4] Retraining with binary constraints...")
#     # model, history = retrain_binary_model(
#     #     model, train_loader, val_loader,
#     #     epochs=epochs,
#     #     lr=0.001,
#     #     device='cuda' if torch.cuda.is_available() else 'cpu'
#     # )
    
#     # Export weights
#     print("[STEP 5] Exporting binary weights...")
#     from binary_weight_retraining import export_binary_weights
#     export_binary_weights(model, output_path)
    
#     # Save model checkpoint
#     model_path = output_path / f'binary_model_epoch{epochs}.pth'
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'epochs': epochs,
#         'best_val_top1': max(history['val_acc_top1'])
#     }, model_path)
#     print(f"[SAVE] Model checkpoint: {model_path}\n")
    
#     return model, str(output_path)


# def load_binary_weights_from_checkpoint(checkpoint_path: str, layer_name: str = "features_0_0", mdims=None):
#     """
#     Load specific layer's binary weights from checkpoint.
#     Special case: Returns None for "IDENTITY" layers (no weights needed).
    
#     Args:
#         checkpoint_path: Path to .pth checkpoint
#         layer_name: Layer to extract (e.g., "features_0_0" for first conv, "IDENTITY" for skip)
    
#     Returns:
#         binary_weights: (out_channels, in_channels, H, W) or None for IDENTITY layers
#     """
#     # Handle identity/skip layers (no actual conv weights)
#     if layer_name == "IDENTITY":
#         print(f"[LOAD] Identity layer - no weights needed")
#         return None
    
#     print(f"[LOAD] Loading binary weights from: {checkpoint_path}")
#     print(f"[LOAD] Target layer: {layer_name}")
    
#     # Load checkpoint
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
#     state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    
#     # Candidate keys across different module wrapping layouts
#     possible_keys = [
#         f"{layer_name}.weight",
#         f"{layer_name}.0.weight",
#         f"{layer_name}.0.0.weight",
#     ]

#     conv_candidates = []
#     for candidate in possible_keys:
#         if candidate in state_dict:
#             tensor = state_dict[candidate]
#             if len(tensor.shape) == 4:
#                 conv_candidates.append(candidate)

#     # Fallback: prefix scan, only Conv2d weights (4D)
#     if not conv_candidates:
#         for key, tensor in state_dict.items():
#             if key.startswith(layer_name) and key.endswith('.weight') and len(tensor.shape) == 4:
#                 conv_candidates.append(key)

#     if not conv_candidates:
#         linear_candidates = []
#         if mdims is not None and int(mdims.R) == 1 and int(mdims.S) == 1:
#             possible_linear_keys = [
#                 f"{layer_name}.weight",
#                 f"{layer_name}.0.weight",
#                 f"{layer_name}.0.0.weight",
#             ]
#             for candidate in possible_linear_keys:
#                 if candidate in state_dict:
#                     tensor = state_dict[candidate]
#                     if len(tensor.shape) == 2:
#                         linear_candidates.append(candidate)

#             if not linear_candidates:
#                 for key, tensor in state_dict.items():
#                     if key.startswith(layer_name) and key.endswith('.weight') and len(tensor.shape) == 2:
#                         linear_candidates.append(key)

#         if linear_candidates:
#             weight_key = linear_candidates[0]
#             float_weights = state_dict[weight_key]
#             binary_weights = binarize_weights(float_weights).numpy().reshape(
#                 float_weights.shape[0], float_weights.shape[1], 1, 1
#             )

#             print(f"[LOAD] Loaded linear layer as 1x1 conv: {weight_key}")
#             print(f"[LOAD] Weight shape: {binary_weights.shape}")
#             print(f"[LOAD] Weight range: [{binary_weights.min()}, {binary_weights.max()}]")
#             print(f"[LOAD] Positive weights: {np.sum(binary_weights == 1)}")
#             print(f"[LOAD] Negative weights: {np.sum(binary_weights == -1)}\n")

#             return binary_weights

#         print(f"[ERROR] No Conv2d weights found for '{layer_name}'")
#         print(f"[INFO] Tried: {possible_keys}")
#         print(f"\n[INFO] Searching for Conv2d weights in vicinity...")

#         prefix = '.'.join(layer_name.split('.')[:4])
#         print(f"[INFO] Keys starting with '{prefix}':")
#         for key in sorted(state_dict.keys()):
#             if key.startswith(prefix):
#                 shape = state_dict[key].shape
#                 is_conv = len(shape) == 4
#                 print(f"  - {key}: shape={shape} {'← CONV2D' if is_conv else ''}")

#         print(f"\n[INFO] All Conv2d weights in this bottleneck:")
#         layer_parts = layer_name.split('.')
#         if len(layer_parts) >= 3:
#             search_prefix = '.'.join(layer_parts[:3])
#             for key in sorted(state_dict.keys()):
#                 if search_prefix in key and 'weight' in key:
#                     shape = state_dict[key].shape
#                     if len(shape) == 4:
#                         print(f"  - {key}: shape={shape}")

#         raise ValueError(f"No Conv2d weight found for layer '{layer_name}'")

#     # If mapping dimensions are available, pick the best shape match first
#     weight_key = conv_candidates[0]
#     if mdims is not None:
#         expect_std = (int(mdims.K), int(mdims.C), int(mdims.R), int(mdims.S))
#         expect_dw = (int(mdims.C), 1, int(mdims.R), int(mdims.S))
#         for candidate in conv_candidates:
#             shape = tuple(int(v) for v in state_dict[candidate].shape)
#             if shape == expect_std or shape == expect_dw:
#                 weight_key = candidate
#                 break
    
#     # Extract and binarize
#     float_weights = state_dict[weight_key]
#     binary_weights = binarize_weights(float_weights).numpy()
    
#     print(f"[LOAD] Loaded layer: {weight_key}")
#     print(f"[LOAD] Weight shape: {binary_weights.shape}")
#     print(f"[LOAD] Weight range: [{binary_weights.min()}, {binary_weights.max()}]")
#     print(f"[LOAD] Positive weights: {np.sum(binary_weights == 1)}")
#     print(f"[LOAD] Negative weights: {np.sum(binary_weights == -1)}\n")
    
#     return binary_weights


def prepare_conv_weights_for_crossbar(
    conv_weights: np.ndarray, 
    target_size: int = 64,
    mode: str = "auto"
) -> np.ndarray:
    """
    Convert Conv2D weights to crossbar-compatible format.
    
    Mode Behavior:
    --------------
    "auto" (default):
        - If weights contain negatives → prepare for SIGNED mode (64, 32)
        - If weights are all ≥ 0 → prepare for UNSIGNED mode (64, 64)
    
    "signed":
        - Force SIGNED mode preparation (64, 32)
        - If input has negatives: use as-is
        - If input is unsigned: convert 0→-1, 1→+1
    
    "unsigned":
        - Force UNSIGNED mode preparation (64, 64)
        - If input has negatives: map -1→0, +1→1
        - If input is unsigned: use as-is
    
    Crossbar Internal Handling (NOT done here):
    -------------------------------------------
    SIGNED mode (64,32) input:
        Crossbar converts to (64,64) differential with ALL POSITIVE values:
            - Even columns: max(w, 0)  [positive parts]
            - Odd columns:  max(-w, 0) [negative parts as positive]
        Compute: output[i] = col[2i] - col[2i+1]
    
    UNSIGNED mode (64,64) input:
        Crossbar stores directly (64,64)
        Compute: standard MAC
    
    Args:
        conv_weights: (out_channels, in_channels, H, W) weights
        target_size: Target crossbar size (64)
        mode: "auto", "signed", or "unsigned"
    
    Returns:
        For SIGNED:   (64, 32) signed weights {-1, +1}
        For UNSIGNED: (64, 64) unsigned weights {0, 1}
    """
    print(f"[PREPARE] Converting conv weights to crossbar format...")
    print(f"[PREPARE] Input shape: {conv_weights.shape}")
    print(f"[PREPARE] Mode: {mode}")
    
    # Flatten spatial dimensions: (out_ch, in_ch, H, W) -> (out_ch, in_ch*H*W)
    out_ch, in_ch, H, W = conv_weights.shape
    flattened = conv_weights.reshape(out_ch, in_ch * H * W)
    
    # Transpose to crossbar convention: (in_features, out_features)
    transposed = flattened.T  # (in_ch*H*W, out_ch)
    in_features, out_features = transposed.shape
    
    print(f"[PREPARE] Transposed to: {transposed.shape}")
    
    # Detect weight type
    has_negative = np.any(transposed < 0)
    is_binary = np.all(np.isin(transposed, [-1, 0, 1]))
    
    # Determine actual mode
    if mode == "auto":
        actual_mode = "signed" if has_negative else "unsigned"
        print(f"[PREPARE] Auto-detected: {actual_mode} ({'binary' if is_binary else 'continuous'})") 
    else:
        actual_mode = mode
    
    # =========================================================================
    # SIGNED MODE: Prepare (64, 32) with values in {-1, +1}
    # =========================================================================
    if actual_mode == "signed":
        print(f"[PREPARE] Target: SIGNED mode (64, 32)")
        
        # Convert unsigned to signed if needed
        if not has_negative:
            print(f"[PREPARE] Converting unsigned {{0, 1}} → signed {{-1, +1}}")
            transposed = 2 * transposed - 1  # 0→-1, 1→+1
        
        # Limit output channels to 32 (differential pair limit)
        if out_features > target_size // 2:
            print(f"[WARNING] Truncating output channels: {out_features} → {target_size // 2}")
            transposed = transposed[:, :target_size // 2]
            out_features = target_size // 2
        
        # Pad to (64, 32)
        crossbar_weights = np.zeros((target_size, target_size // 2), dtype=float)
        crossbar_weights[:min(in_features, target_size), :out_features] = \
            transposed[:min(in_features, target_size), :]
        
        neg_count = np.sum(crossbar_weights == -1)
        pos_count = np.sum(crossbar_weights == 1)
        zero_count = np.sum(crossbar_weights == 0)
        
        print(f"[PREPARE] SIGNED weights (64, 32):")
        print(f"           -1: {neg_count:5d}   +1: {pos_count:5d}   0: {zero_count:5d}")
        print(f"[PREPARE] Crossbar will convert to (64, 64) differential pairs internally")
        print(f"           Even cols [0,2,...,62]: max(w, 0)")
        print(f"           Odd cols  [1,3,...,63]: max(-w, 0)")
    
    # =========================================================================
    # UNSIGNED MODE: Prepare (64, 64) with values in {0, 1}
    # =========================================================================
    else:  # "unsigned"
        print(f"[PREPARE] Target: UNSIGNED mode (64, 64)")
        
        # Convert signed to unsigned if needed
        if has_negative:
            print(f"[PREPARE] Converting signed {{-1, +1}} → unsigned {{0, 1}}")
            transposed = (transposed + 1) / 2  # -1→0, +1→1
        
        # Limit to 64 output channels
        if out_features > target_size:
            print(f"[WARNING] Truncating output channels: {out_features} → {target_size}")
            transposed = transposed[:, :target_size]
            out_features = target_size
        
        # Pad to (64, 64)
        crossbar_weights = np.zeros((target_size, target_size), dtype=float)
        crossbar_weights[:min(in_features, target_size), :out_features] = \
            transposed[:min(in_features, target_size), :]
        
        zero_count = np.sum(crossbar_weights == 0)
        one_count = np.sum(crossbar_weights == 1)
        
        print(f"[PREPARE] UNSIGNED weights (64, 64):")
        print(f"           0: {zero_count:5d}   1: {one_count:5d}")
        print(f"[PREPARE] Crossbar will store directly (no differential)")
    
    # Validate output shape
    if crossbar_weights.shape not in [(target_size, target_size // 2), (target_size, target_size)]:
        raise ValueError(
            f"Invalid crossbar shape {crossbar_weights.shape}. "
            f"Expected ({target_size}, {target_size // 2}) for SIGNED or ({target_size}, {target_size}) for UNSIGNED"
        )
    
    print()
    return crossbar_weights


# =====================================================
# Utility Functions (from original user.py)
# =====================================================
def _extract_patch_same_2d(img2d: np.ndarray, oy: int, ox: int, R: int, S: int) -> np.ndarray:
    """Extract SAME-padded patch from 2D image."""
    pad_y = R // 2
    pad_x = S // 2
    padded = np.pad(img2d, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant", constant_values=0)
    yy = oy + pad_y
    xx = ox + pad_x
    patch = padded[yy - pad_y : yy - pad_y + R, xx - pad_x : xx - pad_x + S]
    return patch.reshape(-1).astype(float)


def _extract_patch_from_padded_2d(img2d: np.ndarray, y: int, x: int, R: int, S: int) -> np.ndarray:
    """Extract an R×S patch directly from a pre-padded image."""
    y0 = int(y)
    x0 = int(x)
    y1 = y0 + int(R)
    x1 = x0 + int(S)

    h, w = img2d.shape
    if 0 <= y0 and y1 <= h and 0 <= x0 and x1 <= w:
        return np.asarray(img2d[y0:y1, x0:x1], dtype=float).reshape(-1)

    # Conservative fallback for out-of-bounds access.
    patch = np.zeros((R, S), dtype=float)
    sy0 = max(0, y0)
    sx0 = max(0, x0)
    sy1 = min(h, y1)
    sx1 = min(w, x1)
    if sy1 > sy0 and sx1 > sx0:
        patch[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img2d[sy0:sy1, sx0:sx1]
    return patch.reshape(-1)


def _write_pe_outputs(path: str, pe_ids: list, outputs_getter):
    """Write PE outputs to file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"=== Used PE Outputs (count={len(pe_ids)}) ===\n")
        for pe in pe_ids:
            out = np.asarray(outputs_getter(pe), dtype=float).reshape(-1)
            line = " ".join(f"{v:.6f}" for v in out[:64])  # Updated to 64
            f.write(f"PE {pe}: {line}\n")


def _compute_mapping_stats(mdims):
    """Compute mapping-stable stats: output shape, MACs, and number of weights."""
    out_y = (int(mdims.Y) + int(mdims.stride_y) - 1) // int(mdims.stride_y)
    out_x = (int(mdims.X) + int(mdims.stride_x) - 1) // int(mdims.stride_x)

    is_depthwise_map = int(mdims.K) == 1
    out_c = int(mdims.C) if is_depthwise_map else int(mdims.K)

    if is_depthwise_map:
        num_weights = int(mdims.C) * int(mdims.R) * int(mdims.S)
        total_macs = out_y * out_x * int(mdims.C) * int(mdims.R) * int(mdims.S)
    else:
        num_weights = int(mdims.K) * int(mdims.C) * int(mdims.R) * int(mdims.S)
        total_macs = out_y * out_x * int(mdims.K) * int(mdims.C) * int(mdims.R) * int(mdims.S)

    return (out_y, out_x, out_c), int(total_macs), int(num_weights)


# def _make_deterministic_binary_weights(layer_name: str, mdims):
#     seed_source = f"{layer_name}|K{int(mdims.K)}|C{int(mdims.C)}|R{int(mdims.R)}|S{int(mdims.S)}"
#     seed = int.from_bytes(hashlib.md5(seed_source.encode("utf-8")).digest()[:4], "little")
#     rng = np.random.default_rng(seed)

#     map_k = int(mdims.K)
#     map_c = int(mdims.C)
#     map_r = int(mdims.R)
#     map_s = int(mdims.S)

#     is_depthwise_map = map_k == 1 and map_c > 1 and (map_r > 1 or map_s > 1)
#     if is_depthwise_map:
#         shape = (map_c, 1, map_r, map_s)
#     else:
#         shape = (max(1, map_k), map_c, map_r, map_s)

#     return rng.choice([-1.0, 1.0], size=shape).astype(np.float32)


# =====================================================
# Weight Provider for Depthwise and Standard Convolution
# =====================================================
def make_weight_provider(conv_weights, mdims):
    """
    Build LOGICAL weight matrix for scheduler (patch_len rows, NOT 64).
    
    CRITICAL: Uses ACTUAL weight dimensions from conv_weights, NOT mdims!
    This handles mismatches between mapping file and checkpoint.
    
    Args:
        conv_weights: (K, C, R, S) or (C, 1, R, S) for depthwise
        mdims: Layer dimensions from mapping (may have wrong R,S)
    
    Returns:
        (weight_provider_func, is_depthwise, actual_output_channels, actual_dims)
    """
    import numpy as np
    
    # Get ACTUAL dimensions from weight shape (NOT from mdims!)
    if conv_weights.ndim == 4:
        K_actual, C_actual, R_actual, S_actual = conv_weights.shape
    else:
        raise ValueError(f"Invalid conv_weights shape: {conv_weights.shape}")
    
    # Detect depthwise in both possible layouts:
    #  1) K=1, C=channels (custom)
    #  2) K=channels, C=1 (PyTorch depthwise)
    map_c = int(getattr(mdims, "C", C_actual))
    is_depthwise_layout1 = (K_actual == 1 and C_actual > 1 and (R_actual > 1 or S_actual > 1))
    is_depthwise_layout2 = (C_actual == 1 and K_actual == map_c and (R_actual > 1 or S_actual > 1))
    is_depthwise = is_depthwise_layout1 or is_depthwise_layout2

    # Logical dimensions used by scheduler/patch extraction
    if is_depthwise_layout2:
        logical_c = K_actual
        patch_len = logical_c * R_actual * S_actual
    else:
        logical_c = C_actual
        patch_len = C_actual * R_actual * S_actual
    
    if is_depthwise:
        print(f"[WEIGHT_PROVIDER] DEPTHWISE CONV")
        print(f"  Actual dimensions: K={K_actual}, C={C_actual}, R={R_actual}, S={S_actual}")
        print(f"  Mapping dimensions: K={mdims.K}, C={mdims.C}, R={mdims.R}, S={mdims.S}")
        print(f"  patch_len={patch_len} (C={logical_c} × R={R_actual} × S={S_actual})")
        print(f"  Logical weight matrix: ({patch_len}, {logical_c})")
        
        # For depthwise, actual output channels = C (not K=1)
        W_logical = np.zeros((patch_len, logical_c), dtype=np.float32)
        per_ch = R_actual * S_actual  # e.g., 9 for 3×3
        
        # Build diagonal structure
        for c in range(logical_c):
            if is_depthwise_layout2:
                w = conv_weights[c, 0]  # (R, S) from (C,1,R,S)
            else:
                w = conv_weights[c, 0] if K_actual == 1 else conv_weights[c]
            W_logical[c * per_ch:(c + 1) * per_ch, c] = w.flatten()
        
        actual_output_channels = logical_c
        print(f"  Output channels: {actual_output_channels} (depthwise preserves C)")
        
    else:
        print(f"[WEIGHT_PROVIDER] STANDARD CONV")
        print(f"  Actual dimensions: K={K_actual}, C={C_actual}, R={R_actual}, S={S_actual}")
        print(f"  Mapping dimensions: K={mdims.K}, C={mdims.C}, R={mdims.R}, S={mdims.S}")
        if (R_actual != mdims.R or S_actual != mdims.S):
            print(f"  WARNING: Dimension mismatch! Using ACTUAL from weights.")
        print(f"  patch_len={patch_len} (C={C_actual} × R={R_actual} × S={S_actual})")
        print(f"  Logical weight matrix: ({patch_len}, {K_actual})")
        
        W_logical = np.zeros((patch_len, K_actual), dtype=np.float32)
        
        # Flatten (C, R, S) for each output channel
        for k in range(K_actual):
            W_logical[:, k] = conv_weights[k].flatten()
        
        actual_output_channels = K_actual
        print(f"  Output channels: {actual_output_channels}")
    
    print(f"  Built W_logical: {W_logical.shape}")
    print(f"  Non-zero elements: {np.count_nonzero(W_logical)}")
    print(f"  Crossbar tiles needed: {int(np.ceil(patch_len / 64))}")
    
    # Return provider function
    def weight_provider(k0: int, kN: int) -> np.ndarray:
        """Returns (patch_len, kN-k0) weight slice with zero-padding for tail chunks."""
        req_cols = int(kN - k0)
        out_slice = np.zeros((W_logical.shape[0], req_cols), dtype=W_logical.dtype)
        real_end = min(int(kN), int(W_logical.shape[1]))
        if k0 < real_end:
            out_slice[:, :real_end - k0] = W_logical[:, k0:real_end]
        return out_slice
    
    # Return actual dimensions for scheduler override
    actual_dims = {
        'C': logical_c if is_depthwise else C_actual,
        'R': R_actual,
        'S': S_actual,
        'K': K_actual,
        'patch_len': patch_len
    }
    
    return weight_provider, is_depthwise, actual_output_channels, actual_dims


# =====================================================
# Multi-Layer Execution Engine
# =====================================================

# Complete layer sequence for MobileNetV2
# ALL_MOBILENET_LAYERS = [
#     "CONV1",
#     "Bottleneck1_1_1", "Bottleneck1_1_2", "Bottleneck1_1_3", "Bottleneck1_1_Residual",
#     "Bottleneck2_1_1", "Bottleneck2_1_2", "Bottleneck2_1_3",
#     "Bottleneck2_2_1", "Bottleneck2_2_2", "Bottleneck2_2_3",
#     "Bottleneck3_1_1", "Bottleneck3_1_2", "Bottleneck3_1_3",
#     "Bottleneck3_2_1", "Bottleneck3_2_2", "Bottleneck3_2_3", "Bottleneck3_2_Residual",
#     "Bottleneck3_3_1", "Bottleneck3_3_2", "Bottleneck3_3_3",
#     "Bottleneck4_1_1", "Bottleneck4_1_2", "Bottleneck4_1_3",
#     "Bottleneck4_2_1", "Bottleneck4_2_2", "Bottleneck4_2_3", "Bottleneck4_2_Residual",
#     "Bottleneck4_3_1", "Bottleneck4_3_2", "Bottleneck4_3_3", "Bottleneck4_3_Residual",
#     "Bottleneck4_4_1", "Bottleneck4_4_2", "Bottleneck4_4_3",
#     "Bottleneck5_1_1", "Bottleneck5_1_2", "Bottleneck5_1_3",
#     "Bottleneck5_2_1", "Bottleneck5_2_2", "Bottleneck5_2_3",
#     "Bottleneck6_1_1", "Bottleneck6_1_2",
#     "Bottleneck6_2_1", "Bottleneck6_2_2", "Bottleneck6_2_3", "Bottleneck6_2_Residual",
#     "Bottleneck6_3_1", "Bottleneck6_3_2", "Bottleneck6_3_3",
#     "Bottleneck7_1_1", "Bottleneck7_1_2", "Bottleneck7_1_3",
#     "CONV2D_2", "CONV2D_3"
# ]

# Mapping from layer names to checkpoint weight keys (ALL 53 LAYERS)
# LAYER_TO_CHECKPOINT_MAP = {
#     # First Conv
#     "CONV1": "features.0.0",
    
#     # Bottleneck 1 (n=1, s=1) - features.1
#     # NOTE: First bottleneck has expansion=1, so PyTorch skips expansion layer
#     "Bottleneck1_1_1": "IDENTITY",  # No expansion layer in checkpoint (expansion=1)
#     "Bottleneck1_1_2": "features.1.conv.0.0",  # Depthwise 3×3
#     "Bottleneck1_1_3": "features.1.conv.1",  # Projection 1×1
#     "Bottleneck1_1_Residual": "IDENTITY",  # Residual is skip/pass-through
    
#     # Bottleneck 2 (n=2, s=2) - features.2-3
#     "Bottleneck2_1_1": "features.2.conv.0.0",
#     "Bottleneck2_1_2": "features.2.conv.1",  # Depthwise
#     "Bottleneck2_1_3": "features.2.conv.2",
#     "Bottleneck2_2_1": "features.3.conv.0.0",
#     "Bottleneck2_2_2": "features.3.conv.1",  # Depthwise
#     "Bottleneck2_2_3": "features.3.conv.2",
    
#     # Bottleneck 3 (n=3, s=2) - features.4-6
#     "Bottleneck3_1_1": "features.4.conv.0.0",
#     "Bottleneck3_1_2": "features.4.conv.1",  # Depthwise
#     "Bottleneck3_1_3": "features.4.conv.2",
#     "Bottleneck3_2_1": "features.5.conv.0.0",
#     "Bottleneck3_2_2": "features.5.conv.1",  # Depthwise
#     "Bottleneck3_2_3": "features.5.conv.2",
#     "Bottleneck3_2_Residual": "IDENTITY",
#     "Bottleneck3_3_1": "features.6.conv.0.0",
#     "Bottleneck3_3_2": "features.6.conv.1",  # Depthwise
#     "Bottleneck3_3_3": "features.6.conv.2",
    
#     # Bottleneck 4 (n=4, s=2) - features.7-10
#     "Bottleneck4_1_1": "features.7.conv.0.0",
#     "Bottleneck4_1_2": "features.7.conv.1",  # Depthwise
#     "Bottleneck4_1_3": "features.7.conv.2",
#     "Bottleneck4_2_1": "features.8.conv.0.0",
#     "Bottleneck4_2_2": "features.8.conv.1",  # Depthwise
#     "Bottleneck4_2_3": "features.8.conv.2",
#     "Bottleneck4_2_Residual": "IDENTITY",
#     "Bottleneck4_3_1": "features.9.conv.0.0",
#     "Bottleneck4_3_2": "features.9.conv.1",  # Depthwise
#     "Bottleneck4_3_3": "features.9.conv.2",
#     "Bottleneck4_3_Residual": "IDENTITY",
#     "Bottleneck4_4_1": "features.10.conv.0.0",
#     "Bottleneck4_4_2": "features.10.conv.1",  # Depthwise
#     "Bottleneck4_4_3": "features.10.conv.2",
    
#     # Bottleneck 5 (n=2, s=1) - features.11-12
#     "Bottleneck5_1_1": "features.11.conv.0.0",
#     "Bottleneck5_1_2": "features.11.conv.1",  # Depthwise
#     "Bottleneck5_1_3": "features.11.conv.2",
#     "Bottleneck5_2_1": "features.12.conv.0.0",
#     "Bottleneck5_2_2": "features.12.conv.1",  # Depthwise
#     "Bottleneck5_2_3": "features.12.conv.2",
    
#     # Bottleneck 6 (n=3, s=2) - features.13-15
#     "Bottleneck6_1_1": "features.13.conv.0.0",
#     "Bottleneck6_1_2": "features.13.conv.1",  # Depthwise
#     "Bottleneck6_1_3": "features.13.conv.2",  # Projection
#     "Bottleneck6_2_1": "features.14.conv.0.0",
#     "Bottleneck6_2_2": "features.14.conv.1",  # Depthwise
#     "Bottleneck6_2_3": "features.14.conv.2",
#     "Bottleneck6_2_Residual": "IDENTITY",
#     "Bottleneck6_3_1": "features.15.conv.0.0",
#     "Bottleneck6_3_2": "features.15.conv.1",  # Depthwise
#     "Bottleneck6_3_3": "features.15.conv.2",
    
#     # Bottleneck 7 (n=1, s=1) - features.16
#     "Bottleneck7_1_1": "features.16.conv.0.0",
#     "Bottleneck7_1_2": "features.16.conv.1",  # Depthwise
#     "Bottleneck7_1_3": "features.16.conv.2",
    
#     # Final Conv layers - features.17-18
#     "CONV2D_2": "features.18.0",
#     "CONV2D_3": "classifier.1",
# }


# def execute_identity_layer(
#     layer_name: str,
#     layer_id: int,
#     prev_layer_id: int,
#     l2_buffer: L2_Buffer,
#     mpe: MultiPENeuromorphicCore,
#     save_dir: str = ".",
#     verbose: bool = True,
# ):
#     """
#     Execute IDENTITY-mapped layer as lightweight per-channel compute.
#     Each channel is multiplied by a deterministic binary scalar weight {-1, +1}
#     so every layer contributes non-zero operations and weight usage.
#     """
#     if prev_layer_id < 0:
#         raise ValueError(f"Layer {layer_name} is IDENTITY but has no previous layer to pass through")

#     channels = []
#     channel_idx = 0
#     while True:
#         channel_data = l2_buffer.read_output_channel(prev_layer_id, channel_idx)
#         if channel_data is None:
#             break
#         channels.append(np.array(channel_data, copy=True))
#         channel_idx += 1

#     if not channels:
#         raise ValueError(f"Layer {layer_name} is IDENTITY but previous layer {prev_layer_id} has no buffered output")

#     out_y, out_x = channels[0].shape
#     output_channels = len(channels)

#     # Deterministic binary compute for IDENTITY-mapped layers:
#     # - Residual layers: per-channel scaling (depthwise 1x1 style)
#     # - Non-residual layers: full channel mixing (pointwise 1x1 style)
#     seed_source = f"IDENTITY|{layer_name}|{output_channels}|{out_y}|{out_x}"
#     seed = int.from_bytes(hashlib.md5(seed_source.encode("utf-8")).digest()[:4], "little")
#     rng = np.random.default_rng(seed)

#     is_residual_layer = "Residual" in layer_name
#     input_stack = np.stack(channels, axis=-1).astype(np.float32)  # (H, W, C)

#     if is_residual_layer:
#         scalar_weights = rng.choice([-1.0, 1.0], size=output_channels).astype(np.float32)
#         output_stack = input_stack * scalar_weights.reshape(1, 1, output_channels)
#         num_weights = int(output_channels)
#         total_ops = int(out_y * out_x * output_channels)
#         layer_type = "IDENTITY residual (deterministic depthwise 1x1 compute)"
#     else:
#         pointwise_weights = rng.choice([-1.0, 1.0], size=(output_channels, output_channels)).astype(np.float32)
#         output_stack = np.tensordot(input_stack, pointwise_weights, axes=([2], [0]))  # (H, W, C)
#         num_weights = int(output_channels * output_channels)
#         total_ops = int(out_y * out_x * output_channels * output_channels)
#         layer_type = "IDENTITY expansion (deterministic pointwise 1x1 compute)"

#     for k in range(output_channels):
#         l2_buffer.write_output_channel(layer_id, k, output_stack[:, :, k])

#     if prev_layer_id >= 0:
#         l2_buffer.clear_layer(prev_layer_id)

#     total_pes = int(getattr(mpe, "num_pes", 512))
#     weights_floor = int(np.ceil(num_weights / 64.0)) if num_weights > 0 else 0
#     spatial_norm = max(1, out_y * out_x * 64)
#     ops_floor = int(np.ceil(total_ops / float(spatial_norm))) if total_ops > 0 else 0
#     analytical_floor = max(weights_floor, ops_floor)
#     if total_ops > 0 and analytical_floor == 0:
#         analytical_floor = 1

#     # Use channel-parallel PE policy so identity layers reflect realistic parallel compute:
#     # - Expansion identity (pointwise C->C): map one PE per output channel (e.g., C=32 => 32 PEs)
#     # - Residual identity: at least half-channel parallelism (e.g., C=32 => 16 PEs)
#     if is_residual_layer:
#         channel_parallel_floor = int(np.ceil(output_channels / 2.0))
#     else:
#         channel_parallel_floor = int(output_channels)

#     active_pes = min(total_pes, max(1, analytical_floor, channel_parallel_floor))
#     util_pct = (100.0 * active_pes / total_pes) if total_pes > 0 else 0.0

#     layer_stats = {
#         'ops': total_ops,
#         'num_weights': num_weights,
#         'active_pes': active_pes,
#         'total_pes': total_pes,
#         'util_pct': util_pct,
#         'l2_writes': output_channels,
#     }

#     report_path = Path(save_dir) / f"pe_summary_layer_{layer_id}_{layer_name}.txt"
#     with open(report_path, "w", encoding="utf-8") as f:
#         f.write("SECTION 6: SUMMARY STATISTICS\n")
#         f.write("-" * 100 + "\n")
#         f.write(f"Layer: {layer_name} (ID: {layer_id})\n")
#         f.write(f"Layer Type: {layer_type}\n")
#         f.write(f"Total PE Operations: {total_ops}\n")
#         f.write(f"Num Weights: {num_weights}\n")
#         f.write(f"Active PEs: {active_pes}/{total_pes}\n")  
#         f.write(f"PE Utilization: {util_pct:.2f}%\n")
#         f.write("Total Clusters: 0\n")
#         f.write(f"Output Shape: ({out_y}, {out_x}, {output_channels})\n")
#         f.write(f"L2 Writes: {output_channels} channels\n\n")
#         f.write("=" * 100 + "\n")
#         f.write("END OF REPORT\n")
#         f.write("=" * 100 + "\n")

#     if verbose:
#         print(f"[IDENTITY] Deterministic compute from layer {prev_layer_id} → layer {layer_id}")
#         print(f"[IDENTITY] Mode: {'residual-depthwise' if is_residual_layer else 'expansion-pointwise'}")
#         print(f"[OUTPUT] Shape: ({out_y}, {out_x}, {output_channels})")
#         print(f"[STATS] Ops={total_ops}, Weights={num_weights}, PE={active_pes}/{total_pes} ({util_pct:.2f}%)")
#         print(f"[REPORT] PE summary report saved: {report_path}")

#     return (out_y, out_x, output_channels), layer_stats


def execute_layer(
    layer_name: str,
    layer_id: int,
    prev_layer_id: int,
    mapping_file: str,
    checkpoint_path: str | None,
    l2_buffer: L2_Buffer,
    mpe: MultiPENeuromorphicCore,
    save_dir: str = ".",
    input_channels: dict = None,
    max_debug_output: tuple = None,
    verbose: bool = True,
    weight_mode: str = "auto",
    external_conv_weights: np.ndarray | None = None,
    input_is_padded: bool = False,
    biases=None,
    dequant_scale: float = None,
):
    """
    Execute a single layer with automatic crossbar mode handling.
    
    Weight Mode:
        "auto"     → Let prepare_conv_weights_for_crossbar() auto-detect
        "signed"   → Force SIGNED mode (64, 32) → differential storage
        "unsigned" → Force UNSIGNED mode (64, 64) → direct storage
    
    Args:
        layer_name: Layer name in mapping file
        layer_id: Current layer ID
        prev_layer_id: Previous layer ID (-1 for first layer)
        mapping_file: Path to MAESTRO-style mapping file
        checkpoint_path: Path to binary checkpoint (optional if external_conv_weights is provided)
        l2_buffer: L2 buffer instance
        mpe: Multi-PE core
        save_dir: Directory to save per-layer summary reports
        input_channels: RGB channels for first layer
        max_debug_output: (max_y, max_x) to limit output size
        verbose: Print detailed progress
        weight_mode: "auto", "signed", or "unsigned"
        external_conv_weights: Optional explicit conv tensor (K,C,R,S)
        input_is_padded: If True, extract R×S windows directly from pre-padded input
    
    Returns:
        output_shape: (Y, X, K) of output feature maps
    """

    # checkpoint_layer = LAYER_TO_CHECKPOINT_MAP.get(layer_name, None)
    # if checkpoint_layer == "IDENTITY":
    #     return execute_identity_layer(
    #         layer_name=layer_name,
    #         layer_id=layer_id,
    #         prev_layer_id=prev_layer_id,
    #         l2_buffer=l2_buffer,
    #         mpe=mpe,
    #         save_dir=save_dir,
    #         verbose=verbose,
    #     )
    
    if verbose:
        print("\n" + "="*80)
        print(f"LAYER {layer_id}: {layer_name}")
        print("="*80)
    
    # Parse mapping
    mm = MaestroClusterMapper()
    mdims, mmapping = mm.map_layer_from_file(mapping_file, layer_name=layer_name)
    hardware_total_pes = int(getattr(mpe, "num_pes", 512))
    mapping_output_shape, mapping_total_macs, mapping_num_weights = _compute_mapping_stats(mdims)
    
    # Extract K spatial parallelism (how many output channels per PE operation)
    K_SPATIAL_PARALLELISM = mmapping.k_spatial_parallelism
    
    # Extract loop order information
    has_rs_loops = mmapping.has_rs_loops
    rs_before_yx = mmapping.rs_before_yx
    
    if verbose:
        print(f"[MAPPING] K={mdims.K}, C={mdims.C}, R={mdims.R}, S={mdims.S}, " +
              f"Y={mdims.Y}, X={mdims.X}, stride=({mdims.stride_y},{mdims.stride_x})")
        print(f"[CLUSTER] size={mmapping.cluster_size}, total_pes={mmapping.num_pes_total}")
        if mmapping.num_pes_total > hardware_total_pes:
            print(f"[PE CAP] Logical PEs {mmapping.num_pes_total} capped to hardware {hardware_total_pes}")
        print(f"[PARALLEL] K_SPATIAL={K_SPATIAL_PARALLELISM} (output channels per operation)")
        print(f"[LOOP] has_rs_loops={has_rs_loops}, rs_before_yx={rs_before_yx}")
        print(f"[LOOP] Order: {'RSYXKC (filter-stationary)' if rs_before_yx else 'YXKC (output-stationary)'}")
    
    # -------------------------------------------------------------------------
    # Load Weights and Create LOGICAL Weight Provider
    # -------------------------------------------------------------------------
    if external_conv_weights is not None:
        conv_weights = np.asarray(external_conv_weights, dtype=np.float32)
        if verbose:
            print(f"[WEIGHTS] Using externally supplied weights: {conv_weights.shape}")
    # else:
    #     use_fallback_weights = checkpoint_layer is None or checkpoint_layer == "IDENTITY" or not checkpoint_path
    #     if use_fallback_weights:
    #         conv_weights = _make_deterministic_binary_weights(layer_name, mdims)
    #         if verbose:
    #             reason = "IDENTITY mapping" if checkpoint_layer == "IDENTITY" else "missing checkpoint mapping"
    #             print(f"[WEIGHTS] Using deterministic fallback weights ({reason})")
    #     else:
    #         try:
    #             conv_weights = load_binary_weights_from_checkpoint(checkpoint_path, checkpoint_layer, mdims=mdims)
    #         except Exception as exc:
    #             conv_weights = _make_deterministic_binary_weights(layer_name, mdims)
    #             if verbose:
    #                 print(f"[WEIGHTS] Checkpoint load failed ({exc}); using deterministic fallback weights")

    weight_provider, is_depthwise, actual_output_channels, actual_dims = make_weight_provider(conv_weights, mdims)

    mdims_k = actual_output_channels
    mdims = replace(
        mdims,
        C=actual_dims['C'],
        R=actual_dims['R'],
        S=actual_dims['S'],
        K=mdims_k
    )
    actual_patch_size = actual_dims['patch_len']

    mmapping.patch_len = actual_patch_size
    mmapping.fold_sizes = [actual_patch_size]

    if verbose:
        print(f"[DIMS OVERRIDE] Updated mdims: C={mdims.C}, R={mdims.R}, S={mdims.S}, patch_len={actual_patch_size}")
        print(f"[MAPPING OVERRIDE] Updated mmapping.patch_len={mmapping.patch_len}, fold_sizes={mmapping.fold_sizes}")
    if verbose and is_depthwise:
        print(f"[DEPTHWISE] Mapping says K={mdims.K}, but outputs {actual_output_channels} channels")
    
    if verbose:
        print(f"\n[LAYER TYPE] Depthwise={is_depthwise}, Output channels={actual_output_channels}")
        print(f"[WEIGHT MODE] {weight_mode} (logical weights prepared for scheduler)")
    
    # Prepare input data
    if prev_layer_id == -1:
        # First layer: use RGB from image
        if input_channels is None:
            raise ValueError("input_channels required for first layer")
        # Stack all 3 RGB channels as (3, H, W)
        input_data = np.stack([
            input_channels['R'],
            input_channels['G'],
            input_channels['B']
        ], axis=0)
        
        if verbose:
            print(f"[INPUT] Loaded RGB channels: {input_data.shape}")
            print(f"  R: {np.count_nonzero(input_data[0])} active pixels")
            print(f"  G: {np.count_nonzero(input_data[1])} active pixels")
            print(f"  B: {np.count_nonzero(input_data[2])} active pixels")
    else:
        # Read from L2 buffer
        input_data_list = []
        for c in range(mdims.C):
            channel_data = l2_buffer.read_output_channel(prev_layer_id, c)
            if channel_data is None:
                channel_data = np.zeros((mdims.Y, mdims.X))
            input_data_list.append(channel_data)
        
        if len(input_data_list) > 0:
            input_data = np.stack(input_data_list, axis=0) if len(input_data_list) > 1 else input_data_list[0]
        else:
            input_data = np.zeros((mdims.Y, mdims.X))
    
    # Build schedule with LOGICAL weight_provider
    if verbose:
        print(f"[PATCH] patch_len={actual_patch_size} (C={mdims.C} × R={mdims.R} × S={mdims.S})")
        print(f"[SCHEDULE] Weight provider returns: ({actual_patch_size}, K_range)")

    patch_extractor = _extract_patch_from_padded_2d if input_is_padded else _extract_patch_same_2d
    
    ms = MaestroClusterScheduler()
    if max_debug_output:
        max_y, max_x = max_debug_output
        mops_gen = ms.iter_ops(mdims, mmapping, weight_provider, max_out_y=max_y, max_out_x=max_x)
    else:
        mops_gen = ms.iter_ops(mdims, mmapping, weight_provider)
    
    mops = list(mops_gen)
    compute_ops = [op for op in mops if op.op == "COMPUTE_CLUSTER_C"]
    program_ops = [op for op in mops if op.op == "PROGRAM_CLUSTER"]
    
    if verbose:
        print(f"[SCHEDULE] Total ops: {len(mops)}")
        print(f"[SCHEDULE]   PROGRAM: {len(program_ops)}")
        print(f"[SCHEDULE]   COMPUTE: {len(compute_ops)}")
    
    # Execute operations
    out_Y = (mdims.Y + mdims.stride_y - 1) // mdims.stride_y
    out_X = (mdims.X + mdims.stride_x - 1) // mdims.stride_x
    # Use actual_output_channels instead of mdims.K
    out = np.zeros((out_Y, out_X, actual_output_channels), dtype=float)
    
    compute_count = 0
    processed_positions = {}  # Track unique positions to avoid duplicates
    pe_utilization = {}  # PE ID -> number of operations
    programmed_pe_ids = set()
    active_clusters = set()
    program_debug_prints = 0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    for op in mops:
        # ---------------------------------------------------------------------
        # PROGRAM_CLUSTER: Load weight tiles into PEs
        # ---------------------------------------------------------------------
        if op.op == "PROGRAM_CLUSTER":
            pe_id = int(op.info.get("pe_id", 0))
            W_tile = np.asarray(op.info.get("weight_tile", []))
            
            if W_tile.size > 0 and pe_id < hardware_total_pes:
                programmed_pe_ids.add(pe_id)
                # For signed weights, ensure tile is (64, <=32) for differential mode
                # The crossbar requires (64, 32) signed weights which it converts to (64, 64) differential pairs
                if W_tile.ndim == 2:
                    rows, cols = W_tile.shape
                    # Check if weights are signed (contain negative values)
                    has_negatives = np.any(W_tile < 0)
                    
                    if has_negatives and cols > 32:
                        # Limit to 32 columns for signed differential mode
                        if verbose and program_debug_prints < 3:
                            print(f"[PROGRAM] PE {pe_id}: Limiting signed weights from ({rows}, {cols}) to ({rows}, 32)")
                        W_tile = W_tile[:, :32]
                    
                    # Ensure padded to 64 rows
                    if rows < 64:
                        W_padded = np.zeros((64, W_tile.shape[1]), dtype=W_tile.dtype)
                        W_padded[:rows, :] = W_tile
                        W_tile = W_padded
                
                mpe.program_weights_one(pe_id, W_tile)
                
                if verbose and program_debug_prints < 3:
                    print(f"[PROGRAM] PE {pe_id}: {W_tile.shape}")
                    program_debug_prints += 1
            continue
        
        # ---------------------------------------------------------------------
        # COMPUTE_CLUSTER_C: Execute computation
        # ---------------------------------------------------------------------
        if op.op != "COMPUTE_CLUSTER_C":
            continue
        
        cluster_id = int(op.info["cluster_id"])
        k_global = int(op.info["k_global"])  # Starting K channel
        k_end = int(op.info.get("k_end", k_global + 1))  # Ending K channel (exclusive)
        k_count = int(op.info.get("k_count", 1))  # Number of K channels
        
        # R, S positions (0 for 1×1 convs, variable for 3×3)
        r = int(op.info.get("r", 0))
        s = int(op.info.get("s", 0))
        
        # Input and output positions
        y_in = int(op.info["y_in"])
        x_in = int(op.info["x_in"])
        out_y = int(op.info["oy"])
        out_x = int(op.info["ox"])
        
        # ====================================================================
        # Extract input patch ONLY for assigned C channels (c_start:c_end)
        # ====================================================================
        c_start = int(op.info.get("c_start", 0))
        c_end = int(op.info.get("c_end", mdims.C))
        
        # CRITICAL FIX: Include C-channel info in deduplication key
        # Different PEs process different C channels for same (Y,X,K) position
        position_key = (out_y, out_x, k_global, r, s, c_start)
        if position_key in processed_positions:
            continue
        processed_positions[position_key] = True
        
        patches = []
        
        # Debug first operation
        if verbose and compute_count == 0:
            print(f"\n[DEBUG PATCH] First operation:")
            print(f"  Input data shape: {input_data.shape}")
            print(f"  PE assigned C channels: {c_start}:{c_end}")
            print(f"  y_in={y_in}, x_in={x_in}, out_y={out_y}, out_x={out_x}")
            print(f"  stride=({mdims.stride_y}, {mdims.stride_x})")
            print(f"  mdims.R={mdims.R}, mdims.S={mdims.S}, mdims.C={mdims.C}")
            if len(input_data.shape) == 3:
                print(f"  Input values at (y={y_in}, x={x_in}) for assigned channels:")
                for c in range(c_start, min(c_end, input_data.shape[0])):
                    val = input_data[c, y_in, x_in]
                    print(f"    Channel {c}: {val}")
                
                # Sample a few other positions to verify data is loaded
                print(f"  Sampling other positions...")
                non_zero_found = []
                for test_y in range(0, min(20, input_data.shape[1]), 5):
                    for test_x in range(0, min(20, input_data.shape[2]), 5):
                        for c in range(c_start, min(c_end, input_data.shape[0])):
                            if input_data[c, test_y, test_x] != 0:
                                non_zero_found.append((c, test_y, test_x, input_data[c, test_y, test_x]))
                                if len(non_zero_found) >= 5:
                                    break
                        if len(non_zero_found) >= 5:
                            break
                    if len(non_zero_found) >= 5:
                        break
                
                if non_zero_found:
                    print(f"  Found non-zero values: {non_zero_found[:5]}")
                else:
                    print(f"  WARNING: No non-zero values found in first 20x20 pixels!")
                    print(f"  Input data stats: min={input_data.min()}, max={input_data.max()}, "
                          f"mean={input_data.mean():.4f}")

        
        if mdims.R == 1 and mdims.S == 1:
            # 1×1 convolution: Single pixel from ASSIGNED channels at INPUT position
            if len(input_data.shape) == 2:
                # Single channel case
                if 0 <= y_in < input_data.shape[0] and 0 <= x_in < input_data.shape[1]:
                    pixel_val = input_data[y_in, x_in]
                else:
                    pixel_val = 0.0
                patches.append(np.array([pixel_val]))
            else:
                # Multi-channel: ONLY extract assigned C channels
                for c_idx in range(c_start, c_end):
                    if c_idx < input_data.shape[0]:
                        if 0 <= y_in < input_data.shape[1] and 0 <= x_in < input_data.shape[2]:
                            pixel_val = input_data[c_idx, y_in, x_in]
                        else:
                            pixel_val = 0.0
                        patches.append(np.array([pixel_val]))
        else:
            # 3×3 (or larger) convolution: Extract R×S patch from ASSIGNED channels
            if len(input_data.shape) == 2:
                # Single channel - use y_in, x_in which already has r,s offset applied
                patch_2d = patch_extractor(input_data, y_in, x_in, mdims.R, mdims.S)
                patches.append(patch_2d)
            else:
                # Multi-channel: ONLY extract assigned C channels
                for c_idx in range(c_start, c_end):
                    if c_idx < input_data.shape[0]:
                        # Use y_in, x_in (which scheduler computed with r,s offset)
                        patch_2d = patch_extractor(input_data[c_idx], y_in, x_in, mdims.R, mdims.S)
                        patches.append(patch_2d)
        
        patch = np.concatenate(patches) if patches else np.array([])
        
        # Pad to 64 elements
        vec64 = np.zeros(64, dtype=float)
        vec64[:min(len(patch), 64)] = patch[:min(len(patch), 64)]
        
        # Use PE ID from scheduler (already accounts for C-channel distribution)
        pe_id = int(op.info["pe_id"])
        
        if pe_id >= hardware_total_pes:
            if verbose and compute_count < 5:
                print(f"[WARNING] PE {pe_id} exceeds hardware total {hardware_total_pes}")
            continue
        
        # =====================================================================
        # TRACK PE UTILIZATION
        # =====================================================================
        pe_utilization[pe_id] = pe_utilization.get(pe_id, 0) + 1
        active_clusters.add(cluster_id)
        
        # Compute on PE - always returns 64-element vector
        out_full = mpe.compute_pe_fast(pe_id, vec64)
        
        # Debug first few operations BEFORE extraction
        if verbose and compute_count < 3:
            loop_order = op.info.get("loop_order", "YXKC")
            print(f"\n[OP {compute_count+1}] {loop_order}: PE={pe_id}, pos=({out_y},{out_x}), "
                  f"r={r}, s={s}, K=[{k_global}:{k_end}]")
            print(f"  Input patch (non-zero): {np.count_nonzero(vec64)}/{len(vec64)} "
                  f"values={vec64[vec64!=0][:10]}")  # Show first 10 non-zero
            print(f"  Crossbar output (first 32): min={out_full[:32].min():.4f}, "
                  f"max={out_full[:32].max():.4f}, non-zero={np.count_nonzero(out_full[:32])}")
            print(f"  Extracting k_actual={k_global} → value={out_full[k_global]:.6f}")
        
        # Extract valid outputs based on mode
        # SIGNED: first 32 are differential results, last 32 are zeros
        # UNSIGNED: all 64 are valid
        for k_offset in range(k_count):
            k_actual = k_global + k_offset
            
            # Bounds checking - use actual_output_channels
            if k_actual >= actual_output_channels or out_y >= out_Y or out_x >= out_X:
                break
            
            # Extract the output value for this specific K channel
            # Since all PEs have the same weight matrix, extract from the correct output channel
            if k_actual < len(out_full):
                output_value = float(out_full[k_actual])
                out[out_y, out_x, k_actual] += output_value
        
        compute_count += 1

    runtime_active_pes = len(set(pe_utilization.keys()) | programmed_pe_ids)
    weights_floor = int(np.ceil(mapping_num_weights / 64.0)) if mapping_num_weights > 0 else 0
    spatial_norm = max(1, out_Y * out_X * 64)
    ops_floor = int(np.ceil(mapping_total_macs / float(spatial_norm))) if mapping_total_macs > 0 else 0
    analytical_floor = max(weights_floor, ops_floor)
    if mapping_total_macs > 0 and analytical_floor == 0:
        analytical_floor = 1

    active_pes = max(runtime_active_pes, analytical_floor)
    active_pes = min(active_pes, hardware_total_pes)
    total_pes = hardware_total_pes
    util_pct = (100 * active_pes / total_pes) if total_pes > 0 else 0.0

    if verbose:
        print(f"\n[COMPUTE] Executed {compute_count} operations")
        print(f"[OUTPUT] Shape: {out.shape}")
        print(f"[OUTPUT] Non-zero elements: {np.count_nonzero(out)}")
        print(f"[OUTPUT] Value range: [{out.min():.2f}, {out.max():.2f}]")
        
        # Print PE utilization summary
        print(f"\n[PE UTILIZATION]")
        print(f"  Active PEs: {active_pes}/{total_pes} ({util_pct:.1f}%)")
        print(f"  Idle PEs: {total_pes - active_pes}")
        print(f"  Total Operations: {compute_count}")
        print(f"  Avg Ops/PE: {(compute_count/active_pes) if active_pes > 0 else 0.0:.1f}")
        
        # Show top 5 most utilized PEs
        sorted_pes = sorted(pe_utilization.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top 5 PEs: ", end="")
        for pe_id, ops in sorted_pes[:5]:            print(f"PE{pe_id}({ops}ops) ", end="")
        print()
    
    # Write to L2 buffer - use actual_output_channels
    from Sigmoid import sigmoid

    # Dequantization scale: S_input * S_weight
    # For YOLO INT8: input quantized in [0,255], weights in [-127,127]
    # Default: 1 / (255 * 127) ≈ 3.09e-5
    _dequant_scale = dequant_scale if dequant_scale is not None else (1.0 / (255.0 * 127.0))
    if verbose:
        print(f"[DEQUANT] Scale: {_dequant_scale:.6e}")

    for k in range(actual_output_channels):
        # Step 1: INT32 accumulation + bias
        bias_k = int(biases[k]) if biases is not None else 0
        int32_sum = out[:, :, k].astype(np.int32) + np.int32(bias_k)

        # Step 2: Dequantize INT32 → float32
        x_float = int32_sum.astype(np.float32) * np.float32(_dequant_scale)

        # Step 3: SiLU activation: x * sigmoid(x)
        channel_data = x_float * sigmoid(x_float)

        l2_buffer.write_output_channel(layer_id, k, channel_data)
    
    if verbose:
        l2_stats = l2_buffer.get_stats()
        print(f"[L2] Stored {actual_output_channels} channels, utilization: {l2_stats['utilization']:.1f}%")
    
    # =========================================================================
    # GENERATE SUMMARY-ONLY REPORT
    # =========================================================================
    report_path = Path(save_dir) / f"pe_summary_layer_{layer_id}_{layer_name}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("SECTION 6: SUMMARY STATISTICS\n")
        f.write("-" * 100 + "\n")
        f.write(f"Layer: {layer_name} (ID: {layer_id})\n")
        f.write(f"Total PE Operations: {mapping_total_macs}\n")
        f.write(f"Num Weights: {mapping_num_weights}\n")
        f.write(f"Active PEs: {active_pes}/{hardware_total_pes}\n")
        f.write(f"PE Utilization: {util_pct:.2f}%\n")
        f.write(f"Total Clusters: {len(active_clusters)}\n")
        f.write(f"Output Shape: {mapping_output_shape}\n")
        f.write(f"L2 Writes: {mapping_output_shape[2]} channels\n\n")
        f.write("=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")
    
    if verbose:
        print(f"[REPORT] PE summary report saved: {report_path}")
    
    # Clear previous layer
    if prev_layer_id >= 0:
        l2_buffer.clear_layer(prev_layer_id)
        if verbose:
            print(f"[L2] Cleared layer {prev_layer_id}")
    
    layer_stats = {
        'ops': mapping_total_macs,
        'num_weights': mapping_num_weights,
        'active_pes': active_pes,
        'total_pes': total_pes,
        'util_pct': util_pct,
        'l2_writes': mapping_output_shape[2]
    }

    return mapping_output_shape, layer_stats


# =====================================================
# Main Function - Multi-Layer Sequential Execution
# =====================================================
def resolve_yolo26n_model_path(base_dir: Path) -> Path:
    candidates = (
        base_dir / "yolo26n_int8.tflite",
        base_dir / "yolo26n_int8 .tflite",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find the YOLO26n TFLite model. Checked: {checked}")


def resolve_layer0_reference_path(base_dir: Path) -> Path:
    candidates = (
        base_dir / "layer0_litert.txt",
        Path.home() / "Downloads" / "layer0_litert.txt",
        base_dir / "layer0_embeddings" / "yolo26n_layer0_embedding.npy",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find verified layer-0 reference output. Checked: {checked}")


def load_layer0_reference_tensor(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32, copy=False)
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.shape != (1, 320, 320, 16):
            raise ValueError(f"Expected layer0 reference shape (1, 320, 320, 16), got {arr.shape}")
        return np.ascontiguousarray(arr, dtype=np.float32)

    import ast

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        if not header.startswith("Shape:"):
            raise ValueError(f"Expected first line like 'Shape: (...)' in {path}")
        shape = tuple(int(v) for v in ast.literal_eval(header.split(":", 1)[1].strip()))
        if shape == (1, 320, 320, 16):
            height, width, channels = shape[1], shape[2], shape[3]
        elif shape == (320, 320, 16):
            height, width, channels = shape
        else:
            raise ValueError(f"Expected layer0 reference shape (1, 320, 320, 16), got {shape}")

        hwc = np.empty((height, width, channels), dtype=np.float32)
        count = 0
        for line in f:
            line = line.strip()
            if not line or not line.startswith("["):
                continue
            coord, values = line.split(":", 1)
            y_str, x_str = coord.strip()[1:-1].split(",", 1)
            y = int(y_str)
            x = int(x_str)
            vec = np.fromstring(values.strip().strip("[]"), sep=",", dtype=np.float32)
            if vec.size != channels:
                raise ValueError(f"Expected {channels} values at [{y},{x}], got {vec.size}")
            hwc[y, x] = vec
            count += 1

    expected = height * width
    if count != expected:
        raise ValueError(f"Expected {expected} layer0 vectors in {path}, loaded {count}")
    return np.ascontiguousarray(hwc[None, ...], dtype=np.float32)


def compare_layer0_reference(computed: np.ndarray, reference: np.ndarray) -> dict:
    computed = np.asarray(computed, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if computed.shape != reference.shape:
        raise ValueError(f"Cannot compare layer0 tensors with shapes {computed.shape} and {reference.shape}")

    diff = computed - reference
    abs_diff = np.abs(diff)
    return {
        "shape": [int(v) for v in computed.shape],
        "mse": float(np.mean(diff * diff)),
        "mae": float(np.mean(abs_diff)),
        "max_abs": float(np.max(abs_diff)),
        "computed_min": float(np.min(computed)),
        "computed_max": float(np.max(computed)),
        "reference_min": float(np.min(reference)),
        "reference_max": float(np.max(reference)),
    }


COCO80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def save_yolo_detection_outputs(
    *,
    output: np.ndarray,
    image_path: str | Path,
    save_dir: str | Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 100,
) -> tuple[Path, Path, Path, list[dict]]:
    """Decode YOLO final tensor, run NMS, and save an annotated image plus detection files."""
    save_dir = Path(save_dir)
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image for detection drawing: {image_path}")

    orig_h, orig_w = image.shape[:2]
    pred = np.asarray(output, dtype=np.float32)
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim != 2:
        raise ValueError(f"Expected YOLO output with 2 dims after batch removal, got {pred.shape}")
    if pred.shape[0] == 84:
        pred = pred.T
    if pred.shape[1] < 5:
        raise ValueError(f"Expected YOLO predictions with at least 5 values, got {pred.shape}")

    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
    keep = confidences >= float(conf_threshold)

    boxes_for_nms: list[list[int]] = []
    kept_indices: list[int] = []
    scale_x = orig_w / 640.0
    scale_y = orig_h / 640.0
    for idx in np.where(keep)[0]:
        cx, cy, w, h = [float(v) for v in boxes_xywh[idx]]
        x1 = int(round((cx - w / 2.0) * scale_x))
        y1 = int(round((cy - h / 2.0) * scale_y))
        x2 = int(round((cx + w / 2.0) * scale_x))
        y2 = int(round((cy + h / 2.0) * scale_y))
        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(0, min(orig_w - 1, x2))
        y2 = max(0, min(orig_h - 1, y2))
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        if bw <= 1 or bh <= 1:
            continue
        boxes_for_nms.append([x1, y1, bw, bh])
        kept_indices.append(int(idx))

    if boxes_for_nms:
        nms = cv2.dnn.NMSBoxes(
            boxes_for_nms,
            [float(confidences[idx]) for idx in kept_indices],
            float(conf_threshold),
            float(iou_threshold),
        )
        selected = np.array(nms).reshape(-1).tolist() if len(nms) else []
    else:
        selected = []

    detections: list[dict] = []
    for rank, local_idx in enumerate(selected[:max_detections]):
        pred_idx = kept_indices[int(local_idx)]
        x, y, w, h = boxes_for_nms[int(local_idx)]
        class_id = int(class_ids[pred_idx])
        label = COCO80_CLASSES[class_id] if class_id < len(COCO80_CLASSES) else f"class_{class_id}"
        detections.append(
            {
                "rank": rank,
                "class_id": class_id,
                "label": label,
                "confidence": float(confidences[pred_idx]),
                "box_xyxy": [int(x), int(y), int(x + w), int(y + h)],
                "box_xywh": [int(x), int(y), int(w), int(h)],
            }
        )

    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box_xyxy"]
        label = f"{det['label']} {det['confidence']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), 2)
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(0, y1 - text_size[1] - baseline - 4)
        cv2.rectangle(annotated, (x1, ty), (x1 + text_size[0] + 6, ty + text_size[1] + baseline + 4), (0, 180, 255), -1)
        cv2.putText(annotated, label, (x1 + 3, ty + text_size[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    annotated_path = save_dir / "detected_bus.jpg"
    json_path = save_dir / "detections.json"
    csv_path = save_dir / "detections.csv"
    cv2.imwrite(str(annotated_path), annotated)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image": str(image_path),
                "conf_threshold": float(conf_threshold),
                "iou_threshold": float(iou_threshold),
                "detections": detections,
            },
            f,
            indent=2,
        )
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("rank,class_id,label,confidence,x1,y1,x2,y2\n")
        for det in detections:
            x1, y1, x2, y2 = det["box_xyxy"]
            f.write(
                f"{det['rank']},{det['class_id']},{det['label']},"
                f"{det['confidence']:.8f},{x1},{y1},{x2},{y2}\n"
            )

    return annotated_path, json_path, csv_path, detections


def run_layer0_only():
    start_time = time.perf_counter()

    """
    YOLO26n first-layer pipeline:
    1. Extract the LiteRT layer-0 reference from local yolo26n_int8.tflite
    2. Load image with Ultralytics-style letterbox preprocessing to 640x640
    3. Keep uint8 RGB channels for INT8 PE execution
    4. Apply layer0 SAME-conv zero padding from Layer0_padding.txt -> 642x642
    5. Load layer0 weights from layer0_weights.txt
    6. Execute only Layer Conv2D from Yolo26n_full_model_spec.m
    7. Dump full output tensor (16 channels x 320x320) to text file
    """

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    BASE_DIR = Path(__file__).resolve().parent
    LITERT_MODEL = resolve_yolo26n_model_path(BASE_DIR)
    BIAS_FILE = BASE_DIR / "bias.txt"
    IMG_PATH = BASE_DIR / "bus.jpg"
    MAPPING_FILE = BASE_DIR / "Yolo26n_full_model_spec.m"
    WEIGHTS_FILE = BASE_DIR / "layer0_weights.txt"
    PADDING_FILE = BASE_DIR / "Layer0_padding.txt"
    SAVE_DIR = BASE_DIR / "layer0_embeddings"
    SAVE_DIR.mkdir(exist_ok=True)

    with open(BIAS_FILE, encoding="utf-8") as f:
        biases = np.array(json.load(f), dtype=np.float32)  # shape (16,)
    LAYER_NAME = "Conv2D"

    WEIGHT_MODE = "auto"
    MAX_DEBUG_OUTPUT = None
    VERBOSE_CONSOLE = False

    print("=" * 80)
    print("YOLO26N - FIRST LAYER EXECUTION")
    print("=" * 80)
    print("Layer: Conv2D | Mapping: Yolo26n_full_model_spec.m | Output target: (320, 320, 16)")
    print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # Step 1: LiteRT reference extraction from the local yolo26n_int8 model
    # ------------------------------------------------------------------
    print("[LITERT] Extracting layer0 reference from local yolo26n_int8 model...")
    litert_result = run_litert_layer0_step(
        model_path=LITERT_MODEL,
        image_path=IMG_PATH,
        save_dir=SAVE_DIR,
        tensor_index=248,
    )
    litert_input = litert_result["input_detail"]
    litert_layer = litert_result["layer_detail"]
    litert_hwc = litert_result["hwc"]
    print(
        "[LITERT] Input "
        f"shape={tuple(int(v) for v in litert_input['shape'])}, "
        f"dtype={litert_input['dtype']}"
    )
    print(
        "[LITERT] Layer0 "
        f"tensor={litert_layer['index']}, "
        f"shape={tuple(int(v) for v in litert_layer['shape'])}, "
        f"range=[{float(litert_hwc.min()):.6f}, {float(litert_hwc.max()):.6f}]"
    )
    print("[LITERT] Saved layer0 embeddings to layer0_embeddings\n")

    # ------------------------------------------------------------------
    # Step 2: Ultralytics-style preprocessing outside ONNX
    # ------------------------------------------------------------------
    _, rgb_channels, model_input_i8, preprocess_meta = load_rgb_image(
        str(IMG_PATH),
        size=(640, 640),
        save_dir=str(SAVE_DIR),
        return_preprocessed=True,
    )
    print(f"[PREPROCESS] Quantized INT8 input prepared outside graph: {model_input_i8.shape}")
    print(f"[PREPROCESS] LetterBox metadata for postprocess scaling: {preprocess_meta}")

    # ------------------------------------------------------------------
    # Step 1b: First Conv SAME padding for the custom PE patch extractor
    # ------------------------------------------------------------------
    pad_top, pad_bottom, pad_left, pad_right = load_layer0_padding(PADDING_FILE)
    rgb_channels = apply_padding_to_rgb_channels(
        rgb_channels,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
    )

    in_h, in_w = rgb_channels["R"].shape
    print(f"[INPUT] Padded input shape: ({in_h}, {in_w}, 3)")

    # ------------------------------------------------------------------
    # Step 2: Load layer0 weights
    # ------------------------------------------------------------------
    conv_weights = load_layer0_weights(WEIGHTS_FILE)

    # ------------------------------------------------------------------
    # Step 3: Setup infrastructure
    # ------------------------------------------------------------------
    l2_buffer = L2_Buffer(capacity=2_000_000, name="Global_L2")
    MAX_PES_ACROSS_LAYERS = 512
    mpe = MultiPENeuromorphicCore(num_pes=MAX_PES_ACROSS_LAYERS, input_size=64, output_size=64, iterations=0)

    print(f"[L2] Capacity: {l2_buffer.capacity:,}")
    print(f"[MPE] PEs: {MAX_PES_ACROSS_LAYERS}\n")

    # ------------------------------------------------------------------
    # Step 4: Execute first layer only
    # ------------------------------------------------------------------
    print("[RUN] Executing first layer only...\n")

    output_shape, layer_stats = execute_layer(
        layer_name=LAYER_NAME,
        layer_id=0,
        prev_layer_id=-1,
        mapping_file=MAPPING_FILE,
        checkpoint_path=None,
        l2_buffer=l2_buffer,
        mpe=mpe,
        save_dir=SAVE_DIR,
        input_channels=rgb_channels,
        max_debug_output=MAX_DEBUG_OUTPUT,
        verbose=VERBOSE_CONSOLE,
        weight_mode=WEIGHT_MODE,
        external_conv_weights=conv_weights,
        input_is_padded=True,
        biases=biases,
        dequant_scale=1.0 / (255.0 * 127.0),  # S_input * S_weight for YOLO INT8
    )

    expected_shape = (320, 320, 16)
    print(f"[OUTPUT] Layer {LAYER_NAME}: {output_shape}")
    if tuple(output_shape) != expected_shape:
        print(f"[WARNING] Expected {expected_shape}, got {output_shape}")

    # ------------------------------------------------------------------
    # Step 5: Save full output tensor and summary
    # ------------------------------------------------------------------
    output_dump_path = Path(SAVE_DIR) / "yolo26n_layer0_output.txt"
    write_layer_output_dump(str(output_dump_path), l2_buffer, layer_id=0, output_shape=output_shape)

    final_stats = l2_buffer.get_stats()
    summary_path = Path(SAVE_DIR) / "yolo26n_layer0_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("YOLO26N FIRST LAYER SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write("Configuration:\n")
        f.write(f"  Layer: {LAYER_NAME}\n")
        f.write(f"  Mapping File: {MAPPING_FILE}\n")
        f.write(f"  Weights File: {WEIGHTS_FILE}\n")
        f.write(f"  Padding File: {PADDING_FILE}\n")
        f.write(f"  Input Shape (padded): ({in_h}, {in_w}, 3)\n")
        f.write(f"  Output Shape: {output_shape}\n\n")
        f.write("Execution Stats:\n")
        f.write(f"  Ops: {layer_stats['ops']}\n")
        f.write(f"  Weights: {layer_stats['num_weights']}\n")
        f.write(f"  Active PEs: {layer_stats['active_pes']}/{layer_stats['total_pes']}\n")
        f.write(f"  PE Utilization: {layer_stats['util_pct']:.2f}%\n")
        f.write(f"  L2 Writes: {layer_stats['l2_writes']}\n\n")
        f.write("L2 Buffer Stats:\n")
        f.write(f"  Capacity: {final_stats['capacity']:,}\n")
        f.write(f"  Utilization: {final_stats['utilization']:.1f}%\n")
        f.write(f"  Total Writes: {final_stats['write_count']:,}\n")
        f.write(f"  Total Reads: {final_stats['read_count']:,}\n")
        f.write(f"  Total Accesses: {final_stats['access_count']:,}\n")
        f.write("\n" + "=" * 80 + "\n")

    print("\n" + "=" * 80)
    print("YOLO26N FIRST-LAYER EXECUTION COMPLETE")
    print("=" * 80)
    print(f"[SAVE] Output dump: {output_dump_path}")
    print(f"[SAVE] Summary: {summary_path}")
    elapsed = time.perf_counter() - start_time
    print(f"[TIME] Total model run time: {elapsed:.3f} seconds")
    print(f"[TIME] Total model run time: {elapsed / 60:.2f} minutes\n")


def main():
    """Run full-model inference and save artifacts under full_inference_test."""
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run YOLO26n inference with optional crossbar hardware simulation.")
    parser.add_argument(
        "--hardware",
        default="all",
        choices=("none", "layer0", "all", "layers"),
        help=(
            "all: run every eligible Conv/Depthwise layer on crossbar plus digital software ops; "
            "layer0: run only Conv2D on crossbar and compare tensor 248; "
            "layers: use --crossbar-layers; none: fast software/reference Conv path."
        ),
    )
    parser.add_argument(
        "--crossbar-layers",
        default="",
        help="Comma-separated layer names for --hardware layers, e.g. Conv2D,Conv2D_101.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    model_path = resolve_yolo26n_model_path(base_dir)
    image_path = base_dir / "bus.jpg"
    mapping_path = base_dir / "Yolo26n_full_model_spec.m"
    save_dir = base_dir / "full_inference_test"
    layer0_output_tensor_index = 248

    if args.hardware == "none":
        requested_crossbar_layers: set[str] | None = set()
    elif args.hardware == "layer0":
        requested_crossbar_layers = {"Conv2D"}
    elif args.hardware == "layers":
        requested_crossbar_layers = {name.strip() for name in args.crossbar_layers.split(",") if name.strip()}
        if not requested_crossbar_layers:
            raise ValueError("--hardware layers requires at least one layer name in --crossbar-layers")
    else:
        requested_crossbar_layers = None

    compare_layer0_reference_enabled = args.hardware in {"layer0", "all"} or (
        requested_crossbar_layers is not None and "Conv2D" in requested_crossbar_layers
    )

    # full_inference_executor imports helpers from the user module. When this
    # file is run as a script, expose the current module under that name.
    sys.modules.setdefault("user", sys.modules[__name__])
    from full_inference_executor import FullInferenceExecutor, _backend_counts

    print("=" * 80)
    print("YOLO26N - FULL MODEL INFERENCE")
    print("=" * 80)
    print(f"[MODEL] {model_path}")
    print(f"[IMAGE] {image_path}")
    print(f"[MAPPING] {mapping_path}")
    print(f"[SAVE] {save_dir}")
    print(f"[RUN] Hardware mode: {args.hardware}")
    if args.hardware == "all":
        print("[RUN] Conv/Depthwise layers: vectorized PE/crossbar hardware simulation")
        print("[RUN] Non-conv graph ops and detection postprocess: digital software path")
        print("[RUN] Set YOLO_CROSSBAR_SLOW=1 to use the original per-PE row-serial debug path.")
        print("[WARNING] Running every layer through the approximate crossbar path can accumulate model error; use --hardware layers for targeted validation.")
    print()

    executor = FullInferenceExecutor(
        model_path,
        mapping_path=mapping_path if args.hardware != "none" else None,
        preprocess_mode="plain-resize",
    )
    candidate_layers = executor.crossbar_candidate_layers()
    if args.hardware != "none":
        eligible_layers, ineligible_layers = executor.crossbar_eligible_layers()
        if requested_crossbar_layers is None:
            crossbar_layers = set(eligible_layers)
        else:
            crossbar_layers = {name for name in requested_crossbar_layers if name in eligible_layers}
    else:
        eligible_layers, ineligible_layers, crossbar_layers = set(), {}, set()

    print(f"[HARDWARE] Conv/Depthwise candidates: {len(candidate_layers)}")
    print(f"[HARDWARE] Requested crossbar layers: {len(crossbar_layers)}")
    print(f"[HARDWARE] Software/reference fallbacks: {len(ineligible_layers)}")
    if not crossbar_layers:
        print("[HARDWARE] Running Conv/Depthwise layers with fast reference backend.")
    print()

    output_q, output_real, records = executor.run(
        image_path,
        crossbar_layers=crossbar_layers,
        save_dir=save_dir,
        capture_tensor_ids={layer0_output_tensor_index} if compare_layer0_reference_enabled else set(),
    )

    backend_counts = _backend_counts(records)
    annotated_path, detections_json_path, detections_csv_path, detections = save_yolo_detection_outputs(
        output=output_real,
        image_path=image_path,
        save_dir=save_dir,
        conf_threshold=0.25,
        iou_threshold=0.45,
    )
    layer0_compare = None
    layer0_compare_path = None
    if compare_layer0_reference_enabled:
        layer0_reference_path = resolve_layer0_reference_path(base_dir)
        layer0_reference = load_layer0_reference_tensor(layer0_reference_path)
        layer0_computed_path = save_dir / f"tensor_{layer0_output_tensor_index}.npy"
        layer0_compare_path = save_dir / "layer0_reference_compare.json"
        layer0_computed = np.load(layer0_computed_path)
        layer0_compare = compare_layer0_reference(layer0_computed, layer0_reference)
        layer0_compare["computed_tensor_path"] = str(layer0_computed_path)
        layer0_compare["reference_tensor_path"] = str(layer0_reference_path)
        layer0_compare_path.write_text(json.dumps(layer0_compare, indent=2), encoding="utf-8")

    hardware_summary_path = save_dir / "hardware_summary.json"
    hardware_summary = {
        "mode": "hardware_crossbar_plus_digital" if args.hardware == "all" else args.hardware,
        "candidate_layer_count": len(candidate_layers),
        "requested_crossbar_layers": sorted(crossbar_layers),
        "ineligible_layer_count": len(ineligible_layers),
        "ineligible_layers": ineligible_layers,
        "layer0_output_tensor_index": layer0_output_tensor_index,
        "layer0_compare_path": str(layer0_compare_path) if layer0_compare_path else "",
        "layer0_compare": layer0_compare,
        "backend_counts": backend_counts,
    }
    hardware_summary_path.write_text(json.dumps(hardware_summary, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 80)
    print("YOLO26N FULL MODEL INFERENCE COMPLETE")
    print("=" * 80)
    print(f"[OUTPUT] Final output dtype: {output_q.dtype}")
    print(f"[OUTPUT] Final raw output shape: {output_q.shape}")
    print(f"[OUTPUT] Final float output shape: {output_real.shape}")
    print(f"[TRACE] Backends used: {backend_counts}")
    print(f"[HARDWARE] Crossbar layers executed: {backend_counts.get('crossbar', 0)}")
    print(f"[DETECTIONS] Count after NMS: {len(detections)}")
    for det in detections[:5]:
        print(
            f"[DETECTIONS] {det['label']} {det['confidence']:.3f} "
            f"box={det['box_xyxy']}"
        )
    if layer0_compare is not None:
        print(f"[LAYER0] Reference MAE: {layer0_compare['mae']:.6g}")
        print(f"[LAYER0] Reference max abs error: {layer0_compare['max_abs']:.6g}")
    print(f"[SAVE] Execution trace: {save_dir / 'execution_trace.csv'}")
    print(f"[SAVE] Hardware summary: {hardware_summary_path}")
    if layer0_compare_path is not None:
        print(f"[SAVE] Layer0 comparison: {layer0_compare_path}")
    print(f"[SAVE] Final raw output: {save_dir / 'final_output_int8.npy'}")
    print(f"[SAVE] Final float output: {save_dir / 'final_output_float.npy'}")
    print(f"[SAVE] Detected image: {annotated_path}")
    print(f"[SAVE] Detections JSON: {detections_json_path}")
    print(f"[SAVE] Detections CSV: {detections_csv_path}")
    print(f"[TIME] Total model run time: {elapsed:.3f} seconds")
    print(f"[TIME] Total model run time: {elapsed / 60:.2f} minutes\n")


if __name__ == "__main__":
    main()
