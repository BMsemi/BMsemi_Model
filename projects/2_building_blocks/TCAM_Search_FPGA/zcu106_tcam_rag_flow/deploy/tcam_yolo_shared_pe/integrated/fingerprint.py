from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ImageIdentity:
    visual_key: int
    sha256: str
    byte_count: int


def visual_fingerprint(gray: np.ndarray) -> int:
    """Return a 16-bit dHash used only for FPGA candidate retrieval."""

    image = np.asarray(gray)
    if image.ndim == 3:
        image = np.rint(
            image[..., 0] * 0.114
            + image[..., 1] * 0.587
            + image[..., 2] * 0.299
        )
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Fingerprint input must be a non-empty image array.")
    try:
        import cv2

        reduced = cv2.resize(
            image.astype(np.uint8),
            (5, 4),
            interpolation=cv2.INTER_AREA,
        )
    except ImportError:
        from PIL import Image

        reduced = np.asarray(
            Image.fromarray(image.astype(np.uint8)).resize((5, 4))
        )
    comparisons = reduced[:, 1:] >= reduced[:, :-1]
    value = 0
    for bit, enabled in enumerate(comparisons.reshape(-1)):
        value |= int(bool(enabled)) << bit
    return value


def identify_image(path: str | Path) -> ImageIdentity:
    source = Path(path)
    payload = source.read_bytes()
    try:
        import cv2

        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode image {source}.")
    except ImportError:
        from PIL import Image

        image = np.asarray(Image.open(source).convert("RGB"))[:, :, ::-1]
    return ImageIdentity(
        visual_key=visual_fingerprint(image),
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_count=len(payload),
    )
