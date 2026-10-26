from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the packaged X2 YOLO prototype on a ZCU106/PYNQ image."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).resolve().parent / "bus.jpg",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "board_output",
    )
    parser.add_argument(
        "--preprocess",
        choices=("resize", "letterbox"),
        default="resize",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    from x2_zcu106.x2compiler.run_hybrid_yolo import main as run_hybrid

    sys.argv = [
        "run_hybrid_yolo",
        str(root / "model" / "yolo26n_int8.tflite"),
        str(root / "artifacts"),
        str(args.image),
        "--backend",
        "pynq",
        "--bitstream",
        str(root / "overlay" / "zcu106_axi_x2_mvm.bit"),
        "--offload",
        "all",
        "--preprocess",
        args.preprocess,
        "--output-dir",
        str(args.output),
    ]
    run_hybrid()


if __name__ == "__main__":
    main()
