from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[2]


def _copy_python_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble the deployable ZCU106/PYNQ hybrid runtime."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "x2_zcu106" / "deploy" / "zcu106_hybrid_board",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=(
            ROOT
            / "x2_zcu106"
            / "compiler_artifacts"
            / "yolo26n_hybrid_calibrated"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=(
            ROOT
            / "YOLO26n-TFLite-FPGA-Partner-Project"
            / "yolo26n_int8.tflite"
        ),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=(
            ROOT / "YOLO26n-TFLite-FPGA-Partner-Project" / "bus.jpg"
        ),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    deploy_root = (ROOT / "x2_zcu106" / "deploy").resolve()
    if deploy_root not in output.parents:
        raise ValueError(f"Package output must remain under {deploy_root}.")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    overlay = output / "overlay"
    overlay.mkdir()
    for suffix in ("bit", "hwh"):
        source = (
            ROOT / "x2_zcu106" / "deploy" / f"zcu106_axi_x2_mvm.{suffix}"
        )
        shutil.copy2(source, overlay / source.name)
    shutil.copytree(args.artifacts, output / "artifacts")
    model_directory = output / "model"
    model_directory.mkdir()
    shutil.copy2(args.model, model_directory / "yolo26n_int8.tflite")
    shutil.copy2(args.image, output / "bus.jpg")

    package_root = output / "x2_zcu106"
    package_root.mkdir()
    shutil.copy2(ROOT / "x2_zcu106" / "__init__.py", package_root)
    _copy_python_tree(
        ROOT / "x2_zcu106" / "x2compiler",
        package_root / "x2compiler",
    )
    _copy_python_tree(
        ROOT / "x2_zcu106" / "runtime",
        package_root / "runtime",
    )
    for filename in (
        "run_on_board.py",
        "run_first_conv_tile0_tdc.py",
        "requirements-board.txt",
        "README.md",
    ):
        shutil.copy2(ROOT / "x2_zcu106" / "board" / filename, output / filename)

    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "target": "ZCU106",
        "accelerator_version_minimum": "0x00020024",
        "pe_array_count": 2,
        "max_scheduled_batch": 64,
        "compiler_artifacts": "yolo26n_hybrid_calibrated",
        "files": {
            str(path.relative_to(output)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        },
    }
    (output / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    archive = output.with_suffix(".zip")
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(
                    path,
                    Path(output.name) / path.relative_to(output),
                )
    print(json.dumps({
        "package": str(output),
        "archive": str(archive),
        "file_count": len(manifest["files"]) + 1,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
    }, indent=2))


if __name__ == "__main__":
    main()
