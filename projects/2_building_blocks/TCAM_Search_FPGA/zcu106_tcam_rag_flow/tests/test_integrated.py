from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for candidate in (ROOT, ROOT / "python" / "tcam_rag"):
    if (candidate / "integrated").is_dir():
        sys.path.insert(0, str(candidate))
        break

from integrated.arbiter import SharedPEArbiter
from integrated.cache import PersistentRagCache
from integrated.fingerprint import ImageIdentity
from integrated.hybrid_backend import (
    EarlyLayerMappedHardwareBackend,
    Layer0HardwareRemainingNeuromorphicBackend,
)
from integrated.legacy_emulator import LegacyNeuromorphicEmulatorBackend
from integrated.pe_tcam import (
    DualPETcamSearch,
    FULL_CARE,
    NumpyMvmEngine,
    TcamEntry,
    encode_query_rows,
    encode_tcam_bank,
)
from integrated.pipeline import IntegratedTcamYoloPipeline


def test_differential_encoding_and_dual_pe_ranking() -> None:
    entries = [
        TcamEntry(0, 0xA5A5, FULL_CARE, 0x1000),
        TcamEntry(1, 0xA5A4, FULL_CARE, 0x1001),
        TcamEntry(2, 0xA5A7, FULL_CARE, 0x1002),
        TcamEntry(16, 0xA5A0, FULL_CARE, 0x2000),
    ]
    engines = (NumpyMvmEngine("pe0"), NumpyMvmEngine("pe1"))
    searcher = DualPETcamSearch(engines)
    searcher.program(entries)
    matches = searcher.search(0xA5A5, top_k=4)

    assert [match.chunk_id for match in matches] == [0x1000, 0x1001, 0x1002, 0x2000]
    assert [match.hamming_distance for match in matches] == [0, 1, 1, 2]
    assert (matches[0].a_current, matches[0].b_current) == (0, 16)
    assert (matches[0].b_minus_a, matches[0].tdc_code) == (16, 32)
    assert engines[0].calls == 2 and engines[1].calls == 2


def test_query_care_and_complementary_cells() -> None:
    entry = TcamEntry(0, 0b001, 0b111, 1)
    matrix = encode_tcam_bank([entry], 0)
    query = encode_query_rows(0b001, 0b011)
    currents = query.astype(np.int32) @ matrix.astype(np.int32)
    assert (int(currents[0]), int(currents[1])) == (0, 2)
    for bit in range(3):
        physical = matrix[2 * bit : 2 * bit + 2, 0:2]
        assert int(physical.sum()) == 2


def test_digest_guard_rejects_short_fingerprint_collision() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        searcher = DualPETcamSearch(
            (NumpyMvmEngine("pe0"), NumpyMvmEngine("pe1"))
        )
        cache = PersistentRagCache(Path(temporary), searcher)
        cat = ImageIdentity(visual_key=0xA5A5, sha256="1" * 64, byte_count=100)
        cat_and_dog = ImageIdentity(
            visual_key=0xA5A5,
            sha256="2" * 64,
            byte_count=120,
        )
        cache.insert(cat, {"detections": ["cat"]}, image_name="cat.jpg")
        hit = cache.lookup(cat)
        miss = cache.lookup(cat_and_dog)
        assert hit.hit and hit.payload == {"detections": ["cat"]}
        assert not miss.hit and miss.reason == "digest_mismatch"


@dataclass(frozen=True)
class _Tile:
    tile_id: int = 0
    k_start: int = 0
    n_start: int = 0
    rows: int = 2
    cols: int = 2
    weight_key: str = "w"


class _Program:
    def __init__(self, op_index: int, n_dim: int = 2) -> None:
        self.name = f"op_{op_index}"
        self.m_dim = 4
        self.k_dim = 2
        self.n_dim = int(n_dim)
        self.tiles = [
            _Tile(
                tile_id=0,
                k_start=0,
                n_start=0,
                rows=2,
                cols=int(n_dim),
                weight_key="w",
            )
        ]
        weights = np.arange(1, (2 * int(n_dim)) + 1, dtype=np.int16).reshape(2, int(n_dim))
        self.constants = {"w": weights.astype(np.int8)}
        self.metadata = {"source": {"op_index": op_index}}


class _Hardware:
    max_batch = 2

    def __init__(self) -> None:
        self.calls = 0
        self.pe_count = 4

    def run_batch(self, vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
        self.calls += 1
        return vectors.astype(np.int32) @ weights.astype(np.int32)


class _EngineHardware:
    max_batch = 2

    def __init__(self, engine_id: int) -> None:
        self.engine_id = int(engine_id)
        self.calls = 0
        self.pe_count = 1

    def run_batch(self, vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
        self.calls += 1
        return vectors.astype(np.int32) @ weights.astype(np.int32)


class _Neuromorphic:
    def __init__(self) -> None:
        self.calls = 0

    def run_program(self, program: _Program, inputs: np.ndarray) -> np.ndarray:
        self.calls += 1
        return inputs.astype(np.int32) @ program.constants["w"].astype(np.int32)


def test_layer_dispatch_policy() -> None:
    hardware = _Hardware()
    neuromorphic = _Neuromorphic()
    arbiter = SharedPEArbiter()
    backend = Layer0HardwareRemainingNeuromorphicBackend(
        hardware,
        neuromorphic,
        layer0_op_index=2,
        arbiter=arbiter,
    )
    inputs = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int8)
    reference = inputs.astype(np.int32) @ np.asarray([[1, 2], [3, 4]], dtype=np.int32)
    np.testing.assert_array_equal(backend.run_program(_Program(2), inputs), reference)
    np.testing.assert_array_equal(backend.run_program(_Program(6), inputs), reference)
    assert hardware.calls == 2
    assert neuromorphic.calls == 1
    assert backend.dispatch_counts == {
        "fpga_4_pe_layer0": 1,
        "neuromorphic_emulator": 1,
    }
    assert arbiter.snapshot().owner is None


def test_early_layer_per_engine_mapping_and_embedding_storage() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        engines = {
            2: _EngineHardware(2),
            3: _EngineHardware(3),
            4: _EngineHardware(4),
            5: _EngineHardware(5),
        }
        neuromorphic = LegacyNeuromorphicEmulatorBackend(quiet=True)
        backend = EarlyLayerMappedHardwareBackend(
            engines,
            neuromorphic,
            fpga_op_engine_map={2: 2, 6: 3, 9: 4, 13: 5},
            op_layer_map={2: 0, 6: 1, 9: 2, 13: 3},
            arbiter=SharedPEArbiter(),
        )
        backend.set_embedding_directory(Path(temporary) / "fpga_layer_embeddings")
        inputs = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int8)
        program_144 = _Program(2, n_dim=144)
        program_32 = _Program(6, n_dim=32)
        program_cpu = _Program(16, n_dim=2)
        reference_144 = inputs.astype(np.int32) @ program_144.constants["w"].astype(np.int32)
        reference_32 = inputs.astype(np.int32) @ program_32.constants["w"].astype(np.int32)
        reference_cpu = inputs.astype(np.int32) @ program_cpu.constants["w"].astype(np.int32)

        np.testing.assert_array_equal(backend.run_program(program_144, inputs), reference_144)
        np.testing.assert_array_equal(backend.run_program(program_32, inputs), reference_32)
        np.testing.assert_array_equal(backend.run_program(program_cpu, inputs), reference_cpu)

        assert engines[2].calls == 8
        assert engines[3].calls == 6
        assert engines[4].calls == 4
        assert engines[5].calls == 4
        diagnostics = backend.diagnostics()
        assert diagnostics["dispatch_counts"] == {
            "fpga_channel_tiled_layer_0": 1,
            "fpga_channel_tiled_layer_1": 1,
            "neuromorphic_emulator": 1,
        }
        assert diagnostics["fpga_op_engine_map"] == {
            "2": 2,
            "6": 3,
            "9": 4,
            "13": 5,
        }
        embeddings = backend.embedding_manifest()
        assert len(embeddings) == 2
        for record in embeddings:
            tensor_path = Path(record["tensor_file"])
            meta_path = Path(record["metadata_file"])
            assert tensor_path.exists()
            assert meta_path.exists()
        assert embeddings[0]["channel_tile_schedule"][0]["oc_start"] == 0
        assert embeddings[0]["channel_tile_schedule"][0]["oc_end"] == 16
        assert embeddings[0]["channel_tile_schedule"][4]["time_step"] == 1
        assert embeddings[0]["channel_tile_schedule"][8]["time_step"] == 2
        assert embeddings[0]["channel_tile_schedule"][8]["engine"] == 2
        assert embeddings[0]["channel_tile_schedule"][8]["oc_start"] == 128
        assert embeddings[0]["channel_tile_schedule"][8]["oc_end"] == 144
        np.testing.assert_array_equal(np.load(Path(embeddings[0]["tensor_file"])), reference_144)
        np.testing.assert_array_equal(np.load(Path(embeddings[1]["tensor_file"])), reference_32)
        np.testing.assert_array_equal(neuromorphic.read_layer_tensor(0), reference_144)
        np.testing.assert_array_equal(neuromorphic.read_layer_tensor(1), reference_32)


def test_legacy_emulator_l2_layer_tensor_flow() -> None:
    backend = LegacyNeuromorphicEmulatorBackend(quiet=True)
    program = _Program(6)
    inputs = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int8)
    reference = inputs.astype(np.int32) @ program.constants["w"].astype(np.int32)

    np.testing.assert_array_equal(backend.run_program(program, inputs), reference)
    np.testing.assert_array_equal(backend.read_layer_tensor(6), reference)

    stats = backend.diagnostics()
    assert stats["engine"] == "LegacyNeuromorphicCore"
    assert stats["dataflow"] == [
        "input_buffer",
        "compute_buffer",
        "crossbar_array",
        "output_buffer",
        "L2_Buffer",
    ]
    assert stats["l2_stats"]["write_count"] == 2
    assert stats["layer_outputs"]["6"] == [4, 2]


def test_end_to_end_cache_miss_then_hit() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        image = root / "frame.pgm"
        pixels = bytes(range(20))
        image.write_bytes(b"P5\n5 4\n255\n" + pixels)
        searcher = DualPETcamSearch(
            (NumpyMvmEngine("pe0"), NumpyMvmEngine("pe1"))
        )
        cache = PersistentRagCache(root / "cache", searcher)
        inference_calls = []

        def inference(path: Path, output: Path) -> dict[str, object]:
            inference_calls.append(path)
            output.mkdir(parents=True, exist_ok=True)
            return {"detections": [{"class_name": "cat"}]}

        pipeline = IntegratedTcamYoloPipeline(cache, inference, root / "output")
        first = pipeline.run(image)
        second = pipeline.run(image)
        assert first["path"] == "tcam_miss_then_yolo"
        assert second["path"] == "tcam_cache_hit"
        assert not first["yolo_skipped"] and second["yolo_skipped"]
        assert (
            first["flow_blocks"]["constants_manifest"]["block_name"]
            == "Constants & Manifest + TCAM Cache Index"
        )
        assert (
            second["flow_blocks"]["hybrid_runtime_scheduler"]["decision"]
            == "TCAM HIT: cached detections, skip MLIR"
        )
        assert len(inference_calls) == 1
        index = json.loads((root / "cache" / "index.json").read_text())
        assert index["tcam_cache_index"]["physical_shape_per_pe"] == [32, 32]
        assert index["tcam_cache_index"]["total_engine_count"] == 6
        assert index["engine_partition"]["Engine 2"].startswith("YOLO output-channel")
        assert index["engine_partition"]["Engine 5"].startswith("YOLO output-channel")
        assert index["spatial_temporal_channel_policy"]["channel_tile_size"] == 16
        assert index["hybrid_runtime_contract"]["tcam_capacity_path"].startswith("extra")


if __name__ == "__main__":
    test_differential_encoding_and_dual_pe_ranking()
    test_query_care_and_complementary_cells()
    test_digest_guard_rejects_short_fingerprint_collision()
    test_layer_dispatch_policy()
    test_early_layer_per_engine_mapping_and_embedding_storage()
    test_legacy_emulator_l2_layer_tensor_flow()
    test_end_to_end_cache_miss_then_hit()
    print("PASS")
