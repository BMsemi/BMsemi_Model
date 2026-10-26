"""Shared-PE TCAM retrieval and XForge YOLO integration for ZCU106."""

from .pe_tcam import (
    DualPETcamSearch,
    NumpyMvmEngine,
    TcamEntry,
    TcamMatch,
    encode_query_rows,
    encode_tcam_bank,
)
from .legacy_emulator import LegacyNeuromorphicEmulatorBackend
from .hybrid_backend import EarlyLayerMappedHardwareBackend

__all__ = [
    "DualPETcamSearch",
    "EarlyLayerMappedHardwareBackend",
    "LegacyNeuromorphicEmulatorBackend",
    "NumpyMvmEngine",
    "TcamEntry",
    "TcamMatch",
    "encode_query_rows",
    "encode_tcam_bank",
]
