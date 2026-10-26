from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))

from axi_x2_mvm_driver import AxiX2MvmPynq, numpy_mvm_reference  # noqa: E402


def main() -> None:
    bit_path = Path(__file__).with_name("zcu106_axi_x2_mvm.bit")
    accel = AxiX2MvmPynq(bit_path)

    vector = np.array([1, 2, 3], dtype=np.int8)
    weights = np.array([[4, 5], [6, 7], [8, 9]], dtype=np.int8)
    expected = numpy_mvm_reference(vector, weights)
    result = accel.run_mvm(vector, weights)

    print(f"result={result.tolist()}")
    print(f"expected={expected.tolist()}")
    print(f"stats={accel.stats()}")
    if not np.array_equal(result, expected):
        raise SystemExit("FAIL: hardware result mismatch")
    print("PASS: zcu106_axi_x2_mvm hardware smoke")


if __name__ == "__main__":
    main()
