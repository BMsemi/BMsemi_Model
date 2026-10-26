"""Small deterministic community model for testing platform integration."""
import math


def run(params):
    value = params.get("value", 2)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("value must be a finite number")
    try:
        value = float(value)
    except OverflowError:
        raise ValueError("value is too large") from None
    result = value * 2
    if not math.isfinite(value) or not math.isfinite(result):
        raise ValueError("value and doubled result must be finite")
    return {"value": result}
