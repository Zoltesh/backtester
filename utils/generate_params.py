import math
from typing import Dict, Tuple
import random

ParamSpec = Tuple[float, float, float]  # (min, max, increment)

def _round_to_increment(val: float, step: float) -> float:
    # Round to a sensible number of decimals based on step (e.g., 0.01 -> 2 dp)
    if step.is_integer():
        return float(round(val))
    # count decimals from the string to avoid FP weirdness
    s = f"{step:.16f}".rstrip("0")
    decimals = max(0, len(s.split(".")[1]) if "." in s else 0)
    return round(val, decimals + 2)  # small buffer to be safe

def generate_random_params(
    param_ranges: Dict[str, ParamSpec],
    n_params: int,
    seed: int | None = None,
) -> list[Dict[str, float]]:
    rng = random.Random(seed)

    # Precompute (lo, step, steps)
    specs: Dict[str, Tuple[float, float, int]] = {}
    for name, (lo, hi, step) in param_ranges.items():
        if step <= 0:
            raise ValueError(f"{name}: increment must be > 0")
        steps = math.floor((hi - lo) / step)
        if steps < 0:
            raise ValueError(f"{name}: max must be >= min")
        specs[name] = (lo, step, steps)

    out: list[Dict[str, float]] = []
    for _ in range(n_params):
        params: Dict[str, float] = {}
        for name, (lo, step, steps) in specs.items():
            k = rng.randint(0, steps)  # inclusive
            val = lo + k * step
            val = _round_to_increment(val, float(step))
            params[name] = val
        out.append(params)
    return out