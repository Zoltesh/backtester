from __future__ import annotations

import numpy as np
import polars as pl


def sma(series: pl.Series, window: int) -> np.ndarray:
    """Simple moving average returning numpy array to match Strategy.I expectations."""
    if window <= 0:
        raise ValueError("window must be positive")
    # Compute rolling mean as Float64 and preserve nulls as NaN without Python loops
    # Casting before to_numpy ensures a numeric dtype; fill nulls to keep np.float64
    s = series.rolling_mean(window).cast(pl.Float64).fill_null(np.nan)
    return s.to_numpy()


def crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    prev = np.roll(a - b, 1)
    prev[0] = 0.0
    now = a - b
    return (prev <= 0) & (now > 0)


def crossunder(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    prev = np.roll(a - b, 1)
    prev[0] = 0.0
    now = a - b
    return (prev >= 0) & (now < 0)


