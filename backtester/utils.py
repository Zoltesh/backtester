from __future__ import annotations

import polars as pl


def ensure_ohlcv(df: pl.DataFrame) -> pl.DataFrame:
    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(f"Missing required OHLC columns: {sorted(missing)}")
    return df


