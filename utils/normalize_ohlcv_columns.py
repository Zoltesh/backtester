import polars as pl

def normalize_ohlcv_columns(df: pl.DataFrame) -> pl.DataFrame:
    out = df.rename({k: k.capitalize() for k in ("open", "high", "low", "close", "volume") if k in df.columns})
    if "timestamp" in out.columns:
        out = out.sort("timestamp")
        ts_dtype = out.schema.get("timestamp")
        if ts_dtype == pl.Datetime:
            out = out.with_columns(pl.col("timestamp").dt.epoch("ms").alias("timestamp"))
        elif ts_dtype == pl.Date:
            out = out.with_columns(pl.col("timestamp").cast(pl.Datetime).dt.epoch("ms").alias("timestamp"))
        elif ts_dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
            out = out.with_columns(pl.col("timestamp").cast(pl.Int64).alias("timestamp"))
        elif ts_dtype == pl.Utf8:
            out = out.with_columns(
                pl.col("timestamp").str.strptime(pl.Datetime, strict=False, utc=True).dt.epoch("ms").alias("timestamp")
            )
    for c in ("Open", "High", "Low", "Close"):
        if c not in out.columns:
            raise ValueError(f"Missing required column for PL backtest: {c}")
    return out