import asyncio
from pathlib import Path
from typing import Dict

import polars as pl
from time import perf_counter
import math

from data_loader.data_loader import DataLoader
from backtester.strategy_utils.fvg_detector import FairValueGapDetector
from backtester.backtest import Backtest as PLBacktest
from backtester.strategy_utils.fvg_strategy import FVGStrategy as PLFVGStrategy

# Traditional backtesting.py
from backtesting import Backtest as PDBacktest
from backtester.strategy_utils.fvg_strategy_pd import FVGStrategyPD


async def load_sample_data() -> pl.DataFrame | Dict[str, pl.DataFrame]:
    base_dir = Path("data/ohlcv")
    loader = DataLoader(base_dir)
    df = await loader.load_ohlcv_between_dates(
        symbols=["SOL-USDC"],
        timeframes=["1m", "5m", "15m", "30m", "1h", "6h"],
        start_date="2024-01-01",
        end_date="2025-07-31",
    )
    return df


def to_pd_ohlc(df: pl.DataFrame):
    import pandas as pd  # local import to avoid global dependency
    # Normalize and set DateTimeIndex based on epoch ms timestamp if present
    rename = {k: k.capitalize() for k in ("open","high","low","close","volume") if k in df.columns}
    df2 = df.rename(rename)
    cols = [c for c in ["timestamp","Open","High","Low","Close","Volume"] if c in df2.columns]
    pdf = df2.select(cols).to_pandas()
    if "timestamp" in pdf.columns:
        pdf.index = pd.to_datetime(pdf["timestamp"], unit="ms", utc=True)
        pdf = pdf.drop(columns=["timestamp"])  # backtesting.py wants only OHLCV columns
    else:
        pdf.index = pd.date_range("2025-01-01", periods=len(pdf), freq="T", tz="UTC")
    return pdf


def main() -> None:
    df = asyncio.run(load_sample_data())
    if isinstance(df, dict):
        base_pl = df["1m"]
        htf_map = {tf: d for tf, d in df.items() if tf != "1m"}
    else:
        if "timeframe" in df.columns:
            base_pl = df.filter(pl.col("timeframe") == "1m")
            uniq = df.select(pl.col("timeframe").unique()).to_series().to_list()
            htf_map = {tf: df.filter(pl.col("timeframe") == tf) for tf in ["5m","15m","30m","1h","6h"] if tf in uniq}
        else:
            base_pl, htf_map = df, {}

    # Normalize capitalization for PL engine and ensure epoch ms timestamps
    base_pl_bt = base_pl.rename({k: k.capitalize() for k in ("open","high","low","close","volume") if k in base_pl.columns})
    if "timestamp" in base_pl_bt.columns:
        base_pl_bt = base_pl_bt.sort("timestamp")
        ts_dtype = base_pl_bt.schema.get("timestamp")
        if ts_dtype == pl.Datetime:
            base_pl_bt = base_pl_bt.with_columns(pl.col("timestamp").dt.epoch("ms").alias("timestamp"))
        elif ts_dtype == pl.Date:
            base_pl_bt = base_pl_bt.with_columns(pl.col("timestamp").cast(pl.Datetime).dt.epoch("ms").alias("timestamp"))
        elif ts_dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
            base_pl_bt = base_pl_bt.with_columns(pl.col("timestamp").cast(pl.Int64).alias("timestamp"))
        elif ts_dtype == pl.Utf8:
            base_pl_bt = base_pl_bt.with_columns(
                pl.col("timestamp").str.strptime(pl.Datetime, strict=False, utc=True).dt.epoch("ms").alias("timestamp")
            )
    for c in ("Open","High","Low","Close"):
        if c not in base_pl_bt.columns:
            raise SystemExit(f"Missing required column for PL backtest: {c}")

    # Detector DF (lowercase OHLCV)
    det_cols = {}
    for src, dst in [("High","high"),("Low","low"),("Close","close"),("Volume","volume")]:
        if src in base_pl_bt.columns:
            det_cols[dst] = base_pl_bt.get_column(src)
    if "volume" not in det_cols:
        det_cols["volume"] = pl.Series(name="volume", values=[1.0] * base_pl_bt.height)
    det_pl = pl.DataFrame(det_cols)

    # HTF map for detector
    clean_htf: Dict[str, pl.DataFrame] = {}
    for tf, hdf in htf_map.items():
        if "Close" in hdf.columns:
            clean_htf[tf] = hdf.select(["Close"]).rename({"Close":"close"})
        elif "close" in hdf.columns:
            clean_htf[tf] = hdf.select(["close"])  # already lowercase

    # Detect FVGs identically
    fvgs = FairValueGapDetector(det_pl, threshold_percent=0.2, htf_dataframes=clean_htf or None, ema_length=20).detect_fvg()

    # 1) Run polars-native engine
    pl_bt = PLBacktest(
        data=base_pl_bt.select([c for c in ["timestamp","Open","High","Low","Close"] if c in base_pl_bt.columns]),
        strategy=PLFVGStrategy,
        cash=10_000.0,
        commission=0.002,
    )
    t0 = perf_counter()
    pl_res = pl_bt.run(
        fvg_threshold=0.01,
        position_size=0.3,
        max_fvg_age=2,
        profit_target=0.1,
        loss_target=0.1,
        precomputed_fvgs=fvgs,
    )
    pl_time = perf_counter() - t0

    # 2) Run traditional backtesting.py engine
    pd_df = to_pd_ohlc(base_pl_bt)
    pd_bt = PDBacktest(pd_df, FVGStrategyPD, cash=10_000.0, commission=0.002, exclusive_orders=True, finalize_trades=True)
    t1 = perf_counter()
    pd_stats = pd_bt.run(
        fvg_threshold=0.01,
        position_size=0.3,
        max_fvg_age=2,
        profit_target=0.1,
        loss_target=0.1,
        precomputed_fvgs=fvgs,
    )
    pd_time = perf_counter() - t1

    # Summary for ALL common metrics (exclude Start/End/Duration and internals)
    print("\nSummary (PL vs PD):")
    excluded = {"Start", "End", "Duration", "_strategy", "_equity_curve"}
    def is_excluded(k: str) -> bool:
        kl = str(k).lower()
        return (
            (k in excluded)
            or ("duration" in kl)
            or ("trade" in kl)
            or str(k).startswith("_")
        )

    # Preserve PL stats order, only include keys present in PD stats too
    for key in [k for k in pl_res.stats.keys() if (k in pd_stats) and not is_excluded(k)]:
        pl_val = pl_res.stats.get(key)
        pd_val = pd_stats.get(key)
        try:
            num = float(pl_val)
            denom = float(pd_val)
            if denom != 0.0 and math.isfinite(denom) and math.isfinite(num):
                pct = 100.0 * (num - denom) / denom
                print(f"{key:30} PL={pl_val:<20} PD={pd_val:<20} diff={pct:+.4f}%")
            else:
                print(f"{key:30} PL={pl_val:<20} PD={pd_val:<20} diff=n/a")
        except Exception:
            print(f"{key:30} PL={pl_val} PD={pd_val} diff=n/a")

    # Timing summary
    print("\nTiming:")
    print(f"Polars run time:      {pl_time:.4f} s")
    print(f"Pandas run time:      {pd_time:.4f} s")
    speed = (pd_time / pl_time) if pl_time > 0 else float('inf')
    print(f"Polars Speed:         {speed:.2f}x")

    # Still print a compact header from each for context
    print("\nPolars (backtester) metrics:")
    for k, v in pl_res.stats.items():
        print(f"{k:30} {v}")
        if k == "_strategy":
            break

    print("\nbacktesting.py metrics:")
    for k, v in pd_stats.items():
        print(f"{str(k):30} {v}")
        if str(k) == "_strategy":
            break

    # Write outputs: FVGs (flattened ema_htf_bias) and polars trades
    out_dir = Path("data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine suffix from symbol/timeframe if present
    def _first_str(df: pl.DataFrame, col: str) -> str | None:
        if col in df.columns and df.height:
            try:
                return str(df.select(pl.col(col).first()).item())
            except Exception:
                try:
                    return str(df.get_column(col)[0])
                except Exception:
                    return None
        return None

    symbol = _first_str(base_pl_bt, "symbol")
    tf = _first_str(base_pl_bt, "timeframe")
    suffix = ""
    parts = [p for p in [symbol, tf] if p]
    if parts:
        suffix = "_" + "_".join(parts)

    # Flatten FVGs ema_htf_bias into columns
    all_bias_tfs: set[str] = set()
    for f in fvgs:
        bias = f.get("ema_htf_bias")
        if isinstance(bias, dict):
            all_bias_tfs.update(bias.keys())

    flat_fvgs: list[dict] = []
    has_ts = "timestamp" in base_pl_bt.columns
    ts_col = base_pl_bt.get_column("timestamp") if has_ts else None
    for f in fvgs:
        rec = dict(f)
        bias = rec.pop("ema_htf_bias", None) or {}
        for tf_key in all_bias_tfs:
            rec[f"ema_htf_{tf_key}"] = bias.get(tf_key)
        if has_ts:
            idx = int(rec.get("bar_index", -1))
            if 0 <= idx < base_pl_bt.height:
                rec["timestamp"] = int(ts_col[idx])  # epoch ms
        flat_fvgs.append(rec)

    fvgs_df = pl.DataFrame(flat_fvgs) if flat_fvgs else pl.DataFrame({})
    fvgs_path = out_dir / f"fvgs{suffix}.csv"
    fvgs_df.write_csv(fvgs_path)

    trades_path = out_dir / f"trades_pl{suffix}.csv"
    pl_res.trades.write_csv(trades_path)

if __name__ == "__main__":
    main()


