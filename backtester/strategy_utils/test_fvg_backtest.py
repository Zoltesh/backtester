import asyncio
from pathlib import Path
from typing import Dict

import polars as pl

from data_loader.data_loader import DataLoader
from backtester.strategy_utils.fvg_detector import FairValueGapDetector
from backtester.backtest import Backtest
from backtester.strategy_utils.fvg_strategy import FVGStrategy


async def load_sample_data() -> pl.DataFrame | Dict[str, pl.DataFrame]:
    base_dir = Path("data/ohlcv")
    loader = DataLoader(base_dir)
    df = await loader.load_ohlcv_between_dates(
        symbols=["SOL-USDC"],
        timeframes=["1m", "5m", "15m", "30m", "1h", "6h"],
        start_date="2025-01-01",
        end_date="2025-01-31",
    )
    return df


def main() -> None:
    df = asyncio.run(load_sample_data())
    if df is None:
        raise SystemExit("No data returned from DataLoader")

    # Split into base 1m and HTFs
    if isinstance(df, dict):
        base_df = df.get("1m")
        if base_df is None:
            raise SystemExit("Missing 1m base timeframe in returned data")
        htf_map = {tf: d for tf, d in df.items() if tf != "1m"}
    else:
        if "timeframe" in df.columns:
            base_df = df.filter(pl.col("timeframe") == "1m")
            unique_tfs = df.select(pl.col("timeframe").unique()).to_series().to_list()
            htf_map = {tf: df.filter(pl.col("timeframe") == tf) for tf in ["5m","15m","30m","1h","6h"] if tf in unique_tfs}
        else:
            base_df = df
            htf_map = {}

    # Sort chronologically and normalize timestamp to epoch ms for stats
    if "timestamp" in base_df.columns:
        base_df = base_df.sort("timestamp")
        ts_dtype = base_df.schema.get("timestamp")
        if ts_dtype == pl.Datetime:
            base_df = base_df.with_columns(pl.col("timestamp").dt.epoch("ms").alias("timestamp"))
        elif ts_dtype == pl.Date:
            base_df = base_df.with_columns(pl.col("timestamp").cast(pl.Datetime).dt.epoch("ms").alias("timestamp"))
        elif ts_dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
            base_df = base_df.with_columns(pl.col("timestamp").cast(pl.Int64).alias("timestamp"))
        elif ts_dtype == pl.Utf8:
            base_df = base_df.with_columns(
                pl.col("timestamp").str.strptime(pl.Datetime, strict=False, utc=True).dt.epoch("ms").alias("timestamp")
            )

    # Prepare Backtest OHLC
    bt_df = base_df
    # Normalize OHLC capitalization for engine
    rename_map = {k: k.capitalize() for k in ("open","high","low","close","volume") if k in bt_df.columns}
    if rename_map:
        bt_df = bt_df.rename(rename_map)
    required_ohlc = ["Open","High","Low","Close"]
    missing_ohlc = [c for c in required_ohlc if c not in bt_df.columns]
    if missing_ohlc:
        raise SystemExit(f"Base DF missing required OHLC columns: {missing_ohlc}")

    # Prepare Detector DF (lowercase ohlcv, volume optional -> fallback to 1.0)
    det_cols = {}
    for src, dst in [("High","high"),("Low","low"),("Close","close"),("Volume","volume")]:
        if src in bt_df.columns:
            det_cols[dst] = bt_df.get_column(src)
    if "volume" not in det_cols:
        det_cols["volume"] = pl.Series(name="volume", values=[1.0] * bt_df.height)
    det_df = pl.DataFrame(det_cols)

    # Build HTF map for detector (need 'close' column only)
    clean_htf: Dict[str, pl.DataFrame] = {}
    for tf, hdf in htf_map.items():
        cols = hdf.columns
        if "Close" in cols:
            clean_htf[tf] = hdf.select(["Close"]).rename({"Close":"close"})
        elif "close" in cols:
            clean_htf[tf] = hdf.select(["close"])  # already lowercase

    # Detect FVGs with 0.2% threshold and EMA=20
    detector = FairValueGapDetector(det_df, threshold_percent=0.2, htf_dataframes=clean_htf or None, ema_length=20)
    fvgs = detector.detect_fvg()
    print(f"Detected {len(fvgs)} FVGs")

    # Run backtest with precomputed FVGs
    bt = Backtest(
        data=bt_df.select([c for c in ["timestamp","Open","High","Low","Close"] if c in bt_df.columns]),
        strategy=FVGStrategy,
        cash=10_000.0,
        commission=0.0,
    )
    res = bt.run(
        fvg_threshold=0.2,  # percent
        position_size=0.5,
        max_fvg_age=45,
        profit_target=0.5,
        loss_target=0.01,
        precomputed_fvgs=fvgs,
    )

    # Print full metrics (no pandas)
    print("Polars (backtester) full metrics:")
    for key, value in res.stats.items():
        print(f"{key:30} {value}")
        if key == "_strategy":
            break

    # Save trades for inspection
    out_dir = Path("data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / "trades_SOL-USDC_1m_2025-01.csv"
    res.trades.write_csv(trades_path)
    print(f"\nSaved trades to {trades_path}")


if __name__ == "__main__":
    main()


