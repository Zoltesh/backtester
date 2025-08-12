import asyncio
from data_loader.data_loader import DataLoader
import polars as pl
from pathlib import Path
from typing import Dict, Any, List
from backtester.strategy_utils.fvg_detector import FairValueGapDetector

async def load_sample_data() -> pl.DataFrame:
    if DataLoader is not None:
        try:
            base_dir = Path("data/ohlcv")
            loader = DataLoader(base_dir)
            df = await loader.load_ohlcv_between_dates(
                symbols=["SOL-USDC"],
                timeframes=["1m", "5m", "15m", "30m", "1h", "6h"],
                start_date="2025-01-01",
                end_date="2025-01-31",
            )
            return df
        except Exception:
            pass


def _split_base_and_htf(
    data: Any,
    base_timeframe: str = "1m",
) -> tuple[pl.DataFrame, Dict[str, pl.DataFrame]]:
    """
    Accepts either a dict[timeframe -> pl.DataFrame] or a single pl.DataFrame
    with a `timeframe` column, and returns the 1m base dataframe and a mapping
    of higher timeframe -> dataframe.
    """
    if isinstance(data, dict):
        base_df = data.get(base_timeframe)
        if base_df is None:
            raise ValueError(f"Base timeframe '{base_timeframe}' not found in data dict")
        htf = {tf: df for tf, df in data.items() if tf != base_timeframe}
        return base_df, htf

    if isinstance(data, pl.DataFrame):
        if "timeframe" in data.columns:
            base_df = data.filter(pl.col("timeframe") == base_timeframe)
            htf: Dict[str, pl.DataFrame] = {}
            for tf in data.select(pl.col("timeframe").unique()).to_series().to_list():
                if tf == base_timeframe:
                    continue
                htf[tf] = data.filter(pl.col("timeframe") == tf)
            return base_df, htf
        # Assume single-timeframe dataframe is base
        return data, {}

    raise TypeError("Unsupported data type for timeframe splitting")


def _flatten_bias_rows(rows: List[Dict[str, Any]], expected_tfs: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        bias = r.get("ema_htf_bias") or {}
        flat = {k: v for k, v in r.items() if k != "ema_htf_bias"}
        for tf in expected_tfs:
            flat[f"ema_htf_{tf}"] = bias.get(tf)
        out.append(flat)
    return out


if __name__ == "__main__":
    df = asyncio.run(load_sample_data())

    if df is None or (isinstance(df, pl.DataFrame) and df.height == 0):
        raise SystemExit("No data loaded; cannot run FVG detection.")

    base_df, htf_map = _split_base_and_htf(df, base_timeframe="1m")

    # Ensure base DF is sorted by timestamp if present
    if "timestamp" in base_df.columns:
        base_df = base_df.sort("timestamp")

    # Select only necessary columns for detector (keep timestamp if present for future analysis)
    needed = [c for c in ("high", "low", "close", "volume") if c in base_df.columns]
    missing = set(["high", "low", "close", "volume"]) - set(needed)
    if missing:
        raise SystemExit(f"Base dataframe missing required columns: {sorted(missing)}")
    base_det_df = base_df.select(needed)

    # HTF frames: keep at least close; sort by timestamp if present
    clean_htf: Dict[str, pl.DataFrame] = {}
    for tf, hdf in htf_map.items():
        if not isinstance(hdf, pl.DataFrame) or "close" not in hdf.columns:
            continue
        if "timestamp" in hdf.columns:
            hdf = hdf.sort("timestamp")
        clean_htf[tf] = hdf.select([c for c in hdf.columns if c == "close"])  # only close is needed

    detector = FairValueGapDetector(
        data=base_det_df,
        threshold_percent=0.2,
        htf_dataframes=clean_htf or None,
        ema_length=20,
    )

    results = detector.detect_fvg()

    # Flatten EMA bias dict to columns for parquet friendliness
    expected_tfs = [tf for tf in ("5m", "15m", "30m", "1h", "6h") if tf in clean_htf]
    flat_results = _flatten_bias_rows(results, expected_tfs)

    if not flat_results:
        print("No FVGs detected.")
        raise SystemExit(0)

    out_dir = Path("data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fvg_SOL-USDC_1m_2025-01-01_2025-01-31.parquet"

    print(df.shape)
    pl.DataFrame(flat_results).write_parquet(str(out_path))
    print(f"Saved {len(flat_results)} FVGs to {out_path}")