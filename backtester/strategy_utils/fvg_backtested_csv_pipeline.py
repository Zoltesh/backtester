from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

import polars as pl

# Data loader (same usage and parameters as compare_fvg_backtests.load_sample_data)
from data_loader.data_loader import DataLoader

# Polars-native detector/strategy/engine
from backtester.strategy_utils.fvg_detector import FairValueGapDetector
from backtester.backtest import Backtest
from backtester.strategy_utils.fvg_strategy import FVGStrategy


async def load_sample_data() -> pl.DataFrame | Dict[str, pl.DataFrame]:
    """
    Load OHLCV data exactly as in compare_fvg_backtests.load_sample_data.
    Returns either a dict of timeframes -> DataFrame or a single DataFrame.
    """
    base_dir = Path("data/ohlcv")
    loader = DataLoader(base_dir)
    df = await loader.load_ohlcv_between_dates(
        symbols=["SOL-USDC"],
        timeframes=["1m", "5m", "15m", "30m", "1h", "6h"],
        start_date="2024-01-01",
        end_date="2025-07-31",
    )
    return df


def _normalize_base_for_backtest(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure required columns and normalize timestamp to epoch ms; rename OHLCV to caps."""
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
    # Validate
    for c in ("Open", "High", "Low", "Close"):
        if c not in out.columns:
            raise ValueError(f"Missing required column for PL backtest: {c}")
    return out


def _prepare_detector_inputs(base_pl_bt: pl.DataFrame, htf_map: Dict[str, pl.DataFrame]) -> tuple[pl.DataFrame, Dict[str, pl.DataFrame]]:
    # Detector DF (lowercase OHLCV, volume optional -> fallback to 1.0)
    det_cols: Dict[str, pl.Series] = {}
    for src, dst in [("High", "high"), ("Low", "low"), ("Close", "close"), ("Volume", "volume")]:
        if src in base_pl_bt.columns:
            det_cols[dst] = base_pl_bt.get_column(src)
    if "volume" not in det_cols:
        det_cols["volume"] = pl.Series(name="volume", values=[1.0] * base_pl_bt.height)
    det_pl = pl.DataFrame(det_cols)

    # HTF map for detector (need 'close' column only)
    clean_htf: Dict[str, pl.DataFrame] = {}
    for tf, hdf in htf_map.items():
        if "Close" in hdf.columns:
            clean_htf[tf] = hdf.select(["Close"]).rename({"Close": "close"})
        elif "close" in hdf.columns:
            clean_htf[tf] = hdf.select(["close"])  # already lowercase
    return det_pl, clean_htf


def _flatten_fvgs(fvgs: List[Dict[str, Any]], base_pl_bt: pl.DataFrame) -> pl.DataFrame:
    """Flatten ema_htf_bias dicts and attach fvg timestamp; return DataFrame."""
    # Discover all possible timeframe keys
    all_bias_tfs: set[str] = set()
    for f in fvgs:
        bias = f.get("ema_htf_bias") or f.get("htf_ema_bias")
        if isinstance(bias, dict):
            all_bias_tfs.update(bias.keys())

    has_ts = "timestamp" in base_pl_bt.columns
    ts_col = base_pl_bt.get_column("timestamp") if has_ts else None

    flat_records: list[dict] = []
    for f in fvgs:
        rec = {
            "fvg_bar_index": int(f.get("bar_index", -1)),
            "fvg_is_bull": bool(f.get("is_bull", False)),
            "fvg_max_price": f.get("max_price"),
            "fvg_min_price": f.get("min_price"),
            "fvg_midpoint": f.get("midpoint"),
            "fvg_gap_size_percent": f.get("gap_size_percent"),
            "fvg_displacement_strength": f.get("displacement_strength"),
        }
        bias = f.get("ema_htf_bias") or f.get("htf_ema_bias") or {}
        if isinstance(bias, dict):
            for tf_key in all_bias_tfs:
                rec[f"ema_htf_{tf_key}"] = bias.get(tf_key)
        else:
            for tf_key in all_bias_tfs:
                rec[f"ema_htf_{tf_key}"] = None
        if has_ts:
            idx = rec["fvg_bar_index"]
            if 0 <= idx < base_pl_bt.height:
                rec["fvg_timestamp"] = int(ts_col[idx])
            else:
                rec["fvg_timestamp"] = None
        flat_records.append(rec)

    return pl.DataFrame(flat_records) if flat_records else pl.DataFrame({})


def _build_fvgs_by_bar(
    fvgs: List[Dict[str, Any]], *, fvg_threshold: float
) -> Dict[int, List[Dict[str, Any]]]:
    """Group FVGs by bar index; filter by threshold in percent units (same as strategy)."""
    by_bar: Dict[int, List[Dict[str, Any]]] = {}
    for f in fvgs:
        try:
            bar_idx = int(f.get("bar_index"))
            gap_sz = float(f.get("gap_size_percent", 0.0))
        except Exception:
            continue
        if gap_sz < float(fvg_threshold):
            continue
        by_bar.setdefault(bar_idx, []).append(f)
    return by_bar


def _match_trade_to_fvg(
    *,
    entry_idx: int,
    direction: int,
    base_high: Any,
    base_low: Any,
    fvgs_by_bar: Dict[int, List[Dict[str, Any]]],
    max_fvg_age: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    """
    Replicate FVGStrategy's matching logic:
    - signal is evaluated at bar i = entry_idx - 1 (market orders fill next bar)
    - scan bars [i - max_fvg_age, i-1] in ascending order
    - select the first FVG whose zone intersects the signal bar's [low, high]
      and whose direction matches the trade.
    Returns (matched_fvg_dict_or_None, signal_index_or_None).
    """
    i = int(entry_idx) - 1
    if i <= 0:
        return None, None
    current_high = float(base_high[i])
    current_low = float(base_low[i])
    start_bar = max(0, i - int(max_fvg_age))
    for bar_idx in range(start_bar, i):
        bucket = fvgs_by_bar.get(bar_idx)
        if not bucket:
            continue
        for fvg in bucket:
            is_bull = bool(fvg.get("is_bull"))
            # Direction 1 for bull, -1 for bear
            if (direction > 0) != is_bull:
                continue
            try:
                max_price = float(fvg["max_price"])  # top of zone for bulls
                min_price = float(fvg["min_price"])  # bottom of zone for bulls
            except Exception:
                continue
            in_zone = (current_low <= max_price) and (current_high >= min_price)
            if in_zone:
                return fvg, i
    return None, None


def build_master_dataset(
    *,
    base_pl_bt: pl.DataFrame,
    fvgs: List[Dict[str, Any]],
    trades_df: pl.DataFrame,
    fvg_threshold: float,
    max_fvg_age: int,
    match_tolerance_pct: float = 0.0,
    run_meta: Optional[Dict[str, Any]] = None,
) -> pl.DataFrame:
    """
    Produce a combined dataset joining each trade to its originating FVG
    (if any). All operations are Polars-native except lightweight Python loops
    over trades bounded by max_fvg_age for matching efficiency.
    """
    # Convenience arrays for matching
    base_high = base_pl_bt.get_column("High").to_numpy()
    base_low = base_pl_bt.get_column("Low").to_numpy()
    has_ts = "timestamp" in base_pl_bt.columns
    ts_col = base_pl_bt.get_column("timestamp") if has_ts else None

    fvgs_by_bar = _build_fvgs_by_bar(fvgs, fvg_threshold=fvg_threshold)

    # Build rows by iterating trades and attaching matched FVG fields
    master_rows: List[dict] = []
    for row in trades_df.iter_rows(named=True):
        entry_idx = int(row.get("EntryIdx", -1))
        direction = int(row.get("Direction", 0))
        matched_fvg, signal_idx = _match_trade_to_fvg(
            entry_idx=entry_idx,
            direction=direction,
            base_high=base_high,
            base_low=base_low,
            fvgs_by_bar=fvgs_by_bar,
            max_fvg_age=max_fvg_age,
        )

        rec: dict[str, Any] = dict(row)
        # Keep TradeId if present in trades_df
        if "TradeId" in trades_df.columns:
            rec["TradeId"] = row.get("TradeId")
        # Attach timestamps if available
        if has_ts:
            rec["EntryTimestamp"] = int(ts_col[entry_idx]) if (0 <= entry_idx < base_pl_bt.height) else None
            exit_idx = int(row.get("ExitIdx", -1))
            rec["ExitTimestamp"] = int(ts_col[exit_idx]) if (0 <= exit_idx < base_pl_bt.height) else None
            rec["SignalIdx"] = signal_idx
            rec["SignalTimestamp"] = (
                int(ts_col[signal_idx]) if (signal_idx is not None and 0 <= signal_idx < base_pl_bt.height) else None
            )

        # Merge FVG fields (prefixed in _flatten_fvgs)
        if matched_fvg is not None:
            fvg_bar = int(matched_fvg.get("bar_index", -1))
            fvg_mid = matched_fvg.get("midpoint")
            fvg_bull = bool(matched_fvg.get("is_bull"))
            rec.update(
                {
                    "fvg_bar_index": fvg_bar,
                    "fvg_is_bull": fvg_bull,
                    "fvg_max_price": matched_fvg.get("max_price"),
                    "fvg_min_price": matched_fvg.get("min_price"),
                    "fvg_midpoint": fvg_mid,
                    "fvg_gap_size_percent": matched_fvg.get("gap_size_percent"),
                    "fvg_displacement_strength": matched_fvg.get("displacement_strength"),
                }
            )
            # Flatten EMA bias
            bias = matched_fvg.get("ema_htf_bias") or {}
            if isinstance(bias, dict):
                for k, v in bias.items():
                    rec[f"ema_htf_{k}"] = v
            if has_ts and 0 <= fvg_bar < base_pl_bt.height:
                rec["fvg_timestamp"] = int(ts_col[fvg_bar])
        master_rows.append(rec)

    master_df = pl.DataFrame(master_rows) if master_rows else pl.DataFrame({})

    # Derived columns (compute at or before entry using only available info)
    if master_df.height:
        eps = float(match_tolerance_pct)
        # Base safe denominators
        denom_long = (pl.col("EntryPrice") - pl.col("SL"))
        denom_short = (pl.col("SL") - pl.col("EntryPrice"))
        denom_rr = pl.when(pl.col("Direction") == 1).then(denom_long).otherwise(denom_short)
        # Planned R
        planned_num = pl.when(pl.col("Direction") == 1)
        planned_num = planned_num.then(pl.col("TP") - pl.col("EntryPrice")).otherwise(pl.col("EntryPrice") - pl.col("TP"))
        planned_R = pl.when(denom_rr != 0).then(planned_num / denom_rr).otherwise(None)
        # Realized R (uses exit price; outcome)
        realized_num = pl.when(pl.col("Direction") == 1)
        realized_num = realized_num.then(pl.col("ExitPrice") - pl.col("EntryPrice")).otherwise(pl.col("EntryPrice") - pl.col("ExitPrice"))
        realized_R = pl.when((denom_rr != 0) & pl.col("ExitPrice").is_not_null()).then(realized_num / denom_rr).otherwise(None)
        # Holding bars
        holding_bars = pl.when(pl.col("ExitIdx") >= 0).then(pl.col("ExitIdx") - pl.col("EntryIdx")).otherwise(None)
        # Signal to entry delay
        signal_to_entry_delay_bars = pl.when(pl.col("SignalIdx").is_not_null()).then(pl.col("EntryIdx") - pl.col("SignalIdx")).otherwise(None)
        # FVG age at entry
        fvg_age_bars = pl.when(pl.col("fvg_bar_index").is_not_null()).then(pl.col("EntryIdx") - pl.col("fvg_bar_index")).otherwise(None)
        # Zone width
        zone_width_abs = (pl.col("fvg_max_price") - pl.col("fvg_min_price"))
        zone_width_pct = pl.when(pl.col("fvg_midpoint") != 0).then(zone_width_abs / pl.col("fvg_midpoint")).otherwise(None)
        # Matched zone with optional tolerance
        zmin_adj = pl.col("fvg_min_price") * (1.0 - eps)
        zmax_adj = pl.col("fvg_max_price") * (1.0 + eps)
        matched_zone = (
            (pl.col("EntryPrice") >= zmin_adj)
            & (pl.col("EntryPrice") <= zmax_adj)
            & pl.col("fvg_min_price").is_not_null()
            & pl.col("fvg_max_price").is_not_null()
        )
        # Entry position in zone
        entry_pos_in_zone = pl.when(zone_width_abs > 0)
        entry_pos_in_zone = entry_pos_in_zone.then(((pl.col("EntryPrice") - pl.col("fvg_min_price")) / zone_width_abs).clip(lower_bound=0.0, upper_bound=1.0)).otherwise(None)
        signed_entry_pos = pl.when(entry_pos_in_zone.is_not_null()).then((entry_pos_in_zone - 0.5) * pl.col("Direction")).otherwise(None)

        # PnL net and labels
        pnl_net = (pl.col("PnL") - pl.col("Commission")).alias("PnL_net")
        y_win_gross = (pl.col("PnL") > 0).cast(pl.Int8).alias("y_win_gross")
        y_win_net = (pnl_net > 0).cast(pl.Int8).alias("y_win_net")

        master_df = master_df.with_columns([
            pnl_net,
            y_win_gross,
            y_win_net,
            planned_R.alias("planned_R"),
            realized_R.alias("realized_R"),
            holding_bars.alias("holding_bars"),
            signal_to_entry_delay_bars.alias("signal_to_entry_delay_bars"),
            fvg_age_bars.alias("fvg_age_bars"),
            zone_width_abs.alias("zone_width_abs"),
            zone_width_pct.alias("zone_width_pct"),
            matched_zone.alias("matched_zone"),
            entry_pos_in_zone.alias("entry_pos_in_zone"),
            signed_entry_pos.alias("signed_entry_pos"),
        ])

        # HTF bias aggregates over dynamic ema_htf_* columns
        htf_cols = [c for c in master_df.columns if c.startswith("ema_htf_")]
        N = len(htf_cols)
        if N > 0:
            bull_exprs = [pl.when(pl.col(c) == "bullish").then(1).otherwise(0) for c in htf_cols]
            bear_exprs = [pl.when(pl.col(c) == "bearish").then(1).otherwise(0) for c in htf_cols]
            neut_exprs = [pl.when(pl.col(c) == "neutral").then(1).otherwise(0) for c in htf_cols]
            htf_bull_count = pl.sum_horizontal(bull_exprs).alias("htf_bull_count")
            htf_bear_count = pl.sum_horizontal(bear_exprs).alias("htf_bear_count")
            htf_neutral_count = pl.sum_horizontal(neut_exprs).alias("htf_neutral_count")
            # First add counts
            master_df = master_df.with_columns([
                htf_bull_count,
                htf_bear_count,
                htf_neutral_count,
            ])
            # Then compute ratios and logicals based on counts
            master_df = master_df.with_columns([
                (pl.col("htf_bull_count") / float(N)).alias("htf_bull_ratio"),
                (pl.col("htf_bear_count") / float(N)).alias("htf_bear_ratio"),
                (pl.col("htf_bull_count") == N).cast(pl.Int8).alias("htf_all_bullish"),
                (pl.col("htf_bear_count") == N).cast(pl.Int8).alias("htf_all_bearish"),
                (pl.col("htf_bull_count") > 0).cast(pl.Int8).alias("htf_any_bullish"),
                (pl.col("htf_bear_count") > 0).cast(pl.Int8).alias("htf_any_bearish"),
            ])
        else:
            master_df = master_df.with_columns([
                pl.lit(0).cast(pl.Int8).alias("htf_bull_count"),
                pl.lit(0).cast(pl.Int8).alias("htf_bear_count"),
                pl.lit(0).cast(pl.Int8).alias("htf_neutral_count"),
                pl.lit(0.0).alias("htf_bull_ratio"),
                pl.lit(0.0).alias("htf_bear_ratio"),
                pl.lit(0).cast(pl.Int8).alias("htf_all_bullish"),
                pl.lit(0).cast(pl.Int8).alias("htf_all_bearish"),
                pl.lit(0).cast(pl.Int8).alias("htf_any_bullish"),
                pl.lit(0).cast(pl.Int8).alias("htf_any_bearish"),
            ])

        # Time features from EntryTimestamp (UTC ms)
        if "EntryTimestamp" in master_df.columns:
            # Support Polars versions where from_epoch signature differs
            try:
                entry_dt = pl.from_epoch(pl.col("EntryTimestamp"), time_unit="ms")  # type: ignore[call-arg]
            except TypeError:
                entry_dt = pl.from_epoch(pl.col("EntryTimestamp"), unit="ms")  # fallback for newer versions
            master_df = master_df.with_columns([
                entry_dt.dt.hour().alias("hour_of_day"),
                entry_dt.dt.weekday().alias("day_of_week"),
                (entry_dt.dt.weekday() >= 5).cast(pl.Int8).alias("is_weekend"),
                entry_dt.dt.strftime("%Y-%m").alias("cv_month"),
            ])

        # Stamp run metadata (constant columns)
        if run_meta:
            const_cols = [pl.lit(v).alias(k) for k, v in run_meta.items()]
            master_df = master_df.with_columns(const_cols)

    # Attach symbol/timeframe if present on base
    def _first_str(df: pl.DataFrame, col: str) -> Optional[str]:
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
    timeframe = _first_str(base_pl_bt, "timeframe")
    if symbol is not None:
        master_df = master_df.with_columns(pl.lit(symbol).alias("symbol")) if master_df.height else master_df
    if timeframe is not None:
        master_df = master_df.with_columns(pl.lit(timeframe).alias("timeframe")) if master_df.height else master_df

    return master_df


def main() -> None:
    df = asyncio.run(load_sample_data())
    if isinstance(df, dict):
        base_pl = df["1m"]
        htf_map = {tf: d for tf, d in df.items() if tf != "1m"}
    else:
        if "timeframe" in df.columns:
            base_pl = df.filter(pl.col("timeframe") == "1m")
            uniq = df.select(pl.col("timeframe").unique()).to_series().to_list()
            htf_map = {tf: df.filter(pl.col("timeframe") == tf) for tf in ["5m", "15m", "30m", "1h", "6h"] if tf in uniq}
        else:
            base_pl, htf_map = df, {}

    base_pl_bt = _normalize_base_for_backtest(base_pl)
    det_pl, clean_htf = _prepare_detector_inputs(base_pl_bt, htf_map)

    # Detect FVGs (percent units) and run backtest with same approach as compare_fvg_backtests
    fvgs = FairValueGapDetector(
        det_pl, threshold_percent=0.2, htf_dataframes=clean_htf or None, ema_length=20
    ).detect_fvg()

    bt = Backtest(
        data=base_pl_bt.select([c for c in ["timestamp", "Open", "High", "Low", "Close"] if c in base_pl_bt.columns]),
        strategy=FVGStrategy,
        cash=10_000.0,
        commission=0.002,
    )

    # Strategy parameters (percent units for FVG threshold/targets)
    fvg_threshold = 0.01
    max_fvg_age = 20
    res = bt.run(
        fvg_threshold=fvg_threshold,
        position_size=0.3,
        max_fvg_age=max_fvg_age,
        profit_target=0.1,
        loss_target=0.1,
        precomputed_fvgs=fvgs,
    )

    # Build run metadata snapshot
    run_id = uuid.uuid4().hex
    strategy_name = "FVGStrategy"
    commission_rate = 0.002
    slippage_rate = 0.0  # using spread=0.0 in engine
    position_sizing_rule = "fixed_fraction"
    position_fraction = 0.3
    max_concurrent_trades = 1
    # Start/End timestamps (inclusive) from base
    if "timestamp" in base_pl_bt.columns and base_pl_bt.height:
        start_ts = int(base_pl_bt[0, "timestamp"])  # type: ignore[index]
        end_ts = int(base_pl_bt[-1, "timestamp"])  # type: ignore[index]
    else:
        start_ts = 0
        end_ts = 0

    stats = res.stats
    run_meta = {
        "run_id": run_id,
        "strategy_name": strategy_name,
        "symbol": None,
        "timeframe": None,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "max_fvg_age": max_fvg_age,
        "fvg_threshold": fvg_threshold,
        "profit_target": 0.1,
        "loss_target": 0.1,
        "commission_rate": commission_rate,
        "slippage_rate": slippage_rate,
        "position_sizing_rule": position_sizing_rule,
        "position_fraction": position_fraction,
        "max_concurrent_trades": max_concurrent_trades,
        "run_return_ann_pct": float(stats.get("Return (Ann.) [%]", 0.0)),
        "run_vol_ann_pct": float(stats.get("Volatility (Ann.) [%]", 0.0)),
        "run_cagr_pct": float(stats.get("CAGR [%]", 0.0)),
        "run_sharpe": float(stats.get("Sharpe Ratio", 0.0)),
        "run_sortino": float(stats.get("Sortino Ratio", 0.0)),
        "run_calmar": float(stats.get("Calmar Ratio", 0.0)),
        "run_mdd_pct": float(stats.get("Max. Drawdown [%]", 0.0)),
        "run_trades": int(stats.get("# Trades", 0)),
        "run_win_rate_pct": float(stats.get("Win Rate [%]", 0.0)),
        "run_profit_factor": float(stats.get("Profit Factor", 0.0)),
        "run_sqn": float(stats.get("SQN", 0.0)),
        "run_expectancy_pct": float(stats.get("Expectancy [%]", 0.0)),
        "buy_hold_return_pct": float(stats.get("Buy & Hold Return [%]", 0.0)),
    }

    # Fill symbol/timeframe in run_meta if present
    def _first_str(df_: pl.DataFrame, col: str) -> Optional[str]:
        if col in df_.columns and df_.height:
            try:
                return str(df_.select(pl.col(col).first()).item())
            except Exception:
                try:
                    return str(df_.get_column(col)[0])
                except Exception:
                    return None
        return None

    run_meta["symbol"] = _first_str(base_pl_bt, "symbol")
    run_meta["timeframe"] = _first_str(base_pl_bt, "timeframe")

    master_df = build_master_dataset(
        base_pl_bt=base_pl_bt,
        fvgs=fvgs,
        trades_df=res.trades,
        fvg_threshold=fvg_threshold,
        max_fvg_age=max_fvg_age,
        match_tolerance_pct=0.0,
        run_meta=run_meta,
    )

    # Output
    out_dir = Path("data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine suffix from symbol/timeframe if present
    symbol = run_meta.get("symbol")
    tf = run_meta.get("timeframe")
    suffix = ""
    parts = [p for p in [symbol, tf] if p]
    if parts:
        suffix = "_" + "_".join(parts)

    out_path = out_dir / f"fvg_master_dataset{suffix}.csv"
    master_df.write_csv(out_path)
    print(len(res.trades))

if __name__ == "__main__":
    main()


