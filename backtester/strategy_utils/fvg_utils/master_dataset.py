from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Callable

import polars as pl


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


# Extensibility: plug-in feature builder signature
AdditionalFeatureBuilder = Callable[[pl.DataFrame], pl.DataFrame]


def build_master_dataset(
    *,
    base_pl_bt: pl.DataFrame,
    fvgs: List[Dict[str, Any]],
    trades_df: pl.DataFrame,
    fvg_threshold: float,
    max_fvg_age: int,
    match_tolerance_pct: float = 0.0,
    run_meta: Optional[Dict[str, Any]] = None,
    additional_feature_builders: Optional[List[AdditionalFeatureBuilder]] = None,
) -> pl.DataFrame:
    """
    Produce a combined dataset joining each trade to its originating FVG
    (if any). Pure Polars except a tight Python loop bounded by max_fvg_age for
    matching.

    Additional feature builders can be supplied to extend the dataset with
    more features (e.g., ATR, RSI, stochastic). Builders receive and must
    return a DataFrame.
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

    # Optional extensibility hooks
    if additional_feature_builders:
        for builder in additional_feature_builders:
            master_df = builder(master_df)

    return master_df


