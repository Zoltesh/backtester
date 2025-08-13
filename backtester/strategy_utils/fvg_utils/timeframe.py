from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Union

import polars as pl


_TF_RE = re.compile(r"^(\d+)([smhdwSMHDW])$")


def timeframe_to_ms(tf: str) -> Optional[int]:
    """Convert timeframe like '1m','5m','1h','1d','1w' to milliseconds. Return None if unknown."""
    tf = str(tf).strip()
    m = _TF_RE.match(tf)
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2).lower()
    if unit == "s":
        return n * 1_000
    if unit == "m":
        return n * 60_000
    if unit == "h":
        return n * 3_600_000
    if unit == "d":
        return n * 86_400_000
    if unit == "w":
        return n * 7 * 86_400_000
    return None


def select_base_and_htfs(
    data: Union[pl.DataFrame, Dict[str, pl.DataFrame]]
) -> Tuple[pl.DataFrame, Dict[str, pl.DataFrame], Optional[str]]:
    """
    Determine base timeframe as the smallest bar duration among available frames.
    Returns (base_df, htf_map, base_timeframe_str_or_None).

    - If a dict of {timeframe_str: df}: choose the key with min timeframe_to_ms.
      Unknown formats fall back to the df with the largest number of rows.
    - If a DataFrame with a 'timeframe' column: choose the smallest timeframe
      among unique values; return base_df filtered to that timeframe, and map
      the rest.
    - Otherwise: treat the provided DataFrame as base; no HTFs, unknown tf.
    """
    if isinstance(data, dict):
        if not data:
            raise ValueError("Empty data mapping passed to select_base_and_htfs")
        # Sort keys by parsed duration; unknowns go to +inf so they don't win
        scored: List[Tuple[str, int]] = []
        unknown_keys: List[str] = []
        for k, df in data.items():
            dur = timeframe_to_ms(k)
            if dur is None:
                unknown_keys.append(k)
            else:
                scored.append((k, dur))
        base_tf: Optional[str]
        if scored:
            base_tf = min(scored, key=lambda kv: kv[1])[0]
        else:
            # fallback: choose the df with the most rows
            base_tf = max(unknown_keys, key=lambda k: data[k].height if data[k] is not None else -1)
        base_df = data[base_tf]
        htf_map = {tf: d for tf, d in data.items() if tf != base_tf}
        return base_df, htf_map, base_tf

    # Single DataFrame case
    df = data
    if "timeframe" in df.columns:
        uniq = df.select(pl.col("timeframe").unique()).to_series().to_list()
        if not uniq:
            return df, {}, None
        # score each unique tf
        scored2: List[Tuple[str, int]] = []
        unknown2: List[str] = []
        for tf in uniq:
            dur = timeframe_to_ms(tf)
            if dur is None:
                unknown2.append(tf)
            else:
                scored2.append((tf, dur))
        if scored2:
            base_tf = min(scored2, key=lambda kv: kv[1])[0]
        else:
            # fallback: timeframe value with most rows
            counts = df.group_by("timeframe").len().sort("len", descending=True)
            base_tf = str(counts[0, "timeframe"])  # type: ignore[index]
        base_df = df.filter(pl.col("timeframe") == base_tf)
        htf_map = {tf: df.filter(pl.col("timeframe") == tf) for tf in uniq if tf != base_tf}
        return base_df, htf_map, base_tf

    return df, {}, None


