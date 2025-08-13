from __future__ import annotations

from typing import Dict

import polars as pl


def prepare_detector_inputs(base_pl_bt: pl.DataFrame, htf_map: Dict[str, pl.DataFrame]) -> tuple[pl.DataFrame, Dict[str, pl.DataFrame]]:
    """
    Prepare inputs for FairValueGapDetector:
    - detector DataFrame with lowercase ohlcv (volume optional → default 1.0)
    - HTF map containing only a 'close' column
    """
    det_cols: Dict[str, pl.Series] = {}
    for src, dst in [("High", "high"), ("Low", "low"), ("Close", "close"), ("Volume", "volume")]:
        if src in base_pl_bt.columns:
            det_cols[dst] = base_pl_bt.get_column(src)
    if "volume" not in det_cols:
        det_cols["volume"] = pl.Series(name="volume", values=[1.0] * base_pl_bt.height)
    det_pl = pl.DataFrame(det_cols)

    clean_htf: Dict[str, pl.DataFrame] = {}
    for tf, hdf in htf_map.items():
        if "Close" in hdf.columns:
            clean_htf[tf] = hdf.select(["Close"]).rename({"Close": "close"})
        elif "close" in hdf.columns:
            clean_htf[tf] = hdf.select(["close"])  # already lowercase
    return det_pl, clean_htf


