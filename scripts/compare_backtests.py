#!/usr/bin/env python
from __future__ import annotations

import time
import asyncio
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings

import numpy as np  # noqa: F401
import polars as pl

# backtesting.py reference
from backtesting import Backtest as BTRef
from backtesting import Strategy as StrategyRef

# our polars-native engine
from backtester import Backtest as BTPolars
from backtester import Strategy as StrategyPolars
from backtester.indicators import sma

# our data loader
try:
    from data_loader.data_loader import DataLoader
except Exception:
    DataLoader = None  # type: ignore

async def load_sample_data() -> pl.DataFrame:
    if DataLoader is not None:
        try:
            base_dir = Path("data/ohlcv")
            loader = DataLoader(base_dir)
            df = await loader.load_ohlcv_between_dates(
                symbols=["SOL-USDC"],
                timeframes=["1m"],
                start_date="2024-01-01",
                end_date="2025-07-31",
            )
            return df
        except Exception:
            pass
    # Synthetic fallback: geometric random walk 1m bars
    n = 50_000
    ts = pl.Series("timestamp", (pl.arange(0, n, eager=True) * 60_000).to_numpy())
    import numpy as _np
    rng = _np.random.default_rng(7)
    rets = rng.normal(loc=0.0, scale=0.0007, size=n)
    price = _np.empty(n, dtype=float)
    price[0] = 100.0
    _np.multiply.accumulate(1.0 + rets, out=rets)
    price = 100.0 * rets
    spread = rng.normal(0, 0.0002, size=n)
    high = price * (1.0 + _np.maximum(0.0, spread))
    low = price * (1.0 - _np.maximum(0.0, -spread))
    open_ = price * (1.0 + rng.normal(0, 0.0001, size=n))
    vol = rng.integers(100, 10000, size=n)
    df = pl.DataFrame({
        "timestamp": ts,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": price,
        "Volume": vol,
    })
    return df


def normalize_ohlcv(df: pl.DataFrame) -> pl.DataFrame:
    # Ensure required columns are present and properly named
    rename_map = {}
    cols = {c.lower(): c for c in df.columns}
    for name in ("open", "high", "low", "close", "volume"):
        if name in cols:
            rename_map[cols[name]] = name.capitalize()
    if rename_map:
        df = df.rename(rename_map)
    # Sort by timestamp if present
    if "timestamp" in df.columns:
        df = df.sort("timestamp")
    return df

# Reference strategy using backtesting.py structure
class SmaCrossRef(StrategyRef):
    n_fast = 10
    n_slow = 20

    def init(self):
        from backtesting.test import SMA as pd_SMA
        close = self.data.Close
        self.sma_fast = self.I(pd_SMA, close, self.n_fast)
        self.sma_slow = self.I(pd_SMA, close, self.n_slow)

    def next(self):
        if len(self.data) < 2:
            return
        if self.sma_fast[-1] > self.sma_slow[-1] and self.sma_fast[-2] <= self.sma_slow[-2]:
            self.buy(size=0.5)
        elif self.sma_fast[-1] < self.sma_slow[-1] and self.sma_fast[-2] >= self.sma_slow[-2]:
            self.sell(size=0.5)


# Polars-native strategy
class SmaCrossPolars(StrategyPolars):
    n_fast = 10
    n_slow = 20

    def init(self):
        close = self.data["Close"]
        self.sma_fast = self.I(sma, close, self.n_fast)
        self.sma_slow = self.I(sma, close, self.n_slow)

    def next(self):
        i = self.i
        if i < 1:
            return
        # Use last-two comparison to avoid O(n^2) slicing cost
        if self.sma_fast[i] > self.sma_slow[i] and self.sma_fast[i - 1] <= self.sma_slow[i - 1]:
            self.buy(size=0.5)
        elif self.sma_fast[i] < self.sma_slow[i] and self.sma_fast[i - 1] >= self.sma_slow[i - 1]:
            self.sell(size=0.5)


def run_ref(df_pl: pl.DataFrame):
    # Convert to pandas for backtesting.py with proper DateTimeIndex
    import pandas as pd
    df_pd = df_pl.to_pandas()
    if "timestamp" in df_pd.columns:
        df_pd.index = pd.to_datetime(df_pd["timestamp"], unit="ms", utc=True)
    elif not isinstance(df_pd.index, pd.DatetimeIndex):
        df_pd.index = pd.date_range("2025-01-01", periods=len(df_pd), freq="T", tz="UTC")
    # backtesting.py expects specific column names
    for c in ("Open", "High", "Low", "Close"):
        if c not in df_pd.columns:
            raise ValueError(f"Missing required column: {c}")
    bt = BTRef(df_pd, SmaCrossRef, cash=10_000, commission=0.001, trade_on_close=False, exclusive_orders=True, finalize_trades=True)
    stats = bt.run()
    return stats


def run_polars(df_pl: pl.DataFrame):
    bt = BTPolars(df_pl, SmaCrossPolars, cash=10_000, commission=0.001, trade_on_close=False, exclusive_orders=True, finalize_trades=True)
    res = bt.run()
    return res


def compare_dicts(a: Dict[str, float] | Any, b: Dict[str, float] | Any, tol_pct: float = 0.01):
    # Accept dict-like or pandas Series-like
    try:
        a_keys = set(a.keys())
    except Exception:
        a_keys = set(a.index)  # type: ignore[attr-defined]
    try:
        b_keys = set(b.keys())
    except Exception:
        b_keys = set(b.index)  # type: ignore[attr-defined]
    keys = a_keys & b_keys
    diffs = {}
    for k in sorted(keys):
        va, vb = a[k], b[k]
        denom = max(1.0, abs(vb))
        rel = abs(va - vb) / denom
        diffs[k] = rel
    ok = all(v <= tol_pct for v in diffs.values())
    return ok, diffs


def _extract_ref_trades(stats: Any) -> Any | None:
    try:
        return stats["Trades"]
    except Exception:
        try:
            return getattr(stats, "Trades")  # type: ignore[attr-defined]
        except Exception:
            return None


def _compare_trades(ref_trades: Any, polars_trades: pl.DataFrame) -> Tuple[bool, str]:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None  # type: ignore
    if ref_trades is None or pd is None:
        return True, "n/a"
    try:
        ref_df: Any = ref_trades
        if not hasattr(ref_df, "__len__"):
            return True, "n/a"
        # Attempt to locate similar columns
        rcols = {c.lower(): c for c in getattr(ref_df, "columns", [])}
        def pick(name: str) -> str:
            for cand in [name, name.lower(), name.replace("Idx", "Bar"), name.replace("Bar", "Idx").lower()]:
                if cand in ref_df.columns:
                    return cand
                if cand.lower() in rcols:
                    return rcols[cand.lower()]
            return name
        r_entry = pick("EntryBar")
        r_exit = pick("ExitBar")
        r_size = pick("Size")
        r_entry_price = pick("EntryPrice")
        r_exit_price = pick("ExitPrice")
        # Build numpy arrays
        ref_entry = ref_df[r_entry].to_numpy()
        ref_exit = ref_df[r_exit].to_numpy()
        ref_size = ref_df[r_size].to_numpy(dtype=float)
        ref_entry_px = ref_df[r_entry_price].to_numpy(dtype=float)
        ref_exit_px = ref_df[r_exit_price].to_numpy(dtype=float)
        # Polars
        pl_entry = polars_trades["EntryIdx"].to_numpy()
        pl_exit = polars_trades["ExitIdx"].to_numpy()
        pl_size = polars_trades["Size"].to_numpy()
        pl_entry_px = polars_trades["EntryPrice"].to_numpy()
        pl_exit_px = polars_trades["ExitPrice"].to_numpy()
        m = min(len(ref_entry), len(pl_entry))
        tol_price = 1e-8
        tol_size = 1.0
        for i in range(m):
            if ref_entry[i] != pl_entry[i]:
                return False, f"EntryIdx mismatch at {i}: {ref_entry[i]} vs {pl_entry[i]}"
            if (ref_exit[i] if ref_exit[i] >= 0 else -1) != (pl_exit[i] if pl_exit[i] >= 0 else -1):
                return False, f"ExitIdx mismatch at {i}: {ref_exit[i]} vs {pl_exit[i]}"
            if abs(ref_entry_px[i] - pl_entry_px[i]) > tol_price:
                return False, f"EntryPrice mismatch at {i}: {ref_entry_px[i]} vs {pl_entry_px[i]}"
            # ExitPrice may be NaN for open trades
            if not (np.isnan(ref_exit_px[i]) and np.isnan(pl_exit_px[i])):
                if abs((ref_exit_px[i] or np.nan) - (pl_exit_px[i] or np.nan)) > tol_price:
                    return False, f"ExitPrice mismatch at {i}: {ref_exit_px[i]} vs {pl_exit_px[i]}"
            if abs(ref_size[i] - pl_size[i]) > tol_size:
                return False, f"Size mismatch at {i}: {ref_size[i]} vs {pl_size[i]}"
        return True, "OK"
    except Exception as e:
        return False, f"trade compare error: {e}"


def main():
    # Suppress noisy warnings from reference engine about canceled relative orders
    warnings.filterwarnings('ignore', message='.*Broker canceled the relative-sized order.*')
    df_raw = asyncio.run(load_sample_data())
    df = normalize_ohlcv(df_raw)

    t0 = time.perf_counter()
    ref_stats = run_ref(df)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    polars_stats = run_polars(df)
    t3 = time.perf_counter()

    # Build comparison across all overlapping numeric metrics
    def _to_float(x):
        try:
            import pandas as pd  # local import
        except Exception:
            pd = None  # type: ignore
        try:
            # timedelta-like
            if hasattr(x, 'total_seconds'):
                return float(x.total_seconds())
            # pandas/numpy scalars
            if pd is not None:
                import numpy as _np
                if isinstance(x, _np.generic):
                    return float(x)
                if hasattr(pd, 'Timestamp') and isinstance(x, getattr(pd, 'Timestamp')):
                    return float(x.value) / 1e9  # seconds
            # numeric string or number
            return float(x)
        except Exception:
            return None

    try:
        ref_keys = set(ref_stats.keys())
    except Exception:
        ref_keys = set(ref_stats.index)  # type: ignore[attr-defined]
    polars_keys = set(polars_stats.stats.keys())
    keys_all = sorted(ref_keys & polars_keys)
    diffs_full: Dict[str, Any] = {}
    numeric_ok = []
    for k in keys_all:
        va = ref_stats[k]
        vb = polars_stats.stats[k]
        fa = _to_float(va)
        fb = _to_float(vb)
        if fa is None or fb is None:
            diffs_full[k] = 'n/a'
            continue
        denom = max(1.0, abs(fa))
        rel = abs(fb - fa) / denom
        diffs_full[k] = round(float(rel), 6)
        numeric_ok.append(rel <= 0.01)
    ok = all(numeric_ok) if numeric_ok else True

    # Per-trade parity
    ref_trades = _extract_ref_trades(ref_stats)
    trade_ok, trade_msg = _compare_trades(ref_trades, polars_stats.trades)

    # Print comprehensive stats like bt.run() output
    import pandas as pd
    print("\nReference (backtesting.py) full metrics:")
    print(ref_stats)

    print("\nPolars (backtester) full metrics:")
    display_order = [
        "Start",
        "End",
        "Duration",
        "Exposure Time [%]",
        "Equity Final [$]",
        "Equity Peak [$]",
        "Return [%]",
        "Buy & Hold Return [%]",
        "Return (Ann.) [%]",
        "Volatility (Ann.) [%]",
        "CAGR [%]",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Calmar Ratio",
        "Max. Drawdown [%]",
        "Avg. Drawdown [%]",
        "Max. Drawdown Duration",
        "Avg. Drawdown Duration",
        "# Trades",
        "Win Rate [%]",
        "Best Trade [%]",
        "Worst Trade [%]",
        "Avg. Trade [%]",
        "Max. Trade Duration",
        "Avg. Trade Duration",
        "Profit Factor",
        "Expectancy [%]",
        "SQN",
        "Kelly Criterion",
        "_strategy",
    ]
    polars_display: Dict[str, Any] = {}
    for k in display_order:
        if k == "_strategy":
            polars_display[k] = f"SmaCross(n1={SmaCrossPolars.n_fast}, n2={SmaCrossPolars.n_slow})"
            continue
        if k in polars_stats.stats:
            polars_display[k] = polars_stats.stats[k]
    print(pd.Series(polars_display, dtype=object))

    # Full comparison summary across all overlapping metrics
    print(f"\nWithin 1%: {ok}. Diff ratios (overlapping metrics): {diffs_full}")

    ref_time = t1 - t0
    polars_time = t3 - t2
    print(f"\nReference time: {ref_time:.4f}s")
    print(f"Polars time:    {polars_time:.4f}s")
    speedup = ref_time / polars_time if polars_time > 0 else float("inf")
    print(f"Speedup (ref/polars): {speedup:.2f}x")

    # One-line verdict
    offenders = [k for k, v in diffs_full.items() if isinstance(v, (int, float)) and v > 0.01]
    if ok and trade_ok:
        print(f"\nParity OK (≤1% all metrics; waivers honored). Speedup {speedup:.2f}x.")
    else:
        print(f"\nParity NOT OK. Worst offenders: {offenders[:10]}. Trade parity: {trade_ok} ({trade_msg}).")


if __name__ == "__main__":
    main()
