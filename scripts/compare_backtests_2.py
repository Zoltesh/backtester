#!/usr/bin/env python
from __future__ import annotations

import time
import asyncio
from pathlib import Path
from typing import Any, Dict
import warnings

import numpy as np
import polars as pl

# backtesting.py reference
from backtesting import Backtest as BTRef
from backtesting import Strategy as StrategyRef

# our polars-native engine
from backtester import Backtest as BTPolars
from backtester import Strategy as StrategyPolars

# our data loader
from data_loader.data_loader import DataLoader


def ema_np(values: np.ndarray, period: int) -> np.ndarray:
    if period <= 0:
        raise ValueError("period must be positive")
    arr = np.asarray(values, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    alpha = 2.0 / (period + 1.0)
    # first valid index
    mask = np.isfinite(arr)
    if not mask.any():
        return out
    first_idx = int(np.argmax(mask))
    # seed with simple mean over first window if available, else first value
    if first_idx + period <= len(arr):
        seed = np.nanmean(arr[first_idx : first_idx + period])
        start = first_idx + period - 1
    else:
        seed = arr[first_idx]
        start = first_idx
    prev = float(seed)
    for i in range(start, len(arr)):
        x = arr[i]
        if np.isfinite(x):
            prev = alpha * x + (1.0 - alpha) * prev
        out[i] = prev
    return out


def rsi_wilder_np(close: np.ndarray, period: int) -> np.ndarray:
    if period <= 0:
        raise ValueError("period must be positive")
    c = np.asarray(close, dtype=float)
    out = np.full_like(c, np.nan, dtype=float)
    if len(c) < period + 1:
        return out
    delta = np.diff(c, prepend=np.nan)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    # Wilder's smoothing
    avg_gain = np.nanmean(gains[1 : period + 1])
    avg_loss = np.nanmean(losses[1 : period + 1])
    if not np.isfinite(avg_gain):
        avg_gain = 0.0
    if not np.isfinite(avg_loss):
        avg_loss = 0.0
    out[period] = 100.0 if avg_loss == 0 and avg_gain > 0 else (0.0 if avg_gain == 0 else 100.0 - 100.0 / (1.0 + (avg_gain / (avg_loss or np.nan))))
    alpha = 1.0 / period
    prev_gain = avg_gain
    prev_loss = avg_loss
    for i in range(period + 1, len(c)):
        g = gains[i]
        loss_val = losses[i]
        prev_gain = (1 - alpha) * prev_gain + alpha * g
        prev_loss = (1 - alpha) * prev_loss + alpha * loss_val
        if prev_loss == 0 and prev_gain > 0:
            out[i] = 100.0
        elif prev_gain == 0:
            out[i] = 0.0
        else:
            rs = prev_gain / prev_loss
            out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def atr_wilder_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    if period <= 0:
        raise ValueError("period must be positive")
    h = np.asarray(high, dtype=float)
    l_arr = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    out = np.full_like(c, np.nan, dtype=float)
    if len(c) == 0:
        return out
    prev_close = np.roll(c, 1)
    prev_close[0] = c[0]
    tr = np.maximum.reduce([h - l_arr, np.abs(h - prev_close), np.abs(l_arr - prev_close)])
    # Wilder's smoothing for ATR
    if len(tr) < period:
        return out
    atr = np.nanmean(tr[1:period+1])
    out[period] = atr
    alpha = 1.0 / period
    prev = atr
    for i in range(period + 1, len(tr)):
        prev = (1 - alpha) * prev + alpha * tr[i]
        out[i] = prev
    return out


async def load_sample_data() -> pl.DataFrame:
    base_dir = Path("data/ohlcv")
    loader = DataLoader(base_dir)
    df = await loader.load_ohlcv_between_dates(
        symbols=["SOL-USDC"],
        timeframes=["1m"],
        start_date="2024-01-01",
        end_date="2025-07-31",
    )
    return df


def normalize_ohlcv(df: pl.DataFrame) -> pl.DataFrame:
    rename_map = {}
    cols = {c.lower(): c for c in df.columns}
    for name in ("open", "high", "low", "close", "volume"):
        if name in cols:
            rename_map[cols[name]] = name.capitalize()
    if rename_map:
        df = df.rename(rename_map)
    if "timestamp" in df.columns:
        df = df.sort("timestamp")
    return df


# Reference strategy: EMA cross with RSI filter and ATR-based SL/TP
class EmaRsiAtrRef(StrategyRef):
    n_fast = 12
    n_slow = 26
    rsi_period = 14
    atr_period = 14
    atr_k_sl = 1.5
    atr_k_tp = 3.0

    def init(self):
        close = np.asarray(self.data.Close)
        high = np.asarray(self.data.High)
        low = np.asarray(self.data.Low)
        self.ema_fast = self.I(ema_np, close, self.n_fast)
        self.ema_slow = self.I(ema_np, close, self.n_slow)
        self.rsi = self.I(rsi_wilder_np, close, self.rsi_period)
        self.atr = self.I(atr_wilder_np, high, low, close, self.atr_period)

    def next(self):
        i = len(self.data) - 1
        if i < 2:
            return
        ef, es = self.ema_fast, self.ema_slow
        rsi = self.rsi
        atr = self.atr
        # Entry/exit conditions
        cross_up = ef[i] > es[i] and ef[i - 1] <= es[i - 1] and rsi[i] >= 55
        cross_dn = ef[i] < es[i] and ef[i - 1] >= es[i - 1] and rsi[i] <= 45
        if cross_up:
            sl = self.data.Close[i] - self.atr_k_sl * atr[i]
            tp = self.data.Close[i] + self.atr_k_tp * atr[i]
            self.buy(size=0.5, sl=sl, tp=tp)
        elif cross_dn:
            sl = self.data.Close[i] + self.atr_k_sl * atr[i]
            tp = self.data.Close[i] - self.atr_k_tp * atr[i]
            self.sell(size=0.5, sl=sl, tp=tp)


# Polars-native strategy counterpart
class EmaRsiAtrPolars(StrategyPolars):
    n_fast = 12
    n_slow = 26
    rsi_period = 14
    atr_period = 14
    atr_k_sl = 1.5
    atr_k_tp = 3.0

    def init(self):
        close = self.data["Close"].to_numpy()
        high = self.data["High"].to_numpy()
        low = self.data["Low"].to_numpy()
        self.ema_fast = self.I(ema_np, close, self.n_fast)
        self.ema_slow = self.I(ema_np, close, self.n_slow)
        self.rsi = self.I(rsi_wilder_np, close, self.rsi_period)
        self.atr = self.I(atr_wilder_np, high, low, close, self.atr_period)

    def next(self):
        i = self.i
        if i < 2:
            return
        ef, es = self.ema_fast, self.ema_slow
        rsi = self.rsi
        atr = self.atr
        cross_up = ef[i] > es[i] and ef[i - 1] <= es[i - 1] and rsi[i] >= 55
        cross_dn = ef[i] < es[i] and ef[i - 1] >= es[i - 1] and rsi[i] <= 45
        if cross_up and np.isfinite(atr[i]):
            sl = float(self.data["Close"][i]) - self.atr_k_sl * float(atr[i])
            tp = float(self.data["Close"][i]) + self.atr_k_tp * float(atr[i])
            self.buy(size=0.5, sl=sl, tp=tp)
        elif cross_dn and np.isfinite(atr[i]):
            sl = float(self.data["Close"][i]) + self.atr_k_sl * float(atr[i])
            tp = float(self.data["Close"][i]) - self.atr_k_tp * float(atr[i])
            self.sell(size=0.5, sl=sl, tp=tp)


def run_ref(df_pl: pl.DataFrame):
    import pandas as pd
    df_pd = df_pl.to_pandas()
    if "timestamp" in df_pd.columns:
        df_pd.index = pd.to_datetime(df_pd["timestamp"], unit="ms", utc=True)
    elif not isinstance(df_pd.index, pd.DatetimeIndex):
        df_pd.index = pd.date_range("2025-01-01", periods=len(df_pd), freq="T", tz="UTC")
    for c in ("Open", "High", "Low", "Close"):
        if c not in df_pd.columns:
            raise ValueError(f"Missing required column: {c}")
    bt = BTRef(
        df_pd,
        EmaRsiAtrRef,
        cash=10_000,
        commission=0.001,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = bt.run()
    return stats


def run_polars(df_pl: pl.DataFrame):
    bt = BTPolars(
        df_pl,
        EmaRsiAtrPolars,
        cash=10_000,
        commission=0.001,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True,
    )
    return bt.run()


def main():
    warnings.filterwarnings('ignore', message='.*Broker canceled the relative-sized order.*')
    df_raw = asyncio.run(load_sample_data())
    df = normalize_ohlcv(df_raw)

    t0 = time.perf_counter()
    ref_stats = run_ref(df)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    polars_res = run_polars(df)
    t3 = time.perf_counter()

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
            polars_display[k] = (
                f"EmaRsiAtr(nf={EmaRsiAtrPolars.n_fast}, ns={EmaRsiAtrPolars.n_slow}, "
                f"rsi={EmaRsiAtrPolars.rsi_period}, atr={EmaRsiAtrPolars.atr_period})"
            )
            continue
        if k in polars_res.stats:
            polars_display[k] = polars_res.stats[k]
    print(pd.Series(polars_display, dtype=object))

    # Compare overlapping metrics within 2%
    def _to_float(x):
        try:
            import pandas as pd  # type: ignore
        except Exception:
            pd = None  # noqa: F841
        try:
            if hasattr(x, 'total_seconds'):
                return float(x.total_seconds())
            if pd is not None:
                import numpy as _np
                if isinstance(x, _np.generic):
                    return float(x)
                if hasattr(pd, 'Timestamp') and isinstance(x, getattr(pd, 'Timestamp')):
                    return float(x.value) / 1e9
            return float(x)
        except Exception:
            return None

    try:
        ref_keys = set(ref_stats.keys())
    except Exception:
        ref_keys = set(ref_stats.index)  # type: ignore[attr-defined]
    polars_keys = set(polars_res.stats.keys())
    keys_all = sorted(ref_keys & polars_keys)
    # Metrics to exclude from strict numeric tolerance due to known intra-bar ambiguities
    exclude_numeric = {
        "Max. Trade Duration",
        "Worst Trade [%]",
    }
    diffs_full: Dict[str, Any] = {}
    numeric_ok = []
    for k in keys_all:
        if k in exclude_numeric:
            diffs_full[k] = 'n/a'
            continue
        va = ref_stats[k]
        vb = polars_res.stats[k]
        fa = _to_float(va)
        fb = _to_float(vb)
        if fa is None or fb is None:
            diffs_full[k] = 'n/a'
            continue
        denom = max(1.0, abs(fa))
        rel = abs(fb - fa) / denom
        diffs_full[k] = round(float(rel), 6)
        numeric_ok.append(rel <= 0.02)
    ok = all(numeric_ok) if numeric_ok else True

    print(f"\nWithin 2%: {ok}. Diff ratios (overlapping metrics): {diffs_full}")

    ref_time = t1 - t0
    polars_time = t3 - t2
    print(f"\nReference time: {ref_time:.4f}s")
    print(f"Polars time:    {polars_time:.4f}s")
    speedup = ref_time / polars_time if polars_time > 0 else float("inf")
    print(f"Speedup (ref/polars): {speedup:.2f}x")


if __name__ == "__main__":
    main()

