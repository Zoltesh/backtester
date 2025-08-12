from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Type

import numpy as np
import polars as pl

from .broker import Broker
from .strategy import Strategy
from .stats import compute_stats


@dataclass
class Result:
    stats: Dict[str, Any]
    trades: pl.DataFrame
    equity: pl.Series


class Backtest:
    """
    Polars-native backtest engine mirroring backtesting.py API where practical.
    Data must be a Polars DataFrame with columns: Open, High, Low, Close, optional Volume.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        strategy: Type[Strategy],
        *,
        cash: float = 10_000.0,
        commission: float | tuple[float, float] = 0.0,
        spread: float = 0.0,
        margin: float = 1.0,
        trade_on_close: bool = False,
        exclusive_orders: bool = True,
        finalize_trades: bool = True,
    ) -> None:
        if not isinstance(data, pl.DataFrame):
            raise TypeError("data must be a polars.DataFrame")
        required = {"Open", "High", "Low", "Close"}
        missing = required.difference(set(data.columns))
        if missing:
            raise ValueError(f"data is missing required columns: {sorted(missing)}")
        if not isinstance(strategy, type) or not issubclass(strategy, Strategy):
            raise TypeError("strategy must be a Strategy subclass")

        self.data = data
        self.strategy_cls = strategy
        self.broker = Broker(
            data,
            cash=cash,
            commission=commission,
            spread=spread,
            margin=margin,
            trade_on_close=trade_on_close,
            exclusive_orders=exclusive_orders,
        )
        self.finalize_trades = finalize_trades

    def run(self, **params: Any) -> Result:
        strat: Strategy = self.strategy_cls(self.broker, self.data, params)
        strat.init()

        # Determine warmup start based on indicators declared in init
        warmups: list[int] = []
        for ind in getattr(strat, "_indicators", []):
            arr = np.asarray(ind)
            # find first non-NaN index
            if arr.ndim == 1:
                valid = ~np.isnan(arr)
                idx = int(np.argmax(valid)) if valid.any() else 0
                warmups.append(idx)
            else:
                firsts = []
                for col in arr.T:
                    valid = ~np.isnan(col)
                    firsts.append(int(np.argmax(valid)) if valid.any() else 0)
                if firsts:
                    warmups.append(max(firsts))
        start_index = 1 + (max(warmups) if warmups else 0)

        n = len(self.data)
        equity_values = np.empty(n, dtype=float)
        # Pre-fill warmup region with initial cash to avoid repeated writes
        if start_index > 0:
            equity_values[:start_index] = self.broker.cash
        # iterate rows sequentially like backtesting.py
        for i in range(start_index, n):
            strat.i = i
            self.broker.on_bar(i)
            strat.next()
            equity_values[i] = self.broker.equity

        # finalize open trades if requested
        if self.finalize_trades and self.broker.trades:
            last_idx = len(self.data) - 1
            self.broker.close_all(last_idx)
            equity_values[-1] = self.broker.equity

        # wrap results
        trades_df = _trades_to_df(self.broker.trades, self.data)
        equity_series = pl.Series(name="Equity", values=equity_values)
        stats = compute_stats(
            self.data,
            equity_series,
            trades_df,
            first_trading_bar=start_index - 1 if start_index > 0 else 0,
            strategy=strat,
        )
        return Result(stats=stats, trades=trades_df, equity=equity_series)


def _trades_to_df(trades, data: pl.DataFrame) -> pl.DataFrame:
    if not trades:
        return pl.DataFrame(
            {
                "EntryIdx": pl.Series(name="EntryIdx", values=[], dtype=pl.Int64),
                "ExitIdx": pl.Series(name="ExitIdx", values=[], dtype=pl.Int64),
                "Direction": pl.Series(name="Direction", values=[], dtype=pl.Int8),
                "EntryPrice": pl.Series(name="EntryPrice", values=[], dtype=pl.Float64),
                "ExitPrice": pl.Series(name="ExitPrice", values=[], dtype=pl.Float64),
                "SL": pl.Series(name="SL", values=[], dtype=pl.Float64),
                "TP": pl.Series(name="TP", values=[], dtype=pl.Float64),
                "Size": pl.Series(name="Size", values=[], dtype=pl.Float64),
                "PnL": pl.Series(name="PnL", values=[], dtype=pl.Float64),
                "Commission": pl.Series(name="Commission", values=[], dtype=pl.Float64),
                "Tag": pl.Series(name="Tag", values=[], dtype=pl.Utf8),
            }
        )
    m = len(trades)
    entry_idx = np.empty(m, dtype=np.int64)
    exit_idx = np.empty(m, dtype=np.int64)
    direction = np.empty(m, dtype=np.int8)
    entry_price = np.empty(m, dtype=np.float64)
    exit_price = np.empty(m, dtype=np.float64)
    size = np.empty(m, dtype=np.float64)
    pnl = np.empty(m, dtype=np.float64)
    commission = np.empty(m, dtype=np.float64)
    sl_arr = np.empty(m, dtype=np.float64)
    tp_arr = np.empty(m, dtype=np.float64)
    trade_ids: list[str] = [""] * m
    # For tags, build Python list then let Polars handle utf8
    tags: list[str] = [""] * m
    for i, t in enumerate(trades):
        entry_idx[i] = t.entry_index
        exit_idx[i] = -1 if t.exit_index is None else t.exit_index
        direction[i] = t.direction
        entry_price[i] = t.entry_price
        exit_price[i] = np.nan if t.exit_price is None else t.exit_price
        size[i] = t.size
        # Report gross PnL excluding commissions for trade-level parity; commissions in separate column
        pnl[i] = (0.0 if t.exit_price is None else (t.exit_price - t.entry_price) * t.direction * t.size)
        commission[i] = float(getattr(t, "entry_commission", 0.0) + getattr(t, "exit_commission", 0.0))
        tags[i] = t.tag if t.tag is not None else ""
        sl_arr[i] = np.nan if getattr(t, "sl", None) is None else float(getattr(t, "sl"))
        tp_arr[i] = np.nan if getattr(t, "tp", None) is None else float(getattr(t, "tp"))
        trade_ids[i] = getattr(t, "trade_id", "")
    return pl.DataFrame(
        {
            "EntryIdx": pl.Series(name="EntryIdx", values=entry_idx),
            "ExitIdx": pl.Series(name="ExitIdx", values=exit_idx),
            "Direction": pl.Series(name="Direction", values=direction),
            "EntryPrice": pl.Series(name="EntryPrice", values=entry_price),
            "ExitPrice": pl.Series(name="ExitPrice", values=exit_price),
            "Size": pl.Series(name="Size", values=size),
            "SL": pl.Series(name="SL", values=sl_arr),
            "TP": pl.Series(name="TP", values=tp_arr),
            "PnL": pl.Series(name="PnL", values=pnl),
            "Commission": pl.Series(name="Commission", values=commission),
            "Tag": pl.Series(name="Tag", values=tags, dtype=pl.Utf8),
            "TradeId": pl.Series(name="TradeId", values=trade_ids, dtype=pl.Utf8),
        }
    )


