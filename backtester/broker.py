from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import polars as pl


@dataclass(slots=True)
class Trade:
    direction: int  # 1 long, -1 short
    entry_index: int
    entry_price: float
    size: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    tag: Optional[str] = None
    exit_index: Optional[int] = None
    exit_price: Optional[float] = None

    def is_open(self) -> bool:
        return self.exit_index is None

    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.direction * self.size


class Broker:
    """
    Minimal broker with exclusive orders and no hedging.
    Supports market orders at open of next bar (or current close when trade_on_close=True).
    Commission is applied as relative rate.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        cash: float = 10_000.0,
        commission: float | tuple[float, float] = 0.0,
        spread: float = 0.0,
        margin: float = 1.0,
        trade_on_close: bool = False,
        exclusive_orders: bool = True,
    ) -> None:
        self.data = data
        self.cash = float(cash)
        # commission: fixed + relative per order
        if callable(commission):
            raise TypeError("Callable commission not supported in minimal broker")
        try:
            fixed, relative = commission  # type: ignore[misc]
        except Exception:
            fixed, relative = 0.0, float(commission)
        self._commission_fixed = float(fixed)
        self._commission_relative = float(relative)
        self.spread = float(spread)
        self.margin = float(margin)
        self._leverage = 1.0 / (self.margin if self.margin > 0 else 1.0)
        self.trade_on_close = bool(trade_on_close)
        self.exclusive_orders = bool(exclusive_orders)

        self._equity = float(cash)
        self._trades: List[Trade] = []
        self._open_trade_ref: Trade | None = None
        self._pending_orders: list[tuple[int, float, float | None, float | None, Optional[str]]] = []

        # cache columns for fast access
        # Cache as numpy views with stable dtypes for fast arithmetic
        self._open_col = data.get_column("Open").to_numpy()
        self._high_col = data.get_column("High").to_numpy()
        self._low_col = data.get_column("Low").to_numpy()
        self._close_col = data.get_column("Close").to_numpy()

    # Price accessors
    def _price_at(self, index: int) -> float:
        if self.trade_on_close:
            prev = index - 1
            if prev < 0:
                prev = 0
            return float(self._close_col[prev])
        # Fill at current bar open (matches reference engine's market order behavior)
        val = self._open_col[index]
        if np.isnan(val):
            val = self._close_col[index]
        return float(val)

    def _has_long_open(self) -> bool:
        t = self._open_trade_ref
        return True if t is None else (t.direction > 0)

    def _update_equity(self, price: float) -> None:
        # Equity is cash plus mark-to-market value of open position
        open_trade = self._open_trade_ref if (self._open_trade_ref and self._open_trade_ref.is_open()) else None
        position_value = 0.0
        if open_trade is not None:
            position_value = open_trade.direction * open_trade.size * price
        self._equity = self.cash + position_value

    def _commission(self, order_size: float, price: float) -> float:
        return self._commission_fixed + abs(order_size) * price * self._commission_relative

    def market_order(
        self,
        *,
        direction: int,
        index: int,
        size: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
        tag: Optional[str] = None,
    ) -> Optional[Trade]:
        # Queue market order to be executed at the next on_bar() call
        req_size = 1.0 - np.finfo(float).eps if size is None else float(size)
        self._pending_orders.append((direction, req_size, sl, tp, tag))
        return None

    def _close_trade(self, trade: Trade, index: int, price: float) -> None:
        if not trade.is_open():
            return
        trade.exit_index = index
        trade.exit_price = price
        # Close proceeds/cost and commission
        if trade.direction == 1:
            # Close long: receive proceeds
            self.cash += trade.size * price
        else:
            # Close short: pay back shares
            self.cash -= trade.size * price
        self.cash -= self._commission(trade.size, price)
        self._update_equity(price)
        if trade is self._open_trade_ref:
            self._open_trade_ref = None

    def on_bar(self, index: int) -> None:
        # 1) Process pending market orders at bar fill price
        fill_px = self._price_at(index)
        if self._pending_orders:
            # Iterate over a snapshot and clear from the source to avoid repeated list removals
            pending = self._pending_orders
            self._pending_orders = []
            for order in pending:
                direction, req_size, sl, tp, tag = order

                # Exclusive: close opposite trade first at raw fill price
                t = self._open_trade_ref
                if self.exclusive_orders and t is not None and t.is_open() and t.direction != direction:
                    self._close_trade(t, index, fill_px)
                # If exclusive and same direction is already open, ignore additional entries (no pyramiding)
                t = self._open_trade_ref
                if self.exclusive_orders and t is not None and t.is_open() and t.direction == direction:
                    continue

                # After potential close above, recompute equity/margin availability (reference-like)
                open_trade = self._open_trade_ref if (self._open_trade_ref and self._open_trade_ref.is_open()) else None
                margin_used = 0.0
                if open_trade is not None:
                    margin_used = abs(open_trade.entry_price * open_trade.size) / self._leverage
                equity_now = self.cash + (
                    0.0 if open_trade is None else (float(fill_px) - open_trade.entry_price) * open_trade.direction * open_trade.size
                )
                margin_available = max(0.0, equity_now - margin_used)

                # Determine integer units (reference-like sizing)
                entry_px = fill_px * (1 + (self.spread if direction > 0 else -self.spread))
                if 0 < req_size < 1:
                    available_notional = margin_available * self._leverage * abs(req_size)
                    per_unit_cost = entry_px + self._commission(1.0, entry_px)
                    units = int(available_notional // per_unit_cost)
                    size_units = float(max(units, 0))
                else:
                    size_units = float(int(abs(req_size)))
                if size_units <= 0:
                    continue

                # Liquidity check: ensure affordability; if not, downsize to max affordable whole units
                per_unit_cost = entry_px + self._commission(1.0, entry_px)
                max_affordable_units = int((margin_available * self._leverage) // per_unit_cost)
                if size_units > max_affordable_units:
                    size_units = float(max(max_affordable_units, 0))
                if size_units <= 0:
                    continue

                # Open trade at adjusted price; commission at entry on adjusted price
                if direction == 1:
                    # Buy: pay cost
                    self.cash -= size_units * entry_px
                else:
                    # Short: receive proceeds
                    self.cash += size_units * entry_px
                self.cash -= self._commission(size_units, entry_px)
                trade = Trade(direction=direction, entry_index=index, entry_price=entry_px, size=size_units, sl=sl, tp=tp, tag=tag)
                self._trades.append(trade)
                self._open_trade_ref = trade

        # 2) handle SL/TP for open trade
        open_trade = self._open_trade_ref if (self._open_trade_ref and self._open_trade_ref.is_open()) else None
        if open_trade is None:
            self._update_equity(fill_px)
            return

        high = float(self._high_col[index])
        low = float(self._low_col[index])
        price = float(self._close_col[index])

        # check stops
        if open_trade.direction == 1:
            if open_trade.tp is not None and high >= open_trade.tp:
                self._close_trade(open_trade, index, open_trade.tp)
                return
            if open_trade.sl is not None and low <= open_trade.sl:
                self._close_trade(open_trade, index, open_trade.sl)
                return
        else:
            if open_trade.tp is not None and low <= open_trade.tp:
                self._close_trade(open_trade, index, open_trade.tp)
                return
            if open_trade.sl is not None and high >= open_trade.sl:
                self._close_trade(open_trade, index, open_trade.sl)
                return

        # update equity with current price when no close
        self._update_equity(price)

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def trades(self) -> List[Trade]:
        return self._trades

    def close_all(self, index: int) -> None:
        price = float(self._close_col[index])
        for t in list(self._trades):
            if t.is_open():
                self._close_trade(t, index, price)


