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
    # Net accounting
    entry_commission: float = 0.0
    exit_commission: float = 0.0

    def is_open(self) -> bool:
        return self.exit_index is None

    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        gross = (self.exit_price - self.entry_price) * self.direction * self.size
        return gross - (self.entry_commission + self.exit_commission)


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
        commission_on: str = "pre",  # 'pre'|'post' – base for commission calculation (mirror reference)
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
        if commission_on not in ("pre", "post"):
            raise ValueError("commission_on must be 'pre' or 'post'")
        self._commission_on = commission_on

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
        # Equity is cash plus mark-to-market PnL of open position (matches backtesting.py)
        open_trade = self._open_trade_ref if (self._open_trade_ref and self._open_trade_ref.is_open()) else None
        unrealized_pl = 0.0
        if open_trade is not None:
            unrealized_pl = open_trade.direction * open_trade.size * (price - open_trade.entry_price)
        self._equity = self.cash + unrealized_pl

    def _commission(self, order_size: float, price: float) -> float:
        return self._commission_fixed + abs(order_size) * price * self._commission_relative

    def _variable_commission_per_unit(self, price: float) -> float:
        """Return variable (relative) commission per 1 unit at given price (excludes fixed)."""
        return price * self._commission_relative

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
        # Realize PnL and apply commission (match backtesting.py accounting)
        realized_pl = trade.direction * trade.size * (price - trade.entry_price)
        close_commission = self._commission(trade.size, price)
        trade.exit_commission = close_commission
        self.cash += realized_pl
        self.cash -= close_commission
        self._update_equity(price)
        if trade is self._open_trade_ref:
            self._open_trade_ref = None

    def _process_contingent(self, index: int) -> bool:
        open_trade = self._open_trade_ref if (self._open_trade_ref and self._open_trade_ref.is_open()) else None
        if open_trade is None:
            return False
        high = float(self._high_col[index])
        low = float(self._low_col[index])
        open_val = self._open_col[index]
        open_px = float(self._close_col[index]) if np.isnan(open_val) else float(open_val)
        if open_trade.direction == 1:
            sl_hit = (open_trade.sl is not None) and (low <= open_trade.sl)
            tp_hit = (open_trade.tp is not None) and (high >= open_trade.tp)
            if sl_hit and tp_hit:
                # Prioritize SL like reference (SL orders processed first)
                # Long SL: gap-through fills at open if worse than stop
                self._close_trade(open_trade, index, min(open_px, float(open_trade.sl)))
                return True
            if sl_hit:
                # Long SL: price = min(open, stop)
                self._close_trade(open_trade, index, min(open_px, float(open_trade.sl)))
                return True
            if tp_hit:
                # Long TP (limit sell): fill at max(open, limit) to mirror reference behavior
                self._close_trade(open_trade, index, max(open_px, float(open_trade.tp)))
                return True
        else:
            sl_hit = (open_trade.sl is not None) and (high >= open_trade.sl)
            tp_hit = (open_trade.tp is not None) and (low <= open_trade.tp)
            if sl_hit and tp_hit:
                # Prioritize SL for shorts as well; gap-through fills at open if worse than stop
                self._close_trade(open_trade, index, max(open_px, float(open_trade.sl)))
                return True
            if sl_hit:
                # Short SL: price = max(open, stop)
                self._close_trade(open_trade, index, max(open_px, float(open_trade.sl)))
                return True
            if tp_hit:
                # Short TP (limit buy): fill at min(open, limit)
                self._close_trade(open_trade, index, min(open_px, float(open_trade.tp)))
                return True
        return False

    def on_bar(self, index: int) -> None:
        # 1) Process pending market orders at bar fill price (open or prev close)
        fill_px = self._price_at(index)
        # Track whether a new trade opened (unused; contingent processing deferred)
        # new_trade_opened flag intentionally removed to avoid linter warning
        if self._pending_orders:
            pending = self._pending_orders
            self._pending_orders = []
            for order in pending:
                direction, req_size, sl, tp, tag = order

                # Exclusive: close all open trades first (mirror reference semantics)
                if self.exclusive_orders and self._open_trade_ref is not None and self._open_trade_ref.is_open():
                    self._close_trade(self._open_trade_ref, index, fill_px)

                # After potential close above, recompute available cash-based notional for sizing

                # Use cash for available liquidity approximation (matches reference behavior closely
                # because reference uses equity - used margin; with exclusive orders there is at most one trade)
                margin_available = max(0.0, self.cash)

                # Determine integer units (reference-like sizing)
                entry_px = fill_px * (1 + (self.spread if direction > 0 else -self.spread))
                # Commission base price selection (reference uses unadjusted fill price)
                commission_base_price = fill_px if self._commission_on == "pre" else entry_px
                adjusted_price_plus_commission = entry_px + self._commission(req_size, commission_base_price)
                if 0 < req_size < 1:
                    available_notional = margin_available * self._leverage * abs(req_size)
                    units = int(available_notional // adjusted_price_plus_commission) if adjusted_price_plus_commission > 0 else 0
                    size_units = float(max(units, 0))
                    if size_units <= 0:
                        # Fractional order canceled due to insufficient margin
                        continue
                else:
                    size_units = float(int(abs(req_size)))
                    # Liquidity check mirroring reference: compare against adjusted_price_plus_commission computed with original order size
                    if (abs(int(size_units)) * adjusted_price_plus_commission) > (margin_available * self._leverage):
                        continue

                # Open trade at adjusted price; commission computed from unadjusted fill price
                # Reference cash accounting deducts only commission at open
                if direction == 1:
                    pass
                else:
                    pass
                entry_commission = self._commission(size_units, commission_base_price)
                self.cash -= entry_commission
                trade = Trade(
                    direction=direction,
                    entry_index=index,
                    entry_price=entry_px,
                    size=size_units,
                    sl=sl,
                    tp=tp,
                    tag=tag,
                    entry_commission=entry_commission,
                )
                self._trades.append(trade)
                self._open_trade_ref = trade
                # Allow contingent SL/TP to trigger within the same bar immediately after a fill
                self._process_contingent(index)

        # 2) After all orders, ensure contingent orders are processed for any remaining open trade
        if self._open_trade_ref is not None and self._open_trade_ref.is_open():
            self._process_contingent(index)

        # 3) update equity with current close
        self._update_equity(float(self._close_col[index]))

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


