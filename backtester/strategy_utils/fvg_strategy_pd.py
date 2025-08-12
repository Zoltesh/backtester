from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:
    # Traditional backtesting.py library
    from backtesting import Strategy as PDStrategy
except Exception as exc:  # pragma: no cover
    raise RuntimeError("The 'backtesting' package is required for FVGStrategyPD.") from exc


class FVGStrategyPD(PDStrategy):
    """Backtesting.py-compatible FVG retest strategy using the same logic as the polars version.

    Parameters are injected via Backtest.run(**params) like the polars engine.
    """

    # Defaults (overridable via Backtest.run(**params))
    fvg_threshold: float = 0.1  # percent threshold for gap_size_percent
    position_size: float = 0.9
    max_fvg_age: int = 45
    profit_target: float = 0.5
    loss_target: float = 0.01
    precomputed_fvgs: List[Dict[str, Any]] | None = None

    def init(self) -> None:
        if self.precomputed_fvgs is None:
            raise ValueError("Precomputed FVGs are required for FVGStrategyPD")

        # Cache numpy arrays for speed
        d = self.data
        self._high = np.asarray(d.High)
        self._low = np.asarray(d.Low)
        self._close = np.asarray(d.Close)

        # Pre-index FVGs by bar index; filter by threshold (% units)
        self.fvgs_by_bar: Dict[int, List[Dict[str, Any]]] = {}
        count = 0
        thr = float(self.fvg_threshold)
        for fvg in self.precomputed_fvgs:
            try:
                bar_index = int(fvg["bar_index"])  # type: ignore[index]
                gap_sz = float(fvg.get("gap_size_percent", 0.0))
            except Exception:
                continue
            if gap_sz < thr:
                continue
            self.fvgs_by_bar.setdefault(bar_index, []).append(fvg)
            count += 1

        self.traded_fvgs: set[tuple] = set()
        self.fvg_count = count

    def next(self) -> None:
        i = len(self.data) - 1
        if i <= 0:
            return
        if self.position:  # open aggregated position exists
            return

        current_price = float(self._close[i])
        current_high = float(self._high[i])
        current_low = float(self._low[i])

        start_bar = max(0, i - int(self.max_fvg_age))
        for bar_idx in range(start_bar, i):
            if bar_idx not in self.fvgs_by_bar:
                continue
            for fvg in self.fvgs_by_bar[bar_idx]:
                fvg_id = (fvg.get("bar_index"), fvg.get("is_bull"), fvg.get("midpoint"))
                if fvg_id in self.traded_fvgs:
                    continue

                max_price = float(fvg["max_price"])  # top of FVG zone
                min_price = float(fvg["min_price"])  # bottom of FVG zone
                in_fvg_zone = (current_low <= max_price) and (current_high >= min_price)
                if not in_fvg_zone:
                    continue

                is_bull = bool(fvg["is_bull"])  # True for bullish
                if is_bull:
                    sl_price = current_price * (1 - float(self.loss_target))
                    tp_price = current_price * (1 + float(self.profit_target))
                    self.buy(size=float(self.position_size), sl=sl_price, tp=tp_price)
                else:
                    sl_price = current_price * (1 + float(self.loss_target))
                    tp_price = current_price * (1 - float(self.profit_target))
                    self.sell(size=float(self.position_size), sl=sl_price, tp=tp_price)

                self.traded_fvgs.add(fvg_id)
                return


