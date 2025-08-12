from typing import Dict, Any, List
from backtester.strategy import Strategy


class FVGStrategy(Strategy):
    """Signals-only FVG retest strategy - generates entry signals when FVG zones are retested"""

    # Defaults (overridable via Backtest.run(**params))
    fvg_threshold: float = 0.1  # percent threshold for gap_size_percent
    position_size: float = 0.9
    max_fvg_age: int = 45
    profit_target: float = 0.5
    loss_target: float = 0.01
    precomputed_fvgs: List[Dict[str, Any]] | None = None

    def init(self) -> None:
        """Initialize column caches and pre-index FVGs for fast lookup."""
        if self.precomputed_fvgs is None:
            raise ValueError("Precomputed FVGs are required for FVGStrategy")

        # Cache OHLC columns as numpy arrays (engine uses 'Open/High/Low/Close')
        self._high = self.data.get_column("High").to_numpy()
        self._low = self.data.get_column("Low").to_numpy()
        self._close = self.data.get_column("Close").to_numpy()

        # Pre-index FVGs by bar index; accept dict-based FVGs from detector
        self.fvgs_by_bar: Dict[int, List[Dict[str, Any]]] = {}
        count = 0
        for fvg in self.precomputed_fvgs:
            try:
                bar_index = int(fvg["bar_index"])  # type: ignore[index]
                gap_sz = float(fvg.get("gap_size_percent", 0.0))
            except Exception:
                continue
            # filter by threshold (% units)
            if gap_sz < float(self.fvg_threshold):
                continue
            self.fvgs_by_bar.setdefault(bar_index, []).append(fvg)
            count += 1

        self.traded_fvgs: set[tuple] = set()
        self.fvg_count = count

    def _has_open_position(self) -> bool:
        t = getattr(self._broker, "_open_trade_ref", None)
        return bool(t is not None and t.is_open())

    def next(self) -> None:
        i = self.i
        if i <= 0:
            return
        if self._has_open_position():
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

                max_price = float(fvg["max_price"])  # top of FVG zone for bulls
                min_price = float(fvg["min_price"])  # bottom of FVG zone for bulls
                in_fvg_zone = (current_low <= max_price) and (current_high >= min_price)
                if not in_fvg_zone:
                    continue

                is_bull = bool(fvg["is_bull"])  # True for bullish
                if is_bull:
                    sl_price = current_price * (1 - float(self.loss_target))
                    tp_price = current_price * (1 + float(self.profit_target))
                    self.buy(size=float(self.position_size), sl=sl_price, tp=tp_price, tag="bullish_fvg_retest")
                else:
                    sl_price = current_price * (1 + float(self.loss_target))
                    tp_price = current_price * (1 - float(self.profit_target))
                    self.sell(size=float(self.position_size), sl=sl_price, tp=tp_price, tag="bearish_fvg_retest")

                self.traded_fvgs.add(fvg_id)
                return