from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np
import polars as pl


if TYPE_CHECKING:
    from .broker import Broker


class Strategy(metaclass=ABCMeta):
    """
    Polars-native base Strategy.
    Implement `init(self)` and `next(self)` in subclasses.

    Access OHLCV via `self.data` (Polars DataFrame) and current row index via
    `self.i`. Use `self.buy()`/`self.sell()` to place orders.
    """

    def __init__(self, broker: "Broker", data: pl.DataFrame, params: Dict[str, Any]):
        self._broker = broker
        self._data = data
        self._params = self._check_and_bind_params(params)
        self._indicators: List[np.ndarray] = []
        self.i: int = 0

    def _check_and_bind_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in params.items():
            if not hasattr(self, key):
                raise AttributeError(
                    f"Strategy '{self.__class__.__name__}' is missing parameter '{key}'."
                )
            setattr(self, key, value)
        return params

    @property
    def data(self) -> pl.DataFrame:
        return self._data

    # Indicator helper similar to backtesting.py's Strategy.I
    def I(self, func: Callable[..., np.ndarray], *args, **kwargs) -> np.ndarray:
        arr = np.asarray(func(*args, **kwargs), dtype=float)
        if arr.ndim == 1:
            self._indicators.append(arr)
            return arr
        # allow multi-column indicators, flatten list append
        for col in np.asarray(arr).T:
            self._indicators.append(np.asarray(col, dtype=float))
        return arr

    # Order placement proxies
    def buy(self, size: float = 1.0 - np.finfo(float).eps, sl: float | None = None, tp: float | None = None, tag: Optional[str] = None):
        return self._broker.market_order(direction=1, index=self.i, size=size, sl=sl, tp=tp, tag=tag)

    def sell(self, size: float = 1.0 - np.finfo(float).eps, sl: float | None = None, tp: float | None = None, tag: Optional[str] = None):
        return self._broker.market_order(direction=-1, index=self.i, size=size, sl=sl, tp=tp, tag=tag)

    @abstractmethod
    def init(self) -> None:  # to be implemented by user
        raise NotImplementedError

    @abstractmethod
    def next(self) -> None:  # to be implemented by user
        raise NotImplementedError


