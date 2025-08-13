from typing import List, Dict, Literal
import polars as pl

BiasType = Literal["bullish", "bearish", "neutral"]


class EMABiasMatrix:
    def __init__(self, dataframes: Dict[str, pl.DataFrame], ema_length: int = 20):
        self.dataframes = dataframes
        self.ema_length = ema_length
        self.biases: Dict[str, BiasType] = {}

    def compute(self) -> Dict[str, BiasType]:
        for tf, df in self.dataframes.items():
            self.biases[tf] = self._compute_single(df)
        return self.biases

    def compute_series(self) -> Dict[str, List[BiasType]]:
        """
        Computes bias over the full series for each timeframe.
        Returns a dictionary: { "5m": [...], "15m": [...], ... }
        """
        series_result: Dict[str, List[BiasType]] = {}

        for tf, df in self.dataframes.items():
            closes = df["close"].to_list()

            if len(closes) < self.ema_length + 2:
                series_result[tf] = ["neutral"] * len(closes)
                continue

            ema = self._ema(closes, self.ema_length)
            bias_series: List[BiasType] = ["neutral"] * len(closes)

            for i in range(1, len(ema)):
                price_idx = self.ema_length - 1 + i
                if price_idx >= len(closes):
                    break

                price = closes[price_idx]
                curr_ema = ema[i]
                prev_ema = ema[i - 1]

                if price > curr_ema and curr_ema > prev_ema:
                    bias: BiasType = "bullish"
                elif price < curr_ema and curr_ema < prev_ema:
                    bias = "bearish"
                else:
                    bias = "neutral"

                bias_series[price_idx] = bias

            series_result[tf] = bias_series

        return series_result

    def _compute_single(self, df: pl.DataFrame) -> BiasType:
        if df.height < self.ema_length + 2:
            return "neutral"

        closes = df["close"].to_list()
        ema = self._ema(closes, self.ema_length)

        if len(ema) < 2:
            return "neutral"

        price = closes[-1]
        curr_ema = ema[-1]
        prev_ema = ema[-2]

        if price > curr_ema and curr_ema > prev_ema:
            return "bullish"
        elif price < curr_ema and curr_ema < prev_ema:
            return "bearish"
        else:
            return "neutral"

    def _ema(self, values: List[float], length: int) -> List[float]:
        if len(values) < length:
            return []

        alpha = 2 / (length + 1)
        ema: List[float] = [sum(values[:length]) / length]

        for price in values[length:]:
            ema.append((price - ema[-1]) * alpha + ema[-1])

        return ema

    def all_bullish(self) -> bool:
        return all(bias == "bullish" for bias in self.biases.values())

    def all_bearish(self) -> bool:
        return all(bias == "bearish" for bias in self.biases.values())

    def any_bullish(self) -> bool:
        return any(bias == "bullish" for bias in self.biases.values())

    def any_bearish(self) -> bool:
        return any(bias == "bearish" for bias in self.biases.values())

__all__ = ["EMABiasMatrix", "BiasType"]
