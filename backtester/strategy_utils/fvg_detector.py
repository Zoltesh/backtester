from typing import List, Dict, Optional, Any
import polars as pl

# Prefer relative import; fall back to absolute if executed differently
try:  # pragma: no cover - import fallback
    from .ema_bias_matrix import EMABiasMatrix
except Exception:  # pragma: no cover
    from backtester.strategy_utils.ema_bias_matrix import EMABiasMatrix

class FairValueGapDetector:
    """
    Detects bullish and bearish Fair Value Gaps (FVGs) in OHLCV price data.

    Attributes:
        threshold (float): Minimum percentage gap size (e.g. 0.1 for 0.1%).
    """

    def __init__(self, data: pl.DataFrame, threshold_percent: float = 0.0, htf_dataframes: Optional[Dict[str, pl.DataFrame]] = None, ema_length: int = 20):
        """
        Initialize the detector.

        Args:
            data: OHLCV DataFrame with columns: high, low, close, volume
            threshold_percent: Minimum size of gap as a percentage (0.1 means 0.1% gap).
            htf_dataframes: Optional higher timeframe dataframes for EMA bias calculation.
                          Format: {"5m": df_5m, "15m": df_15m, ...}
            ema_length: EMA length for bias calculation (default: 20).
        """
        # Validate required columns
        required_cols = ['high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in data: {missing_cols}")
            
        self.data = data
        self.threshold = threshold_percent / 100.0
        self.htf_dataframes = htf_dataframes
        self.ema_length = ema_length
        self._bias_matrix = None
        self._bias_series = None
        
        # Pre-compute EMA biases if higher timeframe data is provided
        if self.htf_dataframes:
            self._bias_matrix = EMABiasMatrix(self.htf_dataframes, self.ema_length)
            self._bias_series = self._bias_matrix.compute_series()

    def detect_fvg(self) -> List[Dict[str, Any]]:
        """
        Detect bullish and bearish Fair Value Gaps using vectorized Polars operations.

        Returns:
            List of dictionaries, one per detected FVG. Keys:
            max_price, min_price, midpoint, is_bull, bar_index,
            gap_size_percent, displacement_strength, ema_htf_bias
        """
        # Create shifted columns for FVG detection (vectorized)
        df_with_shifts = self.data.with_columns([
            pl.col('high').shift(2).alias('high_i_minus_2'),
            pl.col('low').shift(2).alias('low_i_minus_2'),
            pl.col('close').shift(1).alias('close_i_minus_1'),
            pl.col('close').shift(2).alias('close_i_minus_2'),
            pl.arange(0, pl.len()).alias('bar_index')
        ])
        
        # Define bullish and bearish conditions using vectorized expressions
        bull_mask = (
            (pl.col('low') > pl.col('high_i_minus_2')) &
            (pl.col('close_i_minus_1') > pl.col('high_i_minus_2')) &
            (((pl.col('low') - pl.col('high_i_minus_2')) / pl.col('high_i_minus_2')) > self.threshold)
        )
        
        bear_mask = (
            (pl.col('high') < pl.col('low_i_minus_2')) &
            (pl.col('close_i_minus_1') < pl.col('low_i_minus_2')) &
            (((pl.col('low_i_minus_2') - pl.col('high')) / pl.col('high')) > self.threshold)
        )
        
        # Calculate FVG properties using vectorized expressions
        fvg_df = df_with_shifts.with_columns([
            # Displacement strength (calculate first)
            ((pl.col('close_i_minus_1') - pl.col('close_i_minus_2')).abs() * pl.col('volume')).alias('displacement_strength')
        ]).with_columns([
            # Bullish FVG properties
            pl.when(bull_mask)
              .then(pl.col('low'))
              .when(bear_mask)
              .then(pl.col('low_i_minus_2'))
              .otherwise(None)
              .alias('max_price'),
            
            pl.when(bull_mask)
              .then(pl.col('high_i_minus_2'))
              .when(bear_mask)
              .then(pl.col('high'))
              .otherwise(None)
              .alias('min_price'),
            
            # FVG type
            pl.when(bull_mask)
              .then(True)
              .when(bear_mask)
              .then(False)
              .otherwise(None)
              .alias('is_bull')
        ]).filter(
            # Only keep rows where FVG conditions are met and bar_index >= 2
            (bull_mask | bear_mask) & (pl.col('bar_index') >= 2)
        ).with_columns([
            # Calculate derived fields
            ((pl.col('max_price') + pl.col('min_price')) / 2).alias('midpoint'),
            (((pl.col('max_price') - pl.col('min_price')) / pl.col('min_price')) * 100).alias('gap_size_percent')
        ])
        
        # Convert to plain dictionaries (no Pydantic model for performance)
        fvg_list: List[Dict[str, Any]] = []
        
        for row in fvg_df.iter_rows(named=True):
            bar_index = row['bar_index']
            ema_bias = self._get_ema_bias_at_bar(bar_index)
            
            fvg_list.append({
                'max_price': row['max_price'],
                'min_price': row['min_price'],
                'midpoint': row['midpoint'],
                'is_bull': row['is_bull'],
                'bar_index': bar_index,
                'gap_size_percent': row['gap_size_percent'],
                'displacement_strength': row['displacement_strength'],
                'ema_htf_bias': ema_bias,
            })
        
        return fvg_list

    def _get_ema_bias_at_bar(self, bar_index: int) -> Optional[Dict[str, str]]:
        """
        Get EMA bias for all higher timeframes at a specific bar index.
        
        Args:
            bar_index: The bar index where the FVG was detected.
            
        Returns:
            Dictionary of timeframe -> bias or None if no HTF data provided.
        """
        if not self._bias_series:
            return None
            
        bias_dict = {}
        for tf, bias_list in self._bias_series.items():
            if bar_index < len(bias_list):
                bias_dict[tf] = bias_list[bar_index]
            else:
                bias_dict[tf] = "neutral"  # Default if index out of range
                
        return bias_dict if bias_dict else None
