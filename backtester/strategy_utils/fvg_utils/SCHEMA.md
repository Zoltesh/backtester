## FVG Backtested Master Dataset Schema

### Overview
The master CSV produced by `backtester/strategy_utils/fvg_backtested_csv_pipeline.py` combines executed trades from the polars-native backtester with the Fair Value Gaps (FVGs) that triggered them. One row represents one completed trade. If an FVG could not be matched for a trade (edge cases), the FVG-prefixed columns will be null.

Output path: `data/outputs/fvg_master_dataset_<symbol>_<timeframe>.csv`

### Conventions
- Timestamps are epoch milliseconds (UTC).
- Index columns (e.g., `EntryIdx`, `ExitIdx`, `SignalIdx`, `fvg_bar_index`) are zero-based bar indices into the base timeframe series.
- Percent columns use percentage units (e.g., `0.2` means `0.2%`).
- Dynamic EMA-bias columns use the naming pattern `ema_htf_<timeframe>` (e.g., `ema_htf_5m`) with values: `bullish`, `bearish`, or `neutral`.

### Columns
| Column | Type | Description |
|---|---|---|
| `TradeId` | string | Unique identifier for the trade (stable across entry/exit). Useful for linking to other outputs. |
| `EntryIdx` | int64 | Index of the bar where the trade was filled. Market orders fill at the next bar’s open (or previous close if `trade_on_close=True`). |
| `EntryTimestamp` | int64 (epoch ms) | Timestamp of the entry bar. Null if timestamp not available. |
| `ExitIdx` | int64 | Index of the bar where the trade was closed. `-1` if the trade is still open and `finalize_trades=False`. |
| `ExitTimestamp` | int64 (epoch ms) | Timestamp of the exit bar. Null if not available or still open. |
| `Direction` | int8 | `1` for long, `-1` for short. |
| `EntryPrice` | float64 | Executed entry price. |
| `ExitPrice` | float64 | Executed exit price. NaN if still open. |
| `Size` | float64 | Position size in units. For fractional orders `<1`, the engine resolves feasible integer units. |
| `SL` | float64 | Stop-loss level at entry; NaN if not set. |
| `TP` | float64 | Take-profit level at entry; NaN if not set. |
| `PnL` | float64 | Trade gross PnL excluding commissions: `(ExitPrice - EntryPrice) * Direction * Size` (0 if still open). |
| `Commission` | float64 | Total commission paid for the trade (entry + exit). |
| `Tag` | string | Strategy tag for the trade. For this pipeline typically `bullish_fvg_retest` or `bearish_fvg_retest`. |
| `SignalIdx` | int64 | Index of the signal bar that triggered the trade (usually `EntryIdx - 1`). Null if not applicable. |
| `SignalTimestamp` | int64 (epoch ms) | Timestamp of the signal bar. Null if not available. |
| `fvg_bar_index` | int64 | Index of the bar where the triggering FVG was detected. Null if no FVG matched. |
| `fvg_timestamp` | int64 (epoch ms) | Timestamp of the FVG detection bar. Null if no FVG matched or no timestamps. |
| `fvg_is_bull` | bool | `true` if FVG is bullish; `false` if bearish. Null if no FVG matched. |
| `fvg_max_price` | float64 | Upper price bound of the FVG zone (for bulls). For bears, mapped consistently from detection. Null if not matched. |
| `fvg_min_price` | float64 | Lower price bound of the FVG zone (for bulls). For bears, mapped consistently from detection. Null if not matched. |
| `fvg_midpoint` | float64 | Midpoint of the FVG zone. Null if not matched. |
| `fvg_gap_size_percent` | float64 | FVG gap size in percent units (`((max-min)/min) * 100`). Null if not matched. |
| `fvg_displacement_strength` | float64 | Displacement strength proxy at detection: `abs(close[i-1] - close[i-2]) * volume`. Null if not matched. |
| `ema_htf_<tf>` | string | Dynamic columns flattening HTF EMA bias at FVG detection bar for each available timeframe (e.g., `ema_htf_5m`, `ema_htf_15m`, `ema_htf_30m`, `ema_htf_1h`, `ema_htf_6h`). Values: `bullish` | `bearish` | `neutral`. Null if not matched or timeframe unavailable. |
| `symbol` | string | Base symbol for the dataset if present in source data; otherwise absent. |
| `timeframe` | string | Base timeframe for the dataset if present in source data; otherwise absent. |

### Notes
- FVG matching mirrors the strategy logic: for a trade at `EntryIdx = e`, the signal is evaluated at `i = e - 1`; the algorithm scans `[i - max_fvg_age, i)` and selects the first FVG whose zone intersects the signal bar range and whose direction matches the trade.
- EMA bias columns are generated only for the higher timeframes present in the input data and may vary across runs/datasets.

### Derived Trade & FVG Features (per row)

| Column | Type | Description |
|---|---|---|
| `PnL_net` | float64 | Net PnL after fees: `PnL - Commission`. |
| `y_win_gross` | int8 | 1 if `PnL > 0`, else 0. |
| `y_win_net` | int8 | 1 if `PnL_net > 0`, else 0. (default label) |
| `planned_R` | float64 | Risk-to-reward planned at entry. Long: `(TP - EntryPrice) / (EntryPrice - SL)`; Short: `(EntryPrice - TP) / (SL - EntryPrice)`. |
| `realized_R` | float64 | Realized R multiple using `ExitPrice` (same denominators as `planned_R`). Null if still open. |
| `holding_bars` | int64 | `ExitIdx - EntryIdx`. Null if still open. |
| `signal_to_entry_delay_bars` | int64 | `EntryIdx - SignalIdx`. Null if `SignalIdx` missing. |
| `fvg_age_bars` | int64 | `EntryIdx - fvg_bar_index` (staleness of FVG at entry). Null if no FVG matched. |
| `zone_width_abs` | float64 | `fvg_max_price - fvg_min_price`. Null if no FVG matched. |
| `zone_width_pct` | float64 | `zone_width_abs / fvg_midpoint`. Null if no FVG matched. |
| `matched_zone` | int8/bool | 1/true if `EntryPrice ∈ [fvg_min_price*(1-ε), fvg_max_price*(1+ε)]`; `ε` is `match_tolerance_pct` (default 0). |
| `entry_pos_in_zone` | float64 | Normalized entry location: `((EntryPrice - fvg_min_price) / zone_width_abs)` clipped to [0,1]. Null if no FVG matched. |
| `signed_entry_pos` | float64 | `(entry_pos_in_zone - 0.5) * Direction`. |

### HTF EMA Bias Aggregates

| Column | Type | Description |
|---|---|---|
| `htf_bull_count` | int8 | Count of `bullish` across all `ema_htf_*` columns present. |
| `htf_bear_count` | int8 | Count of `bearish` across all `ema_htf_*`. |
| `htf_neutral_count` | int8 | Count of `neutral` across all `ema_htf_*`. |
| `htf_bull_ratio` | float64 | `htf_bull_count / N`, where `N` is the number of `ema_htf_*` columns present. |
| `htf_bear_ratio` | float64 | `htf_bear_count / N`. |
| `htf_all_bullish` | int8/bool | 1/true if `htf_bull_count == N`. |
| `htf_all_bearish` | int8/bool | 1/true if `htf_bear_count == N`. |
| `htf_any_bullish` | int8/bool | 1/true if `htf_bull_count > 0`. |
| `htf_any_bearish` | int8/bool | 1/true if `htf_bear_count > 0`. |

### Time Features (from `EntryTimestamp`, UTC)

| Column | Type | Description |
|---|---|---|
| `hour_of_day` | int8 | 0–23. |
| `day_of_week` | int8 | 0–6 (Mon=0). |
| `is_weekend` | int8/bool | 1 if Sat/Sun, else 0. |
| `cv_month` | string | `YYYY-MM` (useful for GroupKFold/forward-chain splits). |

### Run Metadata (constant within a run)

| Column | Type | Description |
|---|---|---|
| `run_id` | string | Unique identifier for the run. |
| `strategy_name` | string | e.g., `FVGStrategy`. |
| `symbol` | string | Base symbol, if present. |
| `timeframe` | string | Base timeframe, if present. |
| `start_ts` | int64 | Backtest inclusive start (epoch ms). |
| `end_ts` | int64 | Backtest inclusive end (epoch ms). |
| `max_fvg_age` | int32 | Strategy parameter used. |
| `fvg_threshold` | float64 | Strategy parameter used. |
| `profit_target` | float64 | Strategy parameter used. |
| `loss_target` | float64 | Strategy parameter used. |
| `commission_rate` | float64 | Commission rate used. |
| `slippage_rate` | float64 | Slippage/spread rate used. |
| `position_sizing_rule` | string | e.g., `fixed_fraction`, `kelly_fraction`. |
| `position_fraction` | float64 | Fraction for fixed-fraction sizing. |
| `max_concurrent_trades` | int32 | If enforced. |
| `run_return_ann_pct` | float64 | Annualized return for the run. |
| `run_vol_ann_pct` | float64 | Annualized volatility for the run. |
| `run_cagr_pct` | float64 | CAGR for the run. |
| `run_sharpe` | float64 | Sharpe ratio. |
| `run_sortino` | float64 | Sortino ratio. |
| `run_calmar` | float64 | Calmar ratio. |
| `run_mdd_pct` | float64 | Max drawdown %. |
| `run_trades` | int32 | Number of trades. |
| `run_win_rate_pct` | float64 | Win rate. |
| `run_profit_factor` | float64 | Profit factor. |
| `run_sqn` | float64 | SQN. |
| `run_expectancy_pct` | float64 | Expectancy %. |
| `buy_hold_return_pct` | float64 | Buy & hold return during the run period. |

### Notes (leakage control)
- Derived features use only information available at or before `SignalIdx/EntryIdx`.
- Outcome columns (e.g., `ExitPrice`, `ExitIdx`, `PnL`, `PnL_net`, `realized_R`) are present for labels/evaluation but must be excluded from training features.
- If a given `ema_htf_*` column is absent in a dataset, exclude it from `N` and all associated counts/ratios.

