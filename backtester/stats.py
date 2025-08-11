from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import polars as pl
from datetime import timedelta


def _drawdown_and_durations(
    equity: np.ndarray,
    times: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Compute drawdown series and durations like backtesting.py.

    Vectorized where possible for speed, preserving identical semantics.

    Returns
    -------
    dd : np.ndarray
        Drawdown fraction per point.
    dd_durations_ms : np.ndarray
        Per-point duration since last peak, in ms.
    max_dd_dur_ms : float
        Maximum drawdown duration in ms.
    avg_dd_dur_ms : float
        Average drawdown duration in ms (mean of per-point durations).
    avg_dd_peak : float
        Average of per-episode max drawdown peaks (fraction).
    """
    peak = np.maximum.accumulate(equity)
    denom = np.where(peak == 0, 1.0, peak)
    dd = 1.0 - equity / denom
    if times.size == 0:
        return dd, np.zeros_like(dd), 0.0, 0.0, 0.0

    # Vectorized duration since last peak: compute last_peak_index prefix
    is_new_peak = equity >= peak  # True where we are at the running peak
    # last_peak_idx[i] = index of last True in is_new_peak up to i
    last_peak_idx = np.where(is_new_peak, np.arange(len(equity)), -1)
    # Propagate last seen peak index forward
    np.maximum.accumulate(last_peak_idx, out=last_peak_idx)
    dd_durations_ms = (times - times[np.maximum(last_peak_idx, 0)]).astype(np.float64)
    # For non-drawdown points, set duration to 0
    dd_durations_ms[dd <= 0] = 0.0

    # Episode peaks: find indices where dd==0, ensure last index is included
    zero_points = np.flatnonzero(dd == 0)
    if zero_points.size == 0 or zero_points[-1] != len(dd) - 1:
        zero_points = np.append(zero_points, len(dd) - 1)
    peaks = []
    prev = 0
    for z in zero_points:
        if z > prev + 1:
            peaks.append(np.max(dd[prev : z + 1]))
        prev = z
    max_dd_dur_ms = float(np.max(dd_durations_ms)) if dd_durations_ms.size else 0.0
    avg_dd_dur_ms = float(np.mean(dd_durations_ms)) if dd_durations_ms.size else 0.0
    avg_dd_peak = float(np.mean(peaks)) if peaks else 0.0
    return dd, dd_durations_ms, max_dd_dur_ms, avg_dd_dur_ms, avg_dd_peak


def _geometric_mean(returns: np.ndarray) -> float:
    # Match backtesting.py geometric_mean: fill NaN with 0, use gross returns, abort if any <= 0
    gross = np.nan_to_num(returns, nan=0.0) + 1.0
    if np.any(gross <= 0):
        return 0.0
    return float(np.exp(np.log(gross).sum() / (len(gross) or np.nan)) - 1.0)


def _resample_daily_last(times_ms: np.ndarray, equity: np.ndarray) -> np.ndarray:
    # Group by day and take last equity using vectorized boundary detection
    if times_ms.size == 0:
        return np.array([], dtype=float)
    days = (times_ms // (24 * 3600 * 1000)).astype(np.int64)
    is_last_of_day = np.r_[days[1:] != days[:-1], True]
    return equity[is_last_of_day].astype(float)


def compute_stats(
    data: pl.DataFrame,
    equity: pl.Series,
    trades: pl.DataFrame,
    *,
    first_trading_bar: int | None = None,
    strategy: Any | None = None,
    annual_days: int | str = "auto",
) -> Dict[str, Any]:
    # Extract arrays
    eq = equity.to_numpy()
    close = data.get_column("Close").to_numpy()
    if "timestamp" in data.columns:
        times = data.get_column("timestamp").to_numpy()
    else:
        # fallback synthetic 1-minute spacing
        times = (np.arange(len(data), dtype=np.int64) * 60_000)

    start_time = times[0] if times.size else 0
    end_time = times[-1] if times.size else 0

    # Drawdown and durations
    dd, dd_durations_ms, max_dd_dur_ms, avg_dd_dur_ms, avg_dd_peak = _drawdown_and_durations(eq, times)
    max_dd = float(-np.nan_to_num(np.max(dd)))  # negative value as in reference

    # Equity curve df-like for internal use
    equity_curve = {
        "Equity": eq,
        "DrawdownPct": dd,
        "DrawdownDuration": dd_durations_ms,
    }

    # Buy & Hold return (from first_trading_bar)
    ftb = int(first_trading_bar or 0)
    bh_ret = 0.0
    if len(close) > ftb and close[ftb] != 0:
        bh_ret = float((close[-1] - close[ftb]) / close[ftb])

    # Daily resample for annualized stats (mirror backtesting.py)
    daily_last = _resample_daily_last(times, eq)
    day_rets = (
        np.diff(daily_last) / np.where(daily_last[:-1] == 0, np.nan, daily_last[:-1])
        if len(daily_last) > 1
        else np.array([], dtype=float)
    )
    day_rets = np.nan_to_num(day_rets, nan=0.0)
    gmean_day = _geometric_mean(day_rets) if day_rets.size else 0.0

    # annual trading days estimate (auto or override)
    if isinstance(annual_days, int):
        annual_trading_days = int(annual_days)
    else:
        weekdays = ((times // (24 * 3600 * 1000) + 4) % 7).astype(int)
        have_weekends = (np.mean((weekdays == 5) | (weekdays == 6)) > (2 / 7) * 0.6)
        annual_trading_days = 365 if have_weekends else 252
    annualized_return = (1 + gmean_day) ** annual_trading_days - 1
    volatility_ann = 0.0
    if day_rets.size:
        var = float(np.var(day_rets, ddof=1)) if day_rets.size > 1 else float(np.var(day_rets, ddof=0))
        volatility_ann = float(
            np.sqrt((var + (1 + gmean_day) ** 2) ** annual_trading_days - (1 + gmean_day) ** (2 * annual_trading_days))
        )

    # Sharpe/Sortino/Calmar
    sharpe = (annualized_return * 100) / (volatility_ann * 100 or np.nan) if volatility_ann else np.nan
    downside = np.sqrt(np.mean(np.clip(day_rets, -np.inf, 0) ** 2)) if day_rets.size else 0.0
    sortino = (annualized_return) / (downside * np.sqrt(annual_trading_days) or np.nan) if downside else np.nan
    calmar = (annualized_return) / (-max_dd or np.nan) if max_dd else np.nan

    # Trades stats
    # Use only CLOSED trades for trade-based stats to mirror reference engine
    n_trades = 0.0
    pnl = np.array([], dtype=float)
    size_abs = np.array([], dtype=float)
    entry_price = np.array([], dtype=float)
    commission_arr = np.array([], dtype=float)
    if trades.height:
        exit_idx_arr = trades["ExitIdx"].to_numpy()
        closed_mask = exit_idx_arr >= 0
        n_trades = float(np.sum(closed_mask))
        if n_trades > 0:
            pnl = trades["PnL"].to_numpy()[closed_mask]
            size_abs = np.abs(trades["Size"].to_numpy()[closed_mask])
            entry_price = trades["EntryPrice"].to_numpy()[closed_mask]
            if "Commission" in trades.columns:
                commission_arr = trades["Commission"].to_numpy()[closed_mask]
            else:
                commission_arr = np.zeros_like(pnl)
    net_pnl = pnl - commission_arr
    # Trade returns as fraction using invested notional abs(size)*entry_price
    if n_trades > 0:
        denom = size_abs * entry_price
        with np.errstate(divide='ignore', invalid='ignore'):
            trade_returns = np.where(denom != 0, pnl / denom, 0.0)
    else:
        trade_returns = np.array([], dtype=float)

    win_rate = float(np.mean(pnl > 0)) * 100 if pnl.size else 0.0
    best_trade = float(np.max(trade_returns)) * 100 if trade_returns.size else 0.0
    worst_trade = float(np.min(trade_returns)) * 100 if trade_returns.size else 0.0
    avg_trade = float(_geometric_mean(trade_returns)) * 100 if trade_returns.size else 0.0
    # Profit Factor from trade returns (gross), which matches reference behavior better than net PnL
    if 'trade_returns' in locals() and trade_returns.size:
        pos = trade_returns > 0
        neg = trade_returns < 0
        sum_pos = float(np.sum(trade_returns[pos])) if np.any(pos) else 0.0
        sum_neg = float(-np.sum(trade_returns[neg])) if np.any(neg) else 0.0
        profit_factor = sum_pos / (sum_neg or np.nan)
    else:
        profit_factor = np.nan
    expectancy = float(np.mean(trade_returns)) * 100 if trade_returns.size else 0.0
    # SQN on monetary PnL (matches reference better than using returns)
    sqn = (np.sqrt(n_trades) * (np.mean(pnl) / (np.std(pnl, ddof=0) or np.nan))) if pnl.size > 1 else np.nan
    # Kelly on monetary PnL
    if pnl.size:
        win_mask = pnl > 0
        win_rate_k = float(np.mean(win_mask)) if pnl.size else np.nan
        avg_win = float(np.mean(pnl[win_mask])) if np.any(win_mask) else np.nan
        avg_loss = -float(np.mean(pnl[~win_mask])) if np.any(~win_mask) else np.nan
        kelly = win_rate_k - (1 - win_rate_k) / (avg_win / (avg_loss or np.nan)) if (avg_win and avg_loss) else np.nan
    else:
        kelly = np.nan

    # Durations
    start_val = ms_to_datetime_str(start_time)
    end_val = ms_to_datetime_str(end_time)
    duration_td = ms_to_timedelta(int(end_time - start_time))
    # Round durations to data period resolution to mirror reference engine
    diffs = np.diff(times) if times.size > 1 else np.array([60_000], dtype=np.int64)
    # Use median to be robust to occasional gaps
    period_ms = int(np.median(diffs)) if diffs.size else 60_000
    def ceil_ms_to_period(ms: float, period: int) -> int:
        if period <= 0:
            return int(ms)
        ms_int = int(ms)
        return int(((ms_int + period - 1) // period) * period)
    max_dd_td = ms_to_timedelta(ceil_ms_to_period(max_dd_dur_ms, period_ms))
    avg_dd_td = ms_to_timedelta(ceil_ms_to_period(avg_dd_dur_ms, period_ms))

    stats: Dict[str, Any] = {}
    stats["Start"] = start_val
    stats["End"] = end_val
    stats["Duration"] = duration_td
    # Exposure: proportion of bars with open position
    exposure = 0.0
    if trades.height:
        have_pos = np.zeros(len(eq), dtype=int)
        for row in trades.iter_rows():
            eidx = int(row[0])  # EntryIdx
            xidx = int(row[1]) if row[1] is not None and row[1] != -1 else len(eq) - 1
            have_pos[eidx : xidx + 1] = 1
        exposure = float(np.mean(have_pos) * 100)
    stats["Exposure Time [%]"] = exposure
    stats["Equity Final [$]"] = float(eq[-1]) if eq.size else 0.0
    stats["Equity Peak [$]"] = float(np.max(eq)) if eq.size else 0.0
    stats["Return [%]"] = float(((eq[-1] - eq[0]) / (eq[0] or 1.0)) * 100) if eq.size else 0.0
    stats["Buy & Hold Return [%]"] = bh_ret * 100
    stats["Return (Ann.) [%]"] = annualized_return * 100
    stats["Volatility (Ann.) [%]"] = volatility_ann * 100
    # CAGR approximated via total duration in years
    years = (end_time - start_time) / (365 * 24 * 3600 * 1000) if end_time > start_time else 0
    stats["CAGR [%]"] = (((stats["Equity Final [$]"] / (eq[0] or 1.0)) ** (1 / years) - 1) * 100) if years else np.nan
    stats["Sharpe Ratio"] = float(sharpe) if not np.isnan(sharpe) else 0.0
    stats["Sortino Ratio"] = float(sortino) if not np.isnan(sortino) else 0.0
    stats["Calmar Ratio"] = float(calmar) if not np.isnan(calmar) else 0.0
    stats["Max. Drawdown [%]"] = -float(np.max(dd)) * 100 if dd.size else 0.0
    stats["Avg. Drawdown [%]"] = -avg_dd_peak * 100
    stats["Max. Drawdown Duration"] = max_dd_td
    stats["Avg. Drawdown Duration"] = avg_dd_td
    stats["# Trades"] = float(n_trades)
    stats["Win Rate [%]"] = float(win_rate)
    stats["Best Trade [%]"] = float(best_trade)
    stats["Worst Trade [%]"] = float(worst_trade)
    stats["Avg. Trade [%]"] = float(avg_trade)
    # Trade durations require timestamps; approximate using index if missing
    if trades.height:
        eidx_arr = trades["EntryIdx"].to_numpy()
        xidx_arr = trades["ExitIdx"].to_numpy()
        mask = xidx_arr >= 0
        if np.any(mask):
            eidx = eidx_arr[mask]
            xidx = xidx_arr[mask]
            durations_ms = times[xidx] - times[eidx]
        else:
            durations_ms = np.array([], dtype=np.int64)
        if durations_ms.size:
            max_td_ms = ceil_ms_to_period(float(np.max(durations_ms)), period_ms)
            avg_td_ms = ceil_ms_to_period(float(np.mean(durations_ms)), period_ms)
            stats["Max. Trade Duration"] = ms_to_timedelta(max_td_ms)
            stats["Avg. Trade Duration"] = ms_to_timedelta(avg_td_ms)
        else:
            stats["Max. Trade Duration"] = timedelta(0)
            stats["Avg. Trade Duration"] = timedelta(0)
    else:
        stats["Max. Trade Duration"] = timedelta(0)
        stats["Avg. Trade Duration"] = timedelta(0)
    stats["Profit Factor"] = float(profit_factor) if not np.isnan(profit_factor) else 0.0
    stats["Expectancy [%]"] = float(expectancy)
    stats["SQN"] = float(sqn) if not np.isnan(sqn) else 0.0
    stats["Kelly Criterion"] = float(kelly) if not np.isnan(kelly) else 0.0

    # Beta/Alpha relative to market (Close)
    equity_log_returns = np.log(eq[1:] / np.where(eq[:-1] == 0, np.nan, eq[:-1]))
    market_log_returns = np.log(close[1:] / np.where(close[:-1] == 0, np.nan, close[:-1]))
    beta = np.nan
    if equity_log_returns.size > 1 and market_log_returns.size > 1:
        cov_matrix = np.cov(equity_log_returns, market_log_returns)
        if cov_matrix.shape == (2, 2) and cov_matrix[1, 1] != 0:
            beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    stats["Beta"] = float(beta) if not np.isnan(beta) else beta
    stats["Alpha [%]"] = stats["Return [%]"] - 0.0 - (stats["Buy & Hold Return [%]"] * (stats["Beta"] if stats["Beta"] is not None and not np.isnan(stats["Beta"]) else 0.0))

    # Internal
    stats["_strategy"] = strategy
    stats["_equity_curve"] = equity_curve
    stats["_trades"] = trades
    return stats


def ms_to_datetime_str(ms: int) -> str:
    try:
        seconds = int(ms) // 1000
        return np.datetime_as_string(np.datetime64(seconds, 's'))
    except Exception:
        return "1970-01-01 00:00:00"


def ms_to_timedelta(ms: int) -> timedelta:
    return timedelta(milliseconds=int(ms))


