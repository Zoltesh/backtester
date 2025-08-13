"""
Run N randomized FVGStrategy backtests on the same OHLCV data with different params
and save all trades to a single CSV. Polars-native only.
"""

import asyncio
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import polars as pl
import uuid

from data_loader.data_loader import DataLoader
from backtester.strategy_utils.fvg_utils.fvg_detector import FairValueGapDetector
from backtester.strategy_utils.fvg_utils.fvg_strategy import FVGStrategy
from backtester.backtest import Backtest
from utils.generate_params import generate_random_params
from utils.normalize_ohlcv_columns import normalize_ohlcv_columns
from backtester.strategy_utils.fvg_utils.master_dataset import build_master_dataset
from backtester.strategy_utils.fvg_utils.data_prep import prepare_detector_inputs
from backtester.strategy_utils.fvg_utils.timeframe import select_base_and_htfs


async def main():
    # Generate random parameters
    grid_search_params = {
    'fvg_threshold': (0.05, 0.5, 0.01),
    'position_size': (0.05, 0.9, 0.01),
    'max_fvg_age': (2, 45, 1), # This may need to change based on timeframe
    'profit_target': (0.01, 0.7, 0.01),
    'loss_target': (0.005, 0.01, 0.001)
}
    params_list = generate_random_params(
        param_ranges=grid_search_params,
        n_params=100,
        seed=42,
    )
    
    # Load data
    symbol = "SOL-USDC"
    timeframes = ["1m", "5m", "15m", "30m", "1h", "6h"]
    start_date = "2025-01-01"
    end_date = "2025-03-31"

    base_dir = Path("data/ohlcv")
    loader = DataLoader(base_dir)
    data = await loader.load_ohlcv_between_dates([symbol], timeframes, start_date, end_date)

    # Auto-select base timeframe (smallest duration) and HTFs
    base_pl, htf_map, base_tf = select_base_and_htfs(data)

    base_pl_bt = normalize_ohlcv_columns(base_pl)
    det_pl, clean_htf = prepare_detector_inputs(base_pl_bt, htf_map)

    # Detect all candidate FVGs once for the dataset (low threshold to avoid missing)
    fvgs = FairValueGapDetector(det_pl, threshold_percent=0.0, htf_dataframes=clean_htf or None, ema_length=20).detect_fvg()

    # Run each parameter set on the same data and precomputed FVGs
    all_master_dfs: List[pl.DataFrame] = []
    bt = Backtest(
        data=base_pl_bt.select([c for c in ["timestamp", "Open", "High", "Low", "Close"] if c in base_pl_bt.columns]),
        strategy=FVGStrategy,
        cash=10_000.0,
        commission=0.002,
    )

    # Helper to fetch symbol/timeframe for stamping
    def _first_str(df_: pl.DataFrame, col: str) -> str | None:
        if col in df_.columns and df_.height:
            try:
                return str(df_.select(pl.col(col).first()).item())
            except Exception:
                try:
                    return str(df_.get_column(col)[0])
                except Exception:
                    return None
        return None

    # symbol/timeframe available if present in base data

    highest_return = 0
    for idx, params in enumerate(params_list):

        res = bt.run(
            fvg_threshold=float(params['fvg_threshold']),
            position_size=float(params['position_size']),
            max_fvg_age=int(params['max_fvg_age']),
            profit_target=float(params['profit_target']),
            loss_target=float(params['loss_target']),
            precomputed_fvgs=fvgs,
        )
        trades = res.trades

        # Build run metadata snapshot (align with SCHEMA.md)
        commission_rate = 0.002
        slippage_rate = 0.0  # using spread=0.0 in engine
        position_sizing_rule = "fixed_fraction"
        position_fraction = float(params['position_size'])
        max_concurrent_trades = 1
        # Start/End timestamps (inclusive) from base
        if "timestamp" in base_pl_bt.columns and base_pl_bt.height:
            start_ts = int(base_pl_bt[0, "timestamp"])  # type: ignore[index]
            end_ts = int(base_pl_bt[-1, "timestamp"])  # type: ignore[index]
        else:
            start_ts = 0
            end_ts = 0

        stats = res.stats
        run_meta: Dict[str, Any] = {
            "run_id": uuid.uuid4().hex,
            "strategy_name": "FVGStrategy",
            "symbol": None,
            "timeframe": None,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "max_fvg_age": int(params['max_fvg_age']),
            "fvg_threshold": float(params['fvg_threshold']),
            "profit_target": float(params['profit_target']),
            "loss_target": float(params['loss_target']),
            "commission_rate": commission_rate,
            "slippage_rate": slippage_rate,
            "position_sizing_rule": position_sizing_rule,
            "position_fraction": position_fraction,
            "max_concurrent_trades": max_concurrent_trades,
            "run_return_ann_pct": float(stats.get("Return (Ann.) [%]", 0.0)),
            "run_vol_ann_pct": float(stats.get("Volatility (Ann.) [%]", 0.0)),
            "run_cagr_pct": float(stats.get("CAGR [%]", 0.0)),
            "run_sharpe": float(stats.get("Sharpe Ratio", 0.0)),
            "run_sortino": float(stats.get("Sortino Ratio", 0.0)),
            "run_calmar": float(stats.get("Calmar Ratio", 0.0)),
            "run_mdd_pct": float(stats.get("Max. Drawdown [%]", 0.0)),
            "run_trades": int(stats.get("# Trades", 0)),
            "run_win_rate_pct": float(stats.get("Win Rate [%]", 0.0)),
            "run_profit_factor": float(stats.get("Profit Factor", 0.0)),
            "run_sqn": float(stats.get("SQN", 0.0)),
            "run_expectancy_pct": float(stats.get("Expectancy [%]", 0.0)),
            "buy_hold_return_pct": float(stats.get("Buy & Hold Return [%]", 0.0)),
        }

        # Fill symbol/timeframe in run_meta if present
        def _first_str(df_: pl.DataFrame, col: str) -> Optional[str]:
            if col in df_.columns and df_.height:
                try:
                    return str(df_.select(pl.col(col).first()).item())
                except Exception:
                    try:
                        return str(df_.get_column(col)[0])
                    except Exception:
                        return None
            return None

        run_meta["symbol"] = _first_str(base_pl_bt, "symbol")
        run_meta["timeframe"] = _first_str(base_pl_bt, "timeframe")

        master_df = build_master_dataset(
            base_pl_bt=base_pl_bt,
            fvgs=fvgs,
            trades_df=trades,
            fvg_threshold=float(params['fvg_threshold']),
            max_fvg_age=int(params['max_fvg_age']),
            match_tolerance_pct=0.0,
            run_meta=run_meta,
        )

        # Add run index for traceability across random params
        master_df = master_df.with_columns([
            pl.lit(idx).alias("run_index"),
        ])

        all_master_dfs.append(master_df)
        
        # Print stats
        # Stats to include:
        stats_list = ["Equity Final [$]",
                      "Equity Peak [$]",
                      "Return [%]",
                      "Buy & Hold Return [%]",
                      "Alpha [%]",
                      "Beta",
                      "Sharpe Ratio",
                      "Sortino Ratio",
                      "Calmar Ratio",
                      "CAGR [%]",
                      "SQN",
                      "Kelly Criterion",
                      "Win Rate [%]"]
        
        print(f"\nRun {idx} stats:")
        for k, v in res.stats.items():
            if k in stats_list:
                print(f"{k}: {v}")
        equity_return = res.stats["Return [%]"]
        if equity_return > highest_return:
            highest_return = equity_return

    print(f"Highest Return: {highest_return}")

    out_dir = Path("data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use detected base timeframe in filename if available
    suffix_tf = base_tf or "base"
    out_path = out_dir / f"fvg_bulk_master_dataset_{symbol}_{suffix_tf}.csv"
    output_df = pl.concat(all_master_dfs) if all_master_dfs else pl.DataFrame({})
    output_df.write_csv(out_path)
    print(f"Saved {output_df.height} rows to {out_path}")

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")