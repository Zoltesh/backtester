"""
XGBoost model for FVG backtesting
"""

import xgboost as xgb
import polars as pl


def load_data(path: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load data from a CSV file
    """
    df = pl.read_csv("data/outputs/fvg_master_dataset_SOL-USDC_1m.csv")
    return df

if __name__ == "__main__":
    df = load_data("data/outputs/fvg_master_dataset_SOL-USDC_1m.csv")
    print(df)