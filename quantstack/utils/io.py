import os
from typing import Optional
import pandas as pd


def save_df(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".parquet"):
        df.to_parquet(path)
    else:
        df.to_csv(path, index=True)


def load_df(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)