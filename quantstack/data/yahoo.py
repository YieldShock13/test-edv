from __future__ import annotations
import os
import pandas as pd
import yfinance as yf

from ..config import settings
from ..utils.io import save_df


def _extract_close(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    # Multi-index columns case (multi tickers): select Close or Adj Close
    if isinstance(df.columns, pd.MultiIndex):
        for level_name in ["Adj Close", "Close"]:
            if level_name in df.columns.get_level_values(0):
                out = df[level_name].copy()
                out.columns = [c if isinstance(c, str) else str(c) for c in out.columns]
                return out
        # Fallback: try last level name
        out = df.xs(df.columns.levels[0][-1], axis=1, level=0, drop_level=True)
        out.columns = [c if isinstance(c, str) else str(c) for c in out.columns]
        return out
    # Single ticker case: plain columns
    for col in ["Adj Close", "Close", "Close*", "close"]:
        if col in df.columns:
            s = df[col].rename(tickers[0])
            return s.to_frame()
    # Fallback to first numeric column
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return df[[numeric_cols[0]]].rename(columns={numeric_cols[0]: tickers[0]})
    return pd.DataFrame(index=df.index)


def fetch_prices(tickers: list[str], period: str = "max", interval: str = "1d") -> pd.DataFrame:
    data = yf.download(tickers=tickers, period=period, interval=interval, auto_adjust=True, progress=False, threads=True)
    if isinstance(data, pd.Series):
        px = data.to_frame(name=tickers[0])
    elif isinstance(data, pd.DataFrame):
        px = _extract_close(data, tickers)
    else:
        px = pd.DataFrame()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    return px


def cache_prices(tickers: list[str], cache_name: str) -> str:
    df = fetch_prices(tickers)
    path = os.path.join(settings.data_cache_dir, f"yahoo_{cache_name}.csv")
    save_df(df, path)
    return path